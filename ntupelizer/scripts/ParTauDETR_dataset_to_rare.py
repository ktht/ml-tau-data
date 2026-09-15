"""Add a rare-decay-mode label to the tau daughter (ParTauDETR) dataset.

Usage:
    ParTauDETR_dataset_to_rare.py [-i <input_dir>] [-o <output_dir>]
                                  [--batch-size <n>] [--row-group-size <n>]

Options:
    -i <input_dir>          Directory of tau daughter .parquet files, as written
                            by the DecayProductNtupelizer workflow.
                            [default: /scratch/persistent/laurits/ml-tau/20260818_tauDaughterDataset/]
    -o <output_dir>         Where the labelled copies are written, under the same
                            filenames as the inputs.
                            [default: /scratch/persistent/laurits/ml-tau/20260824_rareDecaysDataset/]
    --batch-size <n>        Rows held in memory at a time. Does not affect the
                            output layout, only peak memory. [default: 100000]
    --row-group-size <n>    Rows per parquet row group in the output; matches the
                            inputs written by merge_files.py. [default: 1024]

This is a 1:1 file transform: one output per input, same filename, same rows
(minus the electron-daughter cut), same row-group size.  Each jet gains a
`gen_jet_tau_decay_mode_rare` column holding the id of its tau decay mode, taken
from the twelve most common modes, or 15 ("other") if it matches none of them.
See the Misc section of the README for the list.

Only that one column is added: the input table is carried through as Arrow, so
every other column keeps the exact type it had in the input rather than being
round-tripped through awkward.

Background (qq) files are skipped -- only the signal sample carries a
meaningful gen-level decay mode.
"""

import glob
import os

import awkward as ak
import pyarrow as pa
import pyarrow.parquet as pq
from docopt import docopt


def get_decay_mode_id(daughter_pdgs):
    # keys = np.unique(ak.flatten(abs(arr.gen_jet_tau_vis_daughter_pdgs)))
    keys = [
        22,
        111,
        130,
        211,
        221,
        223,
        310,
        311,
        321,
        323,
    ]  # Should get the same result as above, but this is just a failsafe.

    # The 12 most common tau decays, in descending branching fraction, stopping
    # just before the first mode with a photon in it (gamma pi0 pi, ~2.8e-3).
    # The class id is therefore the frequency rank.
    #
    # Note which kaon code each row wants.  The neutral kaon from a tau decay is
    # written as 311 (K0) in this sample, not 310 (K0_S) -- 5309 vs 503 daughters
    # over 200k signal jets -- and the single-kaon modes are the *charged* kaon,
    # 321.  Rows 6, 7, 9 and 10 previously asked for 310 and so never matched
    # anything: 7 and 9 were charge-forbidden as written (K0_S / K0_S pi0 from a
    # charged tau), and 6 and 10 looked for the wrong neutral kaon code.
    targets = ak.Array(
        [
            # 22 111 130 211 221 223 310 311 321 323
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # 0: pi + pi0
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 1: pi
            [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],  # 2: 3pi
            [0, 2, 0, 1, 0, 0, 0, 0, 0, 0],  # 3: pi + 2pi0
            [0, 1, 0, 3, 0, 0, 0, 0, 0, 0],  # 4: 3pi + pi0
            [0, 3, 0, 1, 0, 0, 0, 0, 0, 0],  # 5: pi + 3pi0
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # 6: pi + K0
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 7: K
            [0, 2, 0, 3, 0, 0, 0, 0, 0, 0],  # 8: 3pi + 2pi0
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],  # 9: K + pi0
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],  # 10: pi + pi0 + K0
            [0, 0, 0, 2, 0, 0, 0, 0, 1, 0],  # 11: 2pi + K
        ]
    )

    counts = ak.zip({f"n_{k}": ak.sum(abs(daughter_pdgs) == k, axis=1) for k in keys})
    signature = ak.zeros_like(counts.n_22)
    for k in keys:
        signature = signature * 10 + counts[f"n_{k}"]

    # Encode the target configurations using the exact same scheme.
    target_signature = ak.zeros_like(targets[:, 0])

    for i in range(len(keys)):
        target_signature = target_signature * 10 + targets[:, i]

    # Match each jet against the 12 target signatures.
    matches = signature[:, None] == target_signature[None, :]

    # ID of matching target.
    class_id = ak.argmax(matches, axis=1, mask_identity=False)

    # No match -> Other = -1
    class_id = ak.where(
        ak.any(matches, axis=1),
        class_id,
        15,
    )
    return class_id


DAUGHTER_PDG_COLUMN = "gen_jet_tau_vis_daughter_pdgs"
LABEL_COLUMN = "gen_jet_tau_decay_mode_rare"


def label_batch(table):
    """Drop electron-daughter jets and append the decay-mode label column.

    Works on the Arrow table directly so that every pre-existing column keeps
    the type it had in the input; only the daughter PDGs go through awkward,
    because the counting needs a jagged array.
    """
    daughters = ak.from_arrow(table.column(DAUGHTER_PDG_COLUMN))
    # Nearly a no-op in practice: the ntupelizer already removes leptonic taus
    # (decaymode 16), so few electron daughters survive to here.
    keep = ak.to_numpy(ak.sum(abs(daughters) == 11, axis=1) == 0)
    table = table.filter(keep)

    daughters = ak.from_arrow(table.column(DAUGHTER_PDG_COLUMN))
    class_id = ak.to_numpy(get_decay_mode_id(daughters))
    return table.append_column(LABEL_COLUMN, pa.array(class_id, type=pa.int64()))


def label_file(input_path, output_path, batch_size, row_group_size):
    """Write one labelled copy of one input file, streaming a batch at a time."""
    reader = pq.ParquetFile(input_path)
    writer = None
    n_rows = 0
    try:
        for batch in reader.iter_batches(batch_size=batch_size):
            table = label_batch(pa.Table.from_batches([batch]))
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="zstd",
                    compression_level=6,
                )
            writer.write_table(table, row_group_size=row_group_size)
            n_rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    return n_rows


def label_dataset(input_dir, output_dir, batch_size, row_group_size):
    os.makedirs(output_dir, exist_ok=True)
    inputs = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not inputs:
        raise FileNotFoundError(f"No .parquet files found in {input_dir}")

    for path in inputs:
        basename = os.path.basename(path)
        if "qq" in basename:
            continue
        out_path = os.path.join(output_dir, basename)
        if os.path.abspath(out_path) == os.path.abspath(path):
            raise ValueError(
                f"Output would overwrite the input: {path}. "
                "Pick an -o directory different from -i."
            )
        print(f"Processing {basename}")
        n_rows = label_file(path, out_path, batch_size, row_group_size)
        print(f"  -> {out_path} ({n_rows} jets)")


def main():
    args = docopt(__doc__)
    label_dataset(
        args["-i"],
        args["-o"],
        int(args["--batch-size"]),
        int(args["--row-group-size"]),
    )


if __name__ == "__main__":
    main()
