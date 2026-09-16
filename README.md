# ml-tau-data

Data processing pipeline for the machine-learned hadronically-decaying tau lepton reconstruction and identification project. Takes EDM4HEP/PodioROOT simulation files and produces flat Parquet ntuples ready for ML training.

## Overview

The workflow is managed by **Snakemake** and consists of four stages:

1. **ntupelize** — process each input ROOT file into a per-file Parquet, then concatenate the batch into one. One SLURM job handles `files_per_job` ROOT files (default 20), processed sequentially
2. **weights** — accumulate the `(p, theta)` reweighting matrices. Only the `gen_jet_p4` column is read, so this runs directly on the ntupelized batches, before any merging
3. **merge\_and\_split** — stream the ntupelized batches of each dataset into `train` / `test` chunk files of `chunk_size` events, weighting them as they are filled
4. **validation** — produce summary plots comparing signal and background distributions

Stage 3 reads each event once and writes it once. It pulls one row group at a
time from the batches, assigns each event to train or test, and appends it to
that split's fill buffer; a buffer writes an output file as soon as it holds
`chunk_size` events, and when an input file runs out the next one is opened and
keeps filling the file that is still open. So a split is never materialised as
one big file, never as an unweighted copy, and never as a merged intermediate,
and memory is bounded by one output file (~0.75 GB at 100k jets) rather than by
the size of the sample.

Two consequences of streaming are worth knowing:

- **The shuffle is local.** Events are mixed within a fill buffer, i.e. within
  an output file, not across the whole sample. The inputs are independent
  simulation runs of the same process, so this only matters if the training
  loader relies on the file order itself being random. The train/test
  assignment *is* global: it is drawn so that the test events are a uniformly
  random subset of the whole sample and the split sizes land exactly on
  `train_frac`, which is what the parquet-footer pre-scan at the start is for.
- **The weight histograms cover each whole sample**, not only its train split,
  since they are built before the split exists. The split is random, so the two
  distributions are statistically the same; restricting them to the train split
  would mean materialising that split first.

Final outputs land in `output_dir` (configured in `ntupelizer/config/workflow.yaml`):

```
<output_dir>/
  z_train_00000.parquet  # signal train (weighted), <= chunk_size events
  z_train_00001.parquet  # ... as many files as the split needs
  z_test_00000.parquet   # signal test  (weighted)
  qq_train_00000.parquet # background train (weighted)
  qq_test_00000.parquet  # background test  (weighted)
  weights/               # weight matrices and bin edges
  validation/            # validation plots
  .markers/              # Snakemake completion markers (see below)
```

Every split is a numbered series of files starting at `_00000`. The index is
zero-padded to five digits, so a plain lexicographic listing is already in chunk
order. Set `chunk_size` in `workflow.yaml` to change the events per file
(default 100000).

`row_group_size` (default 1024) sets the parquet row group size inside those
files — about 98 row groups per file at the default `chunk_size`. Row groups are
the unit of a partial read, so small ones keep the dataloader's reads cheap. The
trade-off is bulk reads: measured on a 100k-jet file, 1024 rows per group reads
in full 5.4× slower than 20000 rows per group (1.13 s vs 0.21 s), is ~7% bigger
on disk and carries a 375 KB rather than 25 KB footer. Raise it for a dataset
that is mostly read end to end.

Because the number of chunks in a split is only known once the merge has counted
the events, the chunked stages cannot list their output files up front. They
declare a marker file under `<output_dir>/.markers/` instead — for example
`z_train.chunks` — and the downstream stages glob the chunks at run time. The
practical consequence is that deleting a single chunk `.parquet` does not make
Snakemake rebuild it; delete the corresponding marker to force the split to be
rewritten.

The ntupelized per-batch Parquets under `temp_dir` are the pipeline's only
intermediate, and they are kept rather than deleted. They cost about one extra
copy of the dataset, and in exchange stages 2–5 can be re-run on their own: a new
`chunk_size`, a different `train_frac` or recomputed weights all replay from the
batches in minutes instead of re-ntupelizing thousands of ROOT files. Delete
`temp_dir` by hand once a dataset is final.

Note: Input simulation files can be generated using the scripts in the `sim/` directory (see [Simulation](#simulation) below).

## Setup

```bash
git clone https://github.com/HEP-KBFI/ml-tau-data
cd ml-tau-data
git submodule update --init --recursive
```

There are two environments, and the distinction matters:

- **The driver environment** — an ordinary virtualenv on the cluster login node
  holding Snakemake and nothing heavy. This is what you activate, and the only
  thing you install.
- **The Apptainer container** — holds the scientific stack (uproot, awkward,
  vector, fastjet, numba, torch). Every rule enters it for the actual
  processing. You never install into it, and you do not activate it yourself.

So the driver venv needs only the core dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Use `pip install -e ".[full]"` instead only if you also want to run the
ntupelizer scripts by hand on the host, outside the container.

> **Use Python 3.9–3.11.** Snakemake 7 parses the Snakefile with Python's own
> tokeniser, and Python 3.12 changed how f-strings are tokenised (PEP 701). On
> 3.12+ the Snakefile fails to parse with a misleading
> `NameError: name 'dataset' is not defined` (verified on 3.14; 3.11 and the
> cluster's 3.9 are fine). Upgrading Snakemake is not a way out: 8.x replaced
> the `cluster:`/`cluster-status:` profile interface with executor plugins, so
> the SLURM profile here requires 7.x.

> **pulp must stay below 2.8.** Snakemake 7 calls `pulp.list_solvers()`, which
> pulp renamed in 2.8, so a newer pulp makes even `snakemake --help` die with
> `AttributeError: module 'pulp' has no attribute 'list_solvers'`. `setup.py`
> pins this; the note is here in case you install Snakemake by hand.

> **Note:** Use `python3` and **not** `python` to create the venv even if the latter resolves to `python3`.
> The reason is that the former creates the following symlinks in venv:
> ```
> python3 -> python
> python -> /usr/bin/python
> ```
> while a venv `python3` produces these symlinks:
> ```
> python -> python3
> python3 -> /usr/bin/python3
> ```
> This is important because `/usr/bin/python` does not exist on compute nodes, but `/usr/bin/python3` does.
> Even though the jobs are run inside containers, the job orchestration performed by `snakemake` still requires a working installation of Python.

## Configuration

Edit `ntupelizer/config/workflow.yaml` before running:

```yaml
output_dir: /path/to/output          # where final Parquets are written
temp_dir:   /path/to/tmp             # per-batch Parquets; kept, not scratch
chunk_size: 100000                   # max jets per output Parquet file
row_group_size: 1024                 # rows per parquet row group
split_seed: 12345                    # makes the train/test split reproducible
files_per_job: 20                    # ROOT files per SLURM job in stage 1

ntupelizer_class: DecayProductNtupelizer   # adds the gen_jet_tau_vis_daughter_* fields

datasets:
  p8_ee_Z_tautau_ecm91:
    input_dir: /path/to/signal/root/   # or: file_list: /path/to/list.txt
    file_pattern: "*.root"
    short_name: z
    is_signal: true
    train_frac: 0.90

  p8_ee_Z_qq_ecm91:
    input_dir: /path/to/bkg/root/
    file_pattern: "*.root"
    short_name: qq
    is_signal: false
    train_frac: 0.90

weights:
  produce_plots: true
  n_files_per_sample: 100000
```

Exactly one dataset must have `is_signal: true` and one `is_signal: false`; the
weighting stage compares the two.

Inputs are discovered either by globbing `input_dir` with `file_pattern`, or from
a `file_list` text file with one path per line. `file_list` takes priority.
Glob results are cached under `.snakemake_file_lists/`, so delete that directory
after adding input files; `file_list` bypasses the cache entirely.

`split_seed` seeds the train/test assignment and the in-buffer shuffle, so a
rerun over the same inputs with the same `chunk_size` and `train_frac` reproduces
the output byte for byte. Change any of those three and the split is redrawn.

`ntupelizer_class` selects what the ntuples contain: `PodioROOTNtuplelizer` for the
standard jet-level ntuples, or `DecayProductNtupelizer` for the ParTauDETR
(tau daughter) dataset, which additionally fills `gen_jet_tau_vis_daughter_p4s`,
`gen_jet_tau_vis_daughter_pdgs` and `gen_jet_tau_vis_daughter_charges` per gen jet.

Ntupelizer parameters (input collections, branch list, lifetime variables) are in
`ntupelizer/config/podio_root_ntupelizer.yaml` — that is the file `ntupelize.py`
loads. Note that `ntupelizer/config/ntupelizer.yaml` and
`ntupelizer/config/ntupelizer_base/` are unused legacy configs; `new.yaml` there
is a stale duplicate of the branch list and editing it has no effect.

## Running the workflow

On the cluster the whole pipeline is one command:

```bash
source .venv/bin/activate
snakemake --profile ntupelizer/config/slurm
```

That is the entire operation. Snakemake works backwards from `rule all`, figures
out what is missing and drives it to completion on its own: stage 1 goes to
SLURM as one `sbatch` per batch of `files_per_job` ROOT files, and stages 2–4
run on the login node (they are declared `localrules`, being cheap next to the
ntupelization). Every job enters the container by itself, so there is nothing
else to load or activate.

The driver process must stay alive for the whole run, so start it inside `tmux`
or `screen` on the login node.

Dry-run first — it costs seconds and shows the whole plan:

```bash
snakemake --profile ntupelizer/config/slurm -n      # what would run
snakemake --profile ntupelizer/config/slurm -n -p   # ... and the exact commands
```

The profile in `ntupelizer/config/slurm/config.yaml` submits to partition `main`
with `--mem` and `--time` taken from each rule's `resources:`, retries a failed
job once (`restart-times: 1`), sets no ceiling on concurrent jobs
(`jobs: unlimited`), and polls job state through
`ntupelizer/scripts/slurm_status.py` rather than by parsing `sbatch` output.
Logs land in `logs/slurm/<rule>_<wildcards>_<jobid>.{out,err}`.

**Without SLURM**, everything on the current machine:

```bash
snakemake -j12    # 12 parallel jobs
```

### Resuming and re-running

Re-running the same command resumes: anything whose output already exists is
skipped. Since the per-batch Parquets under `temp_dir` are kept rather than
deleted, an interrupted run picks up at the batch it died on, and stages 2–4 can
be replayed without touching stage 1 at all.

```bash
# after a crash or Ctrl-C, Snakemake leaves the directory locked
snakemake --unlock

# redo the merge/split of one dataset (e.g. after changing chunk_size):
rm <output_dir>/.markers/z_train.chunks <output_dir>/.markers/z_test.chunks
snakemake --profile ntupelizer/config/slurm

# force one rule to re-run regardless of timestamps
snakemake --profile ntupelizer/config/slurm -R compute_weights

# build one target only
snakemake --profile ntupelizer/config/slurm <output_dir>/weights/sig_weights.npy
```

Deleting an individual chunk `.parquet` does *not* trigger a rebuild — the merge
stage declares marker files, not chunks (see above), so delete the marker.

### Test runs on a subset

`ntupelizer/config/workflow_test.yaml` runs the same pipeline over a slice of the
inputs, using the `file_list` input mode and its own `output_dir`/`temp_dir` so
it cannot collide with a production run. Build the lists as described in that
file's header, then:

```bash
snakemake --configfile ntupelizer/config/workflow_test.yaml -n
snakemake --configfile ntupelizer/config/workflow_test.yaml --profile ntupelizer/config/slurm
```

## ALEPH data

ALEPH ntuple production has moved to its own repository,
[lep-data](https://github.com/HEP-KBFI/lep-data). It shared no code with this
workflow beyond the helpers in `ntupelizer/tools/features.py`, which were copied
across.

## Misc

### Rare decay mode dataset

`ntupelizer/scripts/ParTauDETR_dataset_to_rare.py` adds a
`gen_jet_tau_decay_mode_rare` column to the tau daughter (ParTauDETR) dataset. It
counts each tau's visible daughters by PDG and matches the resulting multiset
against the twelve most common tau decays; anything else is labelled 15
("other"). Background (`qq`) files are skipped, since only the signal sample has
a meaningful gen-level decay mode.

The twelve targets are the most common modes in descending branching fraction,
stopping just before the first mode containing a photon (`γ π⁰ π`, 2.8e-3), so
the class id is the frequency rank. Fractions measured on 200k signal jets:

| class | decay mode | fraction |
|------:|------------|---------:|
| 0  | π⁰ π       | 3.94e-1 |
| 1  | π          | 1.75e-1 |
| 2  | 3π         | 1.41e-1 |
| 3  | 2π⁰ π      | 1.37e-1 |
| 4  | π⁰ 3π      | 6.69e-2 |
| 5  | 3π⁰ π      | 1.50e-2 |
| 6  | π K⁰       | 1.37e-2 |
| 7  | K          | 1.13e-2 |
| 8  | 2π⁰ 3π     | 6.65e-3 |
| 9  | π⁰ K       | 6.46e-3 |
| 10 | π⁰ π K⁰    | 6.16e-3 |
| 11 | 2π K       | 5.43e-3 |
| 15 | other      | 2.22e-2 |

Together the twelve cover 97.8% of signal jets. Note the PDG codes the matcher
expects: the neutral kaon is **311** (`K⁰`) in this sample, not 310 (`K⁰_S`) —
5309 vs 503 daughters over those 200k jets — and the single-kaon modes are the
charged kaon, **321**. A photon among the visible daughters vetoes every class,
so radiative decays land in "other" by design; `γ π⁰ π` alone is 2.8e-3.

Leptonic taus never appear: the ntupelizer drops `gen_jet_tau_decaymode == 16`
before these files are written, which is also why the script's electron-daughter
filter removes almost nothing.

Run it with:

```bash
./run.sh python3 ntupelizer/scripts/ParTauDETR_dataset_to_rare.py \
    -i /scratch/persistent/laurits/ml-tau/20260818_tauDaughterDataset \
    -o /scratch/persistent/laurits/ml-tau/20260824_rareDecaysDataset
```

Both paths default to those values, so plain
`./run.sh python3 ntupelizer/scripts/ParTauDETR_dataset_to_rare.py` does the same
thing. The script runs in one process and is not part of the Snakemake workflow —
run it by hand after the dataset is built.

It is a 1:1 file transform: one output per input, **under the same filename**,
with the same rows (minus the electron cut) and the same `--row-group-size`
(default 1024, matching what `merge_files.py` writes). Only the input's daughter
PDG column goes through awkward; the rest of the table is carried through as
Arrow, so every other column keeps exactly the type it had. It refuses to run
with `-o` equal to `-i`.

`--batch-size` (default 100000) controls only how many rows are held in memory at
once — it has no effect on the output layout, so lower it if the job is tight on
memory. On a 100k-jet input, 100000 peaks around 1.8 GB and 20000 around 1.0 GB,
for the same output. One caveat: a batch boundary inside a file starts a new row
group, so keep `--batch-size` at or above the input's row count if you want the
row groups to come out perfectly uniform.

## Repository structure

```
Snakefile                          # config, includes and `rule all` only
rules/                             # one file per stage, included by the Snakefile
  common.smk                       # shared config, derived constants, path helpers
  ntupelize.smk                    # stage 1
  weights.smk                      # stage 2
  merge_split.smk                  # stage 3 (one rule generated per dataset)
  validation.smk                   # stage 4
ntupelizer/
  config/
    workflow.yaml                  # dataset paths, output dirs, weight settings
    workflow_test.yaml             # same, for a subset test run
    podio_root_ntupelizer.yaml     # EDM4HEP/PodioROOT ntupelizer config (Hydra)
    weighting.yaml                 # (p, theta) binning for the weight matrices
    slurm/config.yaml              # Snakemake SLURM profile
    slurm/jobscript.sh             # cluster jobscript template
  scripts/
    ntupelize.py                   # stage 1 entry point (Hydra)
    concat_batch.py                # stage 1: concatenate one batch's per-file parquets
    merge_files.py                 # stage 3 entry point (merge/split/weight/chunk)
    compute_weights.py             # stage 2
    apply_weights.py               # standalone: re-weight existing chunks
    validate_ntuples.py            # stage 4
    ParTauDETR_dataset_to_rare.py  # standalone: add the rare decay mode label
    slurm_status.py                # Snakemake cluster-status helper
  tools/
    ntupelizing.py                 # PodioROOTNtuplelizer / EDM4HEPNtupelizer
    clustering.py                  # reco and gen jet clustering (FastJet)
    matching.py                    # reco↔gen jet matching
    gen_tau_info_matcher.py        # MC tau decay-mode and visible p4 extraction
    particle_filters.py            # reco and MC particle selection
    lifetime.py                    # track impact-parameter / lifetime variables
    tau_decaymode.py               # decay mode classification
    weight_tools.py                # (p, theta) reweighting utilities
    general.py                     # shared helpers and DUMMY_P4_VECTOR
sim/                               # Generation and simulation scripts
  cld/                             # CLD detector simulation (FCC-ee)
    CLDConfig/                     # Submodule for CLD configuration
    run_sim.sh                     # SLURM script for CLD gen-sim-reco
  clic/                            # CLIC detector simulation
    CLICPerformance/               # Submodule for CLIC configuration
    run_sim.sh                     # SLURM script for CLIC gen-sim-reco
```

## Simulation

Standalone scripts for generating EDM4HEP/PodioROOT files using **Key4hep** are provided in the `sim/` directory. These scripts handle the full generation-simulation-reconstruction chain:
1. **Generation** — Pythia8 events
2. **Simulation** — Geant4 via `ddsim`
3. **Reconstruction** — Detector-specific reconstruction (Gaudi-based)

The simulation scripts are designed to run as SLURM jobs and require access to `/cvmfs/sw.hsf.org`.

### CLD Simulation (FCC-ee)
```bash
cd sim/cld
sbatch run_sim.sh <sample_name> <seed>
```
Sample names (e.g., `p8_ee_Z_tautau_ecm91`) correspond to Pythia cards in `sim/cld/CLDConfig/pythia/`.

### CLIC Simulation
```bash
cd sim/clic
sbatch run_sim.sh <sample_name> <seed>
```
Sample names (e.g., `p8_ee_qq_ecm380`) correspond to Pythia cards in `sim/clic/pythia/`.

## Container

All heavy processing runs inside an Apptainer container:

```
/home/software/singularity/pytorch.simg:2025-09-01
```

No manual setup is needed, but it is worth knowing how it is wired, because it is
not Snakemake's built-in container support. There is no `container:` directive
and no `--use-singularity`: `rules/common.smk` builds an `apptainer exec …`
command prefix once, passes it to each rule as `params.container`, and every rule
prefixes its command with it by hand. The container is therefore entered per
*command*, not per job, and Snakemake itself is unaware of it.

On the cluster that makes three layers: `sbatch` runs the jobscript on the host,
the jobscript re-invokes Snakemake in worker mode for that one target, and the
rule's shell block then calls `apptainer exec python …`. This is why the driver
venv lives on the host and needs only Snakemake, while the scientific stack only
ever has to exist inside the image.

`run.sh` in the repository root is a separate manual wrapper around the same
image, for running a script by hand outside the workflow (see Misc). The
Snakemake workflow does not use it, and the two set different bind mounts.
