"""
Minimal ML-Tau data processing workflow.

Stages, one file each under rules/:
  1. ntupelize     - process each input file 1:1 to output
  2. weights       - accumulate the (p, theta) weight matrices straight from the
                     ntupelized batches
  3. merge_split   - stream the ntupelized batches of each dataset into
                     train/test chunk files of `chunk_size` events, weighting
                     as they are filled
  4. validation    - produce validation plots

The final products are numbered chunk files in OUTPUT_DIR, e.g.
z_train_00000.parquet, z_train_00001.parquet, ..., qq_test_00000.parquet.  A
split is never materialised as a single file, nor as an unweighted copy, nor as
a merged intermediate: stage 3 reads each event once and writes it once,
buffering only as much as one output file holds.

Everything the stages share -- config lookups, derived constants, path helpers
and input discovery -- lives in rules/common.smk, which must be included first.
The stage files carry no logic beyond their own rule: anything needing more than
a few lines belongs in ntupelizer/scripts/, where it can be imported and tested
rather than living inside a shell string.
"""

# Snakemake reads this at startup and makes its contents available as the global
# 'config' dict throughout all rules and Python code in every included file. The
# path is relative to the working directory where you invoke snakemake — which is
# the repo root when running from there.  Dataset-level config lives here;
# processing config lives elsewhere under ntupelizer/config/.
configfile: "ntupelizer/config/workflow.yaml"

# Expand shell variables defined in the YAML config file
import os
for key in ("output_dir", "temp_dir"):
    config[key] = os.path.expandvars(config[key])

# common.smk first: every stage file below uses the names it defines.
include: "rules/common.smk"
include: "rules/ntupelize.smk"
include: "rules/weights.smk"
include: "rules/merge_split.smk"
include: "rules/validation.smk"


# ── rule all — the final target ───────────────────────────────────────────────
# Snakemake works backwards from the requested output files to figure out
# which rules to run.  'rule all' lists the ultimate desired outputs so that
# running `snakemake` with no arguments processes every dataset end-to-end.
#
# Rules named in localrules always run on the local machine even when
# --profile slurm is active.  ntupelize is the only rule submitted to SLURM —
# everything else is fast enough to run locally.
localrules: all, compute_weights, validation


rule all:
    input:
        [chunks_marker(SHORT_NAMES[ds], split) for ds in DATASETS for split in SPLITS],
        f"{OUTPUT_DIR}/validation/.done",
        f"{WEIGHTS_DIR}/sig_weights.npy",
