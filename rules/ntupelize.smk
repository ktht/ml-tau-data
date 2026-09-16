# ── stage 1 : ntupelize (one SLURM job per batch of input files) ────────────
# Each job processes BATCH_SIZE ROOT files and merges them into one parquet.
# Using a batch index wildcard instead of a per-file stem wildcard keeps the
# DAG size at O(N_files / BATCH_SIZE) rather than O(N_files), making DAG
# construction fast even with thousands of input files.
rule ntupelize:
    input:
        # Resolve the list of ROOT files for this batch index.
        # _DATASET_BATCHES is pre-computed at startup so this lookup is O(1).
        lambda wc: _DATASET_BATCHES[wc.dataset][int(wc.batch_idx)]
    output:
        # One output parquet per batch, kept rather than temp(): these are the
        # only intermediate the pipeline has, and holding on to them means the
        # merge/split/weight stage can be re-run — with a different chunk_size
        # or train_frac, say — without re-ntupelizing thousands of ROOT files.
        f"{TEMP_DIR}/{{dataset}}/batch_{{batch_idx}}.parquet"
    params:
        is_signal        = lambda wc: DATASETS[wc.dataset]["is_signal"],
        ntupelizer_class = NTUPELIZER_CLASS,
        container        = CONTAINER,
        # Temporary per-job directory for individual per-file parquets that
        # are concatenated into the single batch output at the end.
        per_file_tmp = lambda wc: f"{TEMP_DIR}/{wc.dataset}/.batch_{wc.batch_idx}_tmp",
    resources:
        # No mem_mb here: the batch's files are processed one at a time, so peak
        # RSS is set by the single largest file, not by files_per_job.  The
        # profile default (2 GB) applies.
        cpus    = 1,
        # Runtime, unlike memory, really is proportional to the batch size:
        # the files are processed sequentially, so their times add up.
        runtime = lambda wc, input: 20 * len(input),
    shell:
        # Process each file in the batch, then concatenate into one parquet.
        # The per-file tmp directory is cleaned up regardless of success/failure.
        # Individual file failures are non-fatal: a single corrupt/empty ROOT
        # file should not abort the entire batch.  The batch only fails if no
        # per-file parquets were produced at all (caught by concat_batch.py).
        """
        mkdir -p {params.per_file_tmp}
        trap 'rm -rf {params.per_file_tmp}' EXIT

        n_ok=0
        n_fail=0
        for f in {input}; do
            stem=$(basename "$f" .root)
            if {params.container} python ntupelizer/scripts/ntupelize.py \
                    ++input_path="$f" \
                    ++output_path="{params.per_file_tmp}/$stem.parquet" \
                    ++is_signal={params.is_signal} \
                    ++ntupelizer_class={params.ntupelizer_class} \
                    hydra.run.dir=/tmp \
                    hydra.output_subdir=null \
                    hydra/job_logging=disabled; then
                n_ok=$((n_ok + 1))
            else
                echo "WARNING: ntupelize failed for $f (exit $?), skipping"
                n_fail=$((n_fail + 1))
            fi
        done
        echo "Batch summary: $n_ok succeeded, $n_fail failed"

        {params.container} python ntupelizer/scripts/concat_batch.py \
            -i {params.per_file_tmp} \
            -o {output}
        """
