# AF Distance Benchmark

End-to-end benchmark for distance-constraint steered diffusion on the 24-system set introduced by Distance-AF[1]. Each system is a chain whose Protenix prediction deviates substantially from the experimental structure. This benchmark experiment measures whether steering methods can generate the sample to satisfy inter-residue distance constraints.

[1] Zhang, Y., Zhang, Z., Kagaya, Y., Terashi, G., Zhao, B., Xiong, Y., and Kihara, D.  
  *Distance-AF improves predicted protein structure models by AlphaFold2 with user-specified distance constraints.*
  Nature Communications Biology **8**(1):1392, 2025.
  [https://www.nature.com/articles/s42003-025-08783-5](https://www.nature.com/articles/s42003-025-08783-5)

Two parallel sweep tasks share the same preparation pipeline:

- **Sweep A — Learning-rate sweep** (08 / 09 / 10 / 11 / 12 / 13): fix `N_step = 200`, vary `lr ∈ {0.005, 0.01, 0.02, 0.05, 0.1, 0.2}`.
- **Sweep B — N_step robustness sweep** (08b / 09b / 10 / 11 / 12b): hold the optimization budget constant at `N_step × lr = 200 × 0.1 = 20`, vary `N_step ∈ {25, 50, 80, 100, 125, 160}` with paired lr.

Run from this directory (`cd examples/AF_distance_benchmark/`) with the pixi env (`pixi install` from the repo root) active.

## Shared prep (steps 01–07)

```bash
# 01–04: download + extract per-chain PDB/FASTA, build minimal JSONs, fetch MSAs, tag base sampler
pixi run python 01_prepare_pdb_and_fasta.py
pixi run python 02_make_minimal_json.py
pixi run python 03_fetch_msa.py
pixi run python 04_add_base_sampler_field.py

# 05: base sampling (24 systems × 3 seeds, parallelized via SLURM array)
sbatch --array=0-5 05_submit_base_jobs.sh

# Wait for base jobs, then build distance constraints from where base predictions deviate most from GT:
pixi run python 06_make_distance_matrix.py
pixi run python 07_make_constraint_dataframes.py
```

## Sweep A: learning-rate sweep (N_step = 200)

```bash
# 08: generate per-system configs (one job per algorithm × lr)
pixi run python 08_make_distance_constraint_jsons.py

# 09: steered sampling — embedopt + dps × 6 lrs × 3 seeds
sbatch --array=0-23 09_submit_steering_jobs.sh

# Analysis (after step 09 completes):
sbatch 10_run_molprobity.sh                                       # requires phenix
pixi run python 11_constraint_satisfaction_molprobity_report.py
pixi run python 12_make_plots.py
pixi run python 13_reward_trace_visualization.py
```

## Sweep B: N_step robustness sweep (N_step × lr = 20)

Re-uses the same constraint set as Sweep A (computed from the base predictions in step 05). Re-uses `10_run_molprobity.sh` as-is and the same `11_…report.py` (with a `--steering-dir-template` override that adds an `nstep` column).

```bash
# 08b: same constraint set as 08, but each job carries (N_step, lr) where product = 20
pixi run python 08b_make_nstep_constraint_jsons.py                # all systems
pixi run python 08b_make_nstep_constraint_jsons.py 1IXC_A         # smoke-test single system

# 09b: one inference call per job (N_step varies within a system; jq parses each job's n_step)
sbatch --array=0-23 09b_submit_nstep_steering_jobs.sh

# Molprobity on the new output dir
sbatch 10_run_molprobity.sh steering_output_nstep_k20             # requires phenix

# Analysis: emit nstep column by pointing 11 at the new output dir
pixi run python 11_constraint_satisfaction_molprobity_report.py \
    --steering-dir-template "steering_output_nstep_k{k}" \
    --output-dir analysis_nstep_k20_relaxed

# 12b: 6-panel plot with N_step on x-axis (also pulls lr=0.1 from analysis_k20_relaxed/ as the 200-step anchor)
pixi run python 12b_make_nstep_plots.py
```

`09b` requires `jq` (system-wide on most clusters; otherwise `mamba install -c conda-forge jq`).

## Note on SLURM job submission

Before submitting any SLURM job (steps 05, 09, 09b, 10):

- Set `CHECKPOINT_DIR` to the directory holding `protenix_base_default_v0.5.0.pt`, e.g. `export CHECKPOINT_DIR=/path/to/checkpoints`. The `v0.5.0` weights auto-download on first run; `protenix-v2` does not (Bytedance CDN blocks downloads outside China; see [issue #296](https://github.com/bytedance/Protenix/issues/296)).
- Set `SBATCH_PARTITION` to your cluster's partition name (the SLURM scripts intentionally omit `--partition` for portability). Use a GPU partition for steps 05 / 09 / 09b and a CPU partition for step 10. Equivalent: pass `sbatch -p <name> ...` at submission time.
- Review the `#SBATCH` directives at the top of `05_submit_base_jobs.sh`, `09_submit_steering_jobs.sh`, `09b_submit_nstep_steering_jobs.sh`, and `10_run_molprobity.sh` (GPU constraint `-C`, time `-t`, memory `--mem`, etc.) and adjust them to match your cluster.

## Note on 10_run_molprobity.sh

Step 10 requires a [Phenix installation](https://phenix-online.org/). Once installed, point the script at its activation script:

```bash
export PHENIX_ENV_SCRIPT=/path/to/phenix-1.21.2-5419/phenix_env.sh
```

The wrapper sources this before invoking `phenix.molprobity`. Alternatively, source `phenix_env.sh` manually in your submitting shell before `sbatch` and the env will propagate to the SLURM job.

If `phenix.molprobity` isn't on PATH after activation, the script bails with a clear error.