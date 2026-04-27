# Synthetic Map Benchmark

Scripts to reproduce the synthetic cryo-EM map benchmark from the EmbedOpt paper. The benchmark evaluates three sampling algorithms — `base` (unconditional), `dps` (Denoising Posterior Sampling), and `embedopt` (our method) — on a set of real protein structures from the PDB, each paired with a synthetic electron density map generated at a target resolution.

---

## Prerequisites

- **pixi** environment (run `pixi install` from the repo root)
- **SLURM** cluster with GPU nodes (scripts use `sbatch`)
- **Phenix** — optional, required only for validation (step 09). License required; download at https://phenix-online.org/download

The PISCES input files needed for step 00 are already included in `pisces_raw/`.

---

## Note on SLURM job submission

Before submitting the SLURM jobs (steps 08, 09):

- Set `SBATCH_PARTITION` to your cluster's partition name (the SLURM scripts intentionally omit `--partition` for portability). Use a GPU partition for step 08 and a CPU partition for step 09. Equivalent: pass `sbatch -p <name> ...` at submission time.
- Review the `#SBATCH` directives at the top of `08_submit_jobs.sh` and `09_run_validity_molprobity_relaxed.sh` (GPU constraint `-C`, time `-t`, memory `--mem`, CPUs, etc.) and adjust them to match your cluster.
- Logs are written to `slurm_logs/` in this directory; the scripts create it on first run.

---

## Pipeline

### Step 00 — Subsample proteins from PISCES

```bash
pixi run python examples/synthetic_map_benchmark/00_pisces_subsample.py
```

Reads a curated PDB list from the [PISCES server](https://dunbrack.fccc.edu/pisces/) (included in `pisces_raw/`), filters to recent structures (PDB IDs starting with "8") and chains A–H, then draws up to 10 proteins from each of 10 length bins (150–1000 aa). Sparse long-sequence bins yield fewer entries, giving **80 proteins** total. Sampling is fully deterministic (`random_state=42`).

The resulting `pisces_subsampled.csv` is already committed to the repo — re-running this step is only needed if you want to generate a different cohort.

Output: `pisces_subsampled.csv`, `pisces_len_distribution.png`

---

### Step 01 — Extract FASTA sequences

```bash
pixi run python examples/synthetic_map_benchmark/01_get_fasta.py
```

Parses a PISCES FASTA file and writes individual `.fa` files for each selected protein.

Output: `fasta/<pdbid>_<chain>.fa`

---

### Step 02 — Download CIF structures

```bash
pixi run python examples/synthetic_map_benchmark/02_fetch_cif.py
```

Downloads CIF files from RCSB for each protein in `pisces_subsampled.csv`.

Output: `cif_raw/<pdbid>_<chain>.cif`

---

### Step 03 — Clean structures to PDB

```bash
pixi run python examples/synthetic_map_benchmark/03_clean_cif_to_pdb.py
```

Extracts the target chain, validates the sequence against the FASTA, sets all B-factors to 50, and places the molecule in a P1 unit cell sized to the bounding box + 10 Å padding on each side. 3 of the 80 entries fail with a sequence mismatch (PDB entries updated on RCSB since the PISCES list was generated); this is expected and consistent with the original benchmark run.

Output: `pdb_clean/<pdbid>_<chain>.pdb` (77 files)

---

### Step 04 — Render PDB files to CCP4 map with SFcalculator

```bash
pixi run python examples/synthetic_map_benchmark/04_generate_map.py
```

Uses [SFC_Torch](https://www.biorxiv.org/content/10.1101/2025.01.12.632630v2) to compute synthetic cryo-EM electron density maps at 5 Å resolution. Each map is normalized to zero mean and unit variance. Requires a CUDA GPU (`DEVICE = "cuda:0"`).

Pass `--test` to run on only 3 systems for a quick sanity check:

```bash
pixi run python examples/synthetic_map_benchmark/04_generate_map.py --test
```

Output: `synthetic_maps/5.0A/<pdbid>_<chain>.map` (77 maps)

---

### Step 05 — Create minimal JSON inputs

```bash
pixi run python examples/synthetic_map_benchmark/05_make_minimal_json.py
```

Writes a minimal Protenix-format JSON for each protein (sequence only, no MSA paths yet).

Pass `--test` to run on only 3 systems:

```bash
pixi run python examples/synthetic_map_benchmark/05_make_minimal_json.py --test
```

Output: `json_minimal/<pdbid>_<chain>_minimal.json`

---

### Step 06 — Fetch MSAs

```bash
bash examples/synthetic_map_benchmark/06_fetch_msa.sh
```

Calls `protenix msa` to fetch MSAs via the Protenix MSA server and writes MSA-updated JSONs.

Pass `--test` to run on only 3 systems:

```bash
bash examples/synthetic_map_benchmark/06_fetch_msa.sh --test
```

Output: `msa/`, `json_minimal/<pdbid>_<chain>_minimal-update-msa.json`

---

### Step 07 — Generate benchmark configs

```bash
pixi run python examples/synthetic_map_benchmark/07_make_benchmark_cfgs.py
```

This generates configs for all 77 systems using the defaults from the paper: `embedopt` and `dps` at 8 learning rates (0.0025, 0.005, 0.01, 0.02, 0.1, 0.2, 0.5, 1.0) plus the unconditional `base` baseline, at 5 Å resolution. Each protein gets one benchmark JSON containing all jobs.

Pass `--test` to run on only 3 systems:

```bash
pixi run python examples/synthetic_map_benchmark/07_make_benchmark_cfgs.py --test
```

Key options:
- `--algorithms` — any subset of `"embedopt"`, `"dps"` (default: both)
- `--learning-rates` — learning rates / guided scales (default: 8 values above)
- `--no-base` — exclude the unconditional baseline job (included by default)
- `--save-every-n-steps` — save trajectory PDB every N diffusion steps (default: 25)
- `--output-folder` — output directory name (default: `benchmark_cfgs_5A`)

Output: `benchmark_cfgs_5A/<pdbid>_<chain>_benchmark.json`

---

### Step 08 — Run inference

Energy relaxation runs **automatically** on the final diffusion step (controlled by `energy_relaxation: true` in the reward config, set by step 07). No separate relaxation step is needed.

The script must be submitted from within its own directory:

```bash
cd examples/synthetic_map_benchmark
sbatch 08_submit_jobs.sh                    # all systems
sbatch 08_submit_jobs.sh 8AHU_A 8ANE_A     # specific systems
sbatch --array=0-4 08_submit_jobs.sh        # parallel job array
sbatch 08_submit_jobs.sh --test             # 3 systems only
```

Key parameters at the top of `08_submit_jobs.sh`:
- `config_folder` — which benchmark config directory to run (default: `benchmark_cfgs_5A`)
- `output_folder` — output root (default: `benchmark_output/`)
- `DIFFUSION_STEPS`, `SEEDS`, `N_CYCLE`
- `CHECKPOINT_DIR` — override via env var if needed. **Note:** `protenix-v2` weights cannot be auto-downloaded outside China (Bytedance CDN returns 403; see [issue #296](https://github.com/bytedance/Protenix/issues/296)); `v0.5.0` and `v1.0.0` download fine

The script sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` automatically to reduce GPU memory fragmentation when running many configs sequentially.

**Outputs** under `benchmark_output/<pdbid>_<chain>/<config_name>/seed_<N>/`:
- `trajectory/step_99_relaxed.pdb` — energy-relaxed final structure (use this for analysis)
- `trajectory/step_<K>.pdb` — intermediate snapshots every `save_pdb_every_n_steps` steps
- `reward_history.json` — per-step map reward values
- `profile.json` — timing and peak GPU memory

---

### Step 09 — Validate relaxed structures (optional, requires Phenix)

> **Phenix license required.** Download at https://phenix-online.org/download

Runs `phenix.molprobity` and `phenix.validation_cryoem` on the energy-relaxed final structures (`step_99_relaxed.pdb`). Before validation, B-factors are set to a uniform value (default: 50) in-place — inference outputs zeros which affect CC scoring, and 50 is consistent with the value used both in the synthetic map generation (step 03) and in our forward model for reward calculation. Refinement of per-residue B-factors will be explored in future work.

Set `PHENIX_ENV_SCRIPT` to the phenix activation script before submitting; the job will source it automatically:

```bash
export PHENIX_ENV_SCRIPT=/path/to/phenix-X.Y.Z/phenix_env.sh
```

Must be submitted from within the `synthetic_map_benchmark/` directory:

```bash
cd examples/synthetic_map_benchmark
sbatch --array=0-12 09_run_validity_molprobity_relaxed.sh           # all systems
sbatch --array=0-5  09_run_validity_molprobity_relaxed.sh --test    # 3 systems only
sbatch --array=0-12 09_run_validity_molprobity_relaxed.sh 8AHU_A    # specific system
sbatch --array=0-12 09_run_validity_molprobity_relaxed.sh --force   # rerun existing
```

Options: `--resolutions 5.0`, `--lrs 0.5,1.0`, `--bfactor 50`, `--force`

Outputs per seed: `validation_relaxed.log`, `molprobity_relaxed_molprobity.log`

---

### Steps 10–12 — Analysis and visualization

Run step 09 (Phenix validation) before running these. All scripts use `pixi run python`:

```bash
pixi run python examples/synthetic_map_benchmark/10_get_summary_csv.py          # all systems
pixi run python examples/synthetic_map_benchmark/10_get_summary_csv.py --test   # 3 systems only
pixi run python examples/synthetic_map_benchmark/11_plot_difficulty.py
pixi run python examples/synthetic_map_benchmark/12_plot_learning_rates.py
```

| Script | Description | Output |
|---|---|---|
| `10_get_summary_csv.py` | Extract CC_mask, MolProbity, Ramachandran outliers, Clashscore from Phenix logs into a flat CSV | `benchmark_summary.csv` |
| `11_plot_difficulty.py` | Difficulty vs. final map correlation: EmbedOpt vs. DPS (Figure 3B) | `difficulty_vs_performance_5A.png` |
| `12_plot_learning_rates.py` | CC_mask and MolProbity vs. learning rate: EmbedOpt vs. DPS (Figure 3C) | `learning_rate_sensitivity_5A.png` |

Scripts 11 and 12 read from `benchmark_summary.csv` and accept `--input` / `--output` to override paths. Script 10 accepts `--test` to run on 3 systems only.
