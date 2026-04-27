# Real Cryo-EM Map Tutorial: 9UGC_A

<p align="center">
  <video src="https://github.com/user-attachments/assets/1b9a7c2e-26c6-4ca5-88d4-12378680e277.mp4" width="500" autoplay loop muted playsinline>
  </video>
</p>

EmbedOpt inference guided by a real cryo-EM density map (EMD-64136, 3.52 Å).

**Inputs already in this folder:**
| File | Description |
|---|---|
| `emd_64136_cropped.mrc` | Cropped cryo-EM map (from EMDB EMD-64136) |
| `9UGC.pdb` | Reference PDB for Kabsch alignment during reward computation |
| `9UGC_A.fasta` | Protein sequence |

---

## Step 0: Crop your map (already done here)

If starting from a raw EMDB download, crop the map with the `embedopt-crop-map` CLI:

```bash
embedopt-crop-map emd_64136.map --thresh 0.35 --pad 10 -o emd_64136_cropped.mrc
```

`--thresh` is map-dependent — inspect the map in ChimeraX to find a density cutoff that tightly encloses the structure. `--pad 10` adds a 10-voxel buffer on each side.

---

## Step 1: Make minimal JSON

```bash
pixi run python examples/real_map_tutorial/01_make_minimal_json.py
```

Reads `9UGC_A.fasta` and writes `9UGC_A_minimal.json` (sequence only, no MSA), which is the input for Step 2.

---

## Step 2: Generate MSA

```bash
bash examples/real_map_tutorial/02_make_msa.sh
```

Fetches MSA via the Protenix/ColabFold MMseqs2 server and writes results to `9UGC_A_msa/9UGC_A/msa/0/`. Also produces `9UGC_A_minimal-update-msa.json` with MSA paths embedded, which Step 3 uses as its basis.

---

## Step 3: Generate inference configs

Edit the variables at the top of `03_make_cfg.py`:
- `ALGORITHMS` — any subset of `"base"` (unconditional), `"dps"` (Denoising Posterior Sampling), `"embedopt"` (our method, embedding optimization)
- `LEARNING_RATES` — list of floats; one config per (algorithm, lr) combination
- `REF_PDB`, `REF_MAP`, `RESOLUTION` — map reward inputs
- `OUT_DIR` — where configs are written (default: `inference_configs/`)

Then run:

```bash
pixi run python examples/real_map_tutorial/03_make_cfg.py
```

Configs are written to `inference_configs/`, e.g. `9UGC_A_embedopt_lr{LEARNING_RATES}.json`.

---

## Step 4: Run inference

Edit the variables at the top of `04_run_inference.sh`:
- `MODEL_NAME` — model checkpoint (`protenix_base_default_v0.5.0`, `protenix_base_default_v1.0.0`, or `protenix-v2`); auto-downloaded on first run. **Note:** `protenix-v2` weights currently cannot be auto-downloaded outside China (Bytedance CDN returns 403; see [issue #296](https://github.com/bytedance/Protenix/issues/296)); use `v0.5.0` or `v1.0.0` instead, or set `CHECKPOINT_DIR` to a local copy
- `INPUT_JSON` — path to a config from `inference_configs/`
- `DUMP_DIR` — output directory (default: `output/`)
- `N_STEPS` — number of diffusion denoising steps
- `SEEDS` — comma-separated seeds; one output structure per seed

Then run:

```bash
bash examples/real_map_tutorial/04_run_inference.sh
```

Energy relaxation runs automatically on the final diffusion step.

**Outputs** under `output/<config_name>/seed_<N>/`:
- `trajectory/step_<last>_relaxed.pdb` — energy-relaxed final structure (recommended for downstream use)
- `predictions/*.cif` — raw model output (before Kabsch alignment and energy relaxation)
- `trajectory/step_<N>.pdb` — intermediate snapshots every `save_pdb_every_n_steps` steps
- `reward_history.json` — per-step map reward values
- `profile.json` — timing and peak GPU memory
