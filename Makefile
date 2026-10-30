# Usage:
#   make cfg_8AHU_A                 # build x* and the inference JSON for 8AHU_A (one EmbedOpt job, lr 0.1)
#   make stage3a                    # Stage 3a oracle on that JSON
#   make stage3a_base               # unguided Protenix baseline on the same system (needs cfg_8AHU_A)
#   make stage3acheck               # plots + RMSD + reward sensitivity for stage3a vs base
#   make stage1mwe CFG=path/to.json # Stage 1 MWE on any map-reward JSON
BENCH      ?= examples/synthetic_map_benchmark
SYSTEM     ?= 8AHU_A
CFG        ?= $(BENCH)/stage3a_cfgs/$(SYSTEM)_benchmark.json
DUMP_DIR   ?= out/$@
MODEL      ?= protenix_base_default_v0.5.0
CKPT_DIR   ?= $(HOME)/.cache/protenix/checkpoint
N_STEP     ?= 100
SEEDS      ?= 101

export LAYERNORM_TYPE := torch_layernorm

RUN_ARGS = --input_json_path $(CFG) --dump_dir $(DUMP_DIR) --model_name $(MODEL) \
           --load_checkpoint_dir $(CKPT_DIR) --use_template false \
           --enable_diffusion_shared_vars_cache false \
           --sample_diffusion.N_step $(N_STEP) --seeds $(SEEDS)

.PHONY: stage1mwe stage3a stage3a_base stage3acheck cfg_8AHU_A

stage1mwe:
	python stage1_mwe.py $(RUN_ARGS)

stage3a:
	python stage3a_oracle.py $(RUN_ARGS)

BASE_CFG = $(BENCH)/stage3a_cfgs/$(SYSTEM)_base.json
$(BASE_CFG): $(CFG)
	python -c "import json; j = json.load(open('$(CFG)'))[0]; j['name'] = j['name'] + '_base'; \
	j['algorithm'] = {'type': 'base', 'params': {}}; json.dump([j], open('$@', 'w'), indent=2)"

stage3a_base: $(BASE_CFG)
	python stage3a_oracle.py $(subst $(CFG),$(BASE_CFG),$(RUN_ARGS))

stage3acheck:
	python stage3a_check.py --ref $(BENCH)/pdb_clean/$(SYSTEM).pdb --runs out/stage3a out/stage3a_base \
	       --out out/stage3a_check.png

# Their synthetic-benchmark pipeline in --test mode, MSA for SYSTEM only (step 04, map rendering, is not needed)
cfg_8AHU_A:
	python $(BENCH)/00_pisces_subsample.py
	python $(BENCH)/01_get_fasta.py
	python $(BENCH)/02_fetch_cif.py
	python $(BENCH)/03_clean_cif_to_pdb.py
	python $(BENCH)/05_make_minimal_json.py --test
	protenix msa --input $(BENCH)/json_minimal/$(SYSTEM)_minimal.json --out_dir $(BENCH)/msa --msa_server_mode protenix
	python $(BENCH)/07_make_benchmark_cfgs.py --test --algorithms embedopt --learning-rates 0.1 \
	       --no-base --output-folder stage3a_cfgs
	@echo "Config: $(CFG)"