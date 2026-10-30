# Usage:
#   make cfg SYSTEM=8AHU_A          # build x* and the inference JSON for SYSTEM (one EmbedOpt job, lr 0.1)
#   make stage3a                    # Stage 3a oracle on that JSON
#   make stage3a_base               # unguided Protenix baseline on the same system (needs cfg)
#   make stage3acheck               # plots + RMSD + reward sensitivity for stage3a vs base (+ map run if present)
#   make stage3a_lr LR=0.5          # stage3a with another EmbedOpt learning rate -> out/$(SYSTEM)/stage3a_lr0.5
#   make maps && make stage_map     # Li et al.'s map-based EmbedOpt on the 5 A synthetic map (upper bound)
#   make stage1mwe CFG=path/to.json # Stage 1 MWE on any map-reward JSON
BENCH      ?= examples/synthetic_map_benchmark
SYSTEM     ?= 8AHU_A
CFG        ?= $(BENCH)/stage3a_cfgs/$(SYSTEM)_benchmark.json
DUMP_DIR   ?= out/$(SYSTEM)/$@
MODEL      ?= protenix_base_default_v0.5.0
CKPT_DIR   ?= $(HOME)/.cache/protenix/checkpoint
N_STEP     ?= 100
SEEDS      ?= 101,102,103

export LAYERNORM_TYPE := torch_layernorm

RUN_ARGS = --input_json_path $(CFG) --dump_dir $(DUMP_DIR) --model_name $(MODEL) \
           --load_checkpoint_dir $(CKPT_DIR) --use_template false \
           --enable_diffusion_shared_vars_cache false \
           --sample_diffusion.N_step $(N_STEP) --seeds $(SEEDS)

.PHONY: stage1mwe stage3a stage3a_base stage3a_lr stage3acheck cfg maps stage_map

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

LR ?= 0.5
LR_CFG = $(BENCH)/stage3a_cfgs/$(SYSTEM)_lr$(LR).json
$(LR_CFG): $(CFG)
	python -c "import json; j = json.load(open('$(CFG)'))[0]; j['name'] = j['name'] + '_lr$(LR)'; \
	j['algorithm']['params'] = {'reward_scale_s': $(LR), 'reward_scale_z': $(LR)}; json.dump([j], open('$@', 'w'), indent=2)"

stage3a_lr: $(LR_CFG)
	python stage3a_oracle.py $(subst $(CFG),$(LR_CFG),$(subst out/$(SYSTEM)/$@,out/$(SYSTEM)/stage3a_lr$(LR),$(RUN_ARGS)))

maps:
	python $(BENCH)/04_generate_map.py

stage_map:
	python -m embedopt.inference.inference $(RUN_ARGS)

stage3acheck:
	python stage3a_check.py --ref $(BENCH)/pdb_clean/$(SYSTEM).pdb \
	       --runs out/$(SYSTEM)/stage3a out/$(SYSTEM)/stage3a_base $(wildcard out/$(SYSTEM)/stage3a_lr*) $(wildcard out/$(SYSTEM)/stage_map) \
	       --out out/$(SYSTEM)/stage3a_check.png

# Their synthetic-benchmark pipeline (all 77 x*/minimal JSONs are cheap), MSA for SYSTEM only (step 04, map rendering, is not needed)
cfg:
	python $(BENCH)/00_pisces_subsample.py
	python $(BENCH)/01_get_fasta.py
	python $(BENCH)/02_fetch_cif.py
	python $(BENCH)/03_clean_cif_to_pdb.py
	python $(BENCH)/05_make_minimal_json.py
	protenix msa --input $(BENCH)/json_minimal/$(SYSTEM)_minimal.json --out_dir $(BENCH)/msa --msa_server_mode protenix
	python $(BENCH)/07_make_benchmark_cfgs.py --algorithms embedopt --learning-rates 0.1 \
	       --no-base --output-folder stage3a_cfgs
	@echo "Config: $(CFG)"