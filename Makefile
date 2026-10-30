# Usage:
#   make cfg SYSTEM=8AHU_A                    # x*, MSA and the base JSON config for SYSTEM
#   make maps                                 # 5 A synthetic maps for all systems (needed by METHOD=map)
#   make run SYSTEM=8AHU_A METHOD=moment LR=0.5   # METHOD in {base, moment, map} -> out/$(SYSTEM)/<method>_lr<LR>
#   make check SYSTEM=8AHU_A                  # summary table + CSV + figure over all runs in out/$(SYSTEM)/
#   make stage1mwe CFG=path/to.json           # Stage 1 MWE on any map-reward JSON
#
# METHOD=base   : unguided Protenix (moment loss logged only)          -> out/$(SYSTEM)/base
# METHOD=moment : EmbedOpt + oracle uniform-mu moments (stage3a_oracle) -> out/$(SYSTEM)/moment_lr$(LR)
# METHOD=map    : EmbedOpt + Li et al.'s MapReward on the 5 A map       -> out/$(SYSTEM)/map_lr$(LR)
BENCH      ?= examples/synthetic_map_benchmark
SYSTEM     ?= 8AHU_A
METHOD     ?= moment
LR         ?= 0.1
CFG        ?= $(BENCH)/stage3a_cfgs/$(SYSTEM)_benchmark.json
MODEL      ?= protenix_base_default_v0.5.0
CKPT_DIR   ?= $(HOME)/.cache/protenix/checkpoint
N_STEP     ?= 100
SEEDS      ?= 101,102,103,104,105

export LAYERNORM_TYPE := torch_layernorm

RUN_NAME   = $(if $(filter base,$(METHOD)),base,$(METHOD)_lr$(LR))
RUN_CFG    = $(BENCH)/stage3a_cfgs/$(SYSTEM)_$(RUN_NAME).json
ENTRY_base   = stage3a_oracle.py
ENTRY_moment = stage3a_oracle.py
ENTRY_map    = -m embedopt.inference.inference

RUN_ARGS = --dump_dir out/$(SYSTEM)/$(RUN_NAME) --model_name $(MODEL) \
           --load_checkpoint_dir $(CKPT_DIR) --use_template false \
           --enable_diffusion_shared_vars_cache false \
           --sample_diffusion.N_step $(N_STEP) --seeds $(SEEDS)

.PHONY: cfg maps run check stage1mwe

run:
	python -c "import json; j = json.load(open('$(CFG)'))[0]; j['name'] = '$(SYSTEM)_$(RUN_NAME)'; \
	j['algorithm'] = {'type': 'base', 'params': {}} if '$(METHOD)' == 'base' else \
	{'type': 'embedopt', 'params': {'reward_scale_s': $(LR), 'reward_scale_z': $(LR)}}; \
	json.dump([j], open('$(RUN_CFG)', 'w'), indent=2)"
	python $(ENTRY_$(METHOD)) --input_json_path $(RUN_CFG) $(RUN_ARGS)

check:
	python stage3a_check.py --ref $(BENCH)/pdb_clean/$(SYSTEM).pdb \
	       --runs $(sort $(wildcard out/$(SYSTEM)/base out/$(SYSTEM)/moment_lr* out/$(SYSTEM)/map_lr*)) \
	       --out out/$(SYSTEM)/check.png --csv out/$(SYSTEM)/summary.csv

maps:
	python $(BENCH)/04_generate_map.py

stage1mwe:
	python stage1_mwe.py --input_json_path $(CFG) $(RUN_ARGS)

# Their synthetic-benchmark pipeline (all 77 x*/minimal JSONs are cheap), MSA for SYSTEM only
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