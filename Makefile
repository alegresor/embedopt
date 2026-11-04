# Usage:
#   make cfg SYSTEM=8AHU_A                        # x*, MSA and the base JSON config for SYSTEM
#   make maps                                     # 5 A synthetic maps for all systems (needed by METHOD=map)
#   make run SYSTEM=8K9Z_A METHOD=moment ORDER=3 LR=0.1 [RES=4.0]
#   make check SYSTEM=8K9Z_A                      # summary table + CSV + figure over all runs in out/$(SYSTEM)/
#
# METHOD=base   : unguided Protenix (second-order moment loss logged only)     -> out/$(SYSTEM)/base
# METHOD=moment : EmbedOpt + oracle uniform-mu moments of order ORDER (2 or 3)  -> out/$(SYSTEM)/moment<ORDER>_lr<LR>[_res<RES>]
# METHOD=map    : EmbedOpt + Li et al.'s MapReward on the 5 A synthetic map      -> out/$(SYSTEM)/map_lr<LR>
BENCH      ?= examples/synthetic_map_benchmark
SYSTEM     ?= 8AHU_A
METHOD     ?= moment
ORDER      ?= 2
LR         ?= 0.1
RES        ?= 5.0
LB         ?= 24
LAMBDA3    ?= 1.0
NKAPPA     ?= 16
CFG        ?= $(BENCH)/stage3a_cfgs/$(SYSTEM)_benchmark.json
MODEL      ?= protenix_base_default_v0.5.0
CKPT_DIR   ?= $(HOME)/.cache/protenix/checkpoint
N_STEP     ?= 100
SEEDS      ?= 101,102,103,104,105

export LAYERNORM_TYPE := torch_layernorm

RES_TAG    = $(if $(filter 5.0,$(RES)),,_res$(RES))
RUN_NAME   = $(if $(filter base,$(METHOD)),base,$(if $(filter map,$(METHOD)),map_lr$(LR),moment$(ORDER)_lr$(LR)$(RES_TAG)))
RUN_CFG    = $(BENCH)/stage3a_cfgs/$(SYSTEM)_$(RUN_NAME).json
MOMENT_ENV = MOMENT_LB=$(LB) MOMENT_LAMBDA3=$(LAMBDA3) MOMENT_NKAPPA=$(NKAPPA)
ENTRY_base   = MOMENT_ORDER=2 $(MOMENT_ENV) python -m moments.reward
ENTRY_moment = MOMENT_ORDER=$(ORDER) $(MOMENT_ENV) python -m moments.reward
ENTRY_map    = python -m embedopt.inference.inference

RUN_ARGS = --dump_dir out/$(SYSTEM)/$(RUN_NAME) --model_name $(MODEL) \
           --load_checkpoint_dir $(CKPT_DIR) --use_template false \
           --enable_diffusion_shared_vars_cache false \
           --sample_diffusion.N_step $(N_STEP) --seeds $(SEEDS)

.PHONY: cfg maps run check

run:
	python -c "import json; j = json.load(open('$(CFG)'))[0]; j['name'] = '$(SYSTEM)_$(RUN_NAME)'; \
	j['algorithm'] = {'type': 'base', 'params': {}} if '$(METHOD)' == 'base' else \
	{'type': 'embedopt', 'params': {'reward_scale_s': $(LR), 'reward_scale_z': $(LR)}}; \
	j['reward']['params']['resolution'] = 5.0 if '$(METHOD)' == 'map' else $(RES); \
	json.dump([j], open('$(RUN_CFG)', 'w'), indent=2)"
	$(ENTRY_$(METHOD)) --input_json_path $(RUN_CFG) $(RUN_ARGS)

check:
	python -m moments.check --ref $(BENCH)/pdb_clean/$(SYSTEM).pdb \
	       --runs $(sort $(wildcard out/$(SYSTEM)/base out/$(SYSTEM)/map_lr* out/$(SYSTEM)/moment*_lr*)) \
	       --out out/$(SYSTEM)/check.png --csv out/$(SYSTEM)/summary.csv

maps:
	python $(BENCH)/04_generate_map.py

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
