# Usage:
#   make cfg SYSTEM=8AHU_A                        # x*, MSA and the base JSON config for SYSTEM
#   make maps                                     # 5 A synthetic maps for all systems (needed by METHOD=map)
#   make run SYSTEM=8K9Z_A METHOD=moment ORDER=3 LR=0.1 [DW=1] [RES=4.0]
#   make check SYSTEM=8K9Z_A                      # summary table + CSV + figure over all runs in out/$(SYSTEM)/
#   make check SYSTEM=8K9Z_A FIG=check_dw RUNS="base map_lr0.5 moment3_lr0.1 moment3dw_lr0.1"   # subset figure
#   INIT=<structure.cif> SIGMA0=<A> runs partial diffusion: start from INIT noised to SIGMA0, run the remaining steps;
#   INIT may contain {seed} and wildcards (per-seed start, e.g. from a previous run's predictions);
#   SAVE_EMB=1 saves the final optimized embedding; WARM=<embeddings_step_*.pt> starts from a saved embedding;
#   TAG=_suffix is appended to the run name (e.g. TAG=_warm)
#
# METHOD=base   : unguided Protenix (second-order moment loss logged only)     -> out/$(SYSTEM)/base
# METHOD=moment : EmbedOpt + oracle uniform-mu moments of order ORDER (2 or 3)  -> out/$(SYSTEM)/moment<ORDER>[dw]_lr<LR>[_res<RES>]
#                 DW=1 weights the moment losses by the denoiser's posterior positional uncertainty (Debye-Waller)
# METHOD=map    : EmbedOpt + Li et al.'s MapReward on the 5 A synthetic map      -> out/$(SYSTEM)/map_lr<LR>
#                 (moments.map_entry = their entry point + optional partial diffusion)
BENCH      ?= examples/synthetic_map_benchmark
SYSTEM     ?= 8AHU_A
METHOD     ?= moment
ORDER      ?= 2
LR         ?= 0.1
RES        ?= 5.0
LB         ?= 24
LAMBDA3    ?= 1.0
NKAPPA     ?= 16
DW         ?= 0
FIG        ?= check
RUNS       ?=
TAG        ?=
WARM       ?=
SAVE_EMB   ?= 0
INIT       ?=
SIGMA0     ?= 4.0
CFG        ?= $(BENCH)/stage3a_cfgs/$(SYSTEM)_benchmark.json
MODEL      ?= protenix_base_default_v0.5.0
CKPT_DIR   ?= $(HOME)/.cache/protenix/checkpoint
N_STEP     ?= 100
SEEDS      ?= 101,102,103,104,105

export LAYERNORM_TYPE := torch_layernorm

RES_TAG    = $(if $(filter 5.0,$(RES)),,_res$(RES))
RUN_NAME   = $(if $(filter base,$(METHOD)),base,$(if $(filter map,$(METHOD)),map_lr$(LR),moment$(ORDER)$(if $(filter 1,$(DW)),dw,)_lr$(LR)$(RES_TAG)))$(TAG)
RUN_CFG    = $(BENCH)/stage3a_cfgs/$(SYSTEM)_$(RUN_NAME).json
PARTIAL_ENV = $(if $(INIT),PARTIAL_INIT=$(abspath $(INIT)) PARTIAL_SIGMA=$(SIGMA0),)
MOMENT_ENV = $(PARTIAL_ENV) MOMENT_DW=$(DW) MOMENT_LB=$(LB) MOMENT_LAMBDA3=$(LAMBDA3) MOMENT_NKAPPA=$(NKAPPA)
ENTRY_base   = MOMENT_ORDER=2 MOMENT_DW=0 MOMENT_LB=$(LB) MOMENT_NKAPPA=$(NKAPPA) python -m moments.reward
ENTRY_moment = MOMENT_ORDER=$(ORDER) $(MOMENT_ENV) python -m moments.reward
ENTRY_map    = $(PARTIAL_ENV) python -m moments.map_entry

RUN_ARGS = --dump_dir out/$(SYSTEM)/$(RUN_NAME) --model_name $(MODEL) \
           --load_checkpoint_dir $(CKPT_DIR) --use_template false \
           --enable_diffusion_shared_vars_cache false \
           --sample_diffusion.N_step $(N_STEP) --seeds $(SEEDS)

.PHONY: cfg maps run check imagechecks

run:
	python -c "import json; j = json.load(open('$(CFG)'))[0]; j['name'] = '$(SYSTEM)_$(RUN_NAME)'; \
	j['algorithm'] = {'type': 'base', 'params': {}} if '$(METHOD)' == 'base' else \
	{'type': 'embedopt', 'params': {'reward_scale_s': $(LR), 'reward_scale_z': $(LR)}}; \
	j['reward']['params']['resolution'] = 5.0 if '$(METHOD)' == 'map' else $(RES); \
	p = j['algorithm']['params']; \
	p.update({'override_embeddings_path': '$(abspath $(WARM))'} if '$(WARM)' else {}); \
	p.update({'save_embeddings_every_n_steps': -1} if '$(SAVE_EMB)' == '1' else {}); \
	json.dump([j], open('$(RUN_CFG)', 'w'), indent=2)"
	$(if $(WARM),test -f $(WARM) || (echo 'missing WARM embedding: $(WARM)'; exit 1),true)
	$(if $(INIT),$(if $(findstring {seed},$(INIT)),true,test -f $(INIT) || (echo 'missing INIT structure: $(INIT)'; exit 1)),true)
	$(ENTRY_$(METHOD)) --input_json_path $(RUN_CFG) $(RUN_ARGS)

check:
	python -m moments.check --ref $(BENCH)/pdb_clean/$(SYSTEM).pdb \
	       --runs $(if $(RUNS),$(addprefix out/$(SYSTEM)/,$(RUNS)),$(sort $(wildcard out/$(SYSTEM)/base out/$(SYSTEM)/map_lr* out/$(SYSTEM)/moment*_lr*))) \
	       --out out/$(SYSTEM)/$(FIG).png --csv out/$(SYSTEM)/$(if $(RUNS),$(FIG),summary).csv

maps:
	python $(BENCH)/04_generate_map.py

# Unit checks for image-based moments (oracle particle, uniform poses): out/<SYS>/image_checks.{json,png}
IC_SYSTEMS ?= 8AHU_A 8K9Z_A 8H1I_A
NMAX       ?= 1000000
imagechecks:
	python -m moments.image_checks --systems $(IC_SYSTEMS) --nmax $(NMAX) 2>&1 | tee out/image_checks.log

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
