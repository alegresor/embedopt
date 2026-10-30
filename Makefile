# Usage:
#   make cfg_8AHU_A                 # build x* and the inference JSON for 8AHU_A (one EmbedOpt job, lr 0.1)
#   make stage3a                    # Stage 3a oracle on that JSON
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

.PHONY: stage1mwe stage3a cfg_8AHU_A

stage3a:
	python stage3a_oracle.py $(RUN_ARGS)

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