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
#   METHOD=map MAP=<map.mrc> MAPREF=<model.pdb> MTAG=_x : map reward on another map (e.g. a RELION reconstruction), with
#   MapReward's Kabsch frame taken from MAPREF (e.g. the prior docked into the map) instead of x*; run map_x_lr<LR>
#   DTAG=_x names moment runs on other image data (DATA=<npz>): moment<ORDER>_imgd_x..._lr<LR>
#   SPACE=image compares in image space (a, c2, c3; moments.image_moments) against DATA=exact (moments of x*) or
#   DATA=images (out/$(SYSTEM)/image_moments_N1000000.npz from `make imagechecks`) or DATA=<path.npz>;
#   run names get _imgx (exact) or _imgd (image data) before _lr. CHECK_DATA=exact|images|<npz> adds image-space
#   columns m12_img, m3_img to `make check`.
#   S3STRIDE=k takes the third-order shells every k-th (as the invariant bispectrum) instead of the lowest NB3; with
#   Q3 (|q| bound) this sets the third-order grid. Non-default grids add _c3s<k>q<Q3> to ORDER=3 image-space run names
#   (VTAG; override e.g. VTAG=_rep for a replicate) and to the image data file. Also used by check / imagetest /
#   imagechecks.
#   M12=plain|white, M3=block|perq|white select the image-space loss variants (moments.image_moments.ImageLoss);
#   non-default ones add _w12 / _m3<M3> to image-space run names (M12_DROP, M3_RCOND, M3B_RCOND: their parameters).
#
# METHOD=base   : unguided Protenix (second-order moment loss logged only)     -> out/$(SYSTEM)/base
# METHOD=moment : EmbedOpt + oracle uniform-mu moments of order ORDER (2 or 3)  -> out/$(SYSTEM)/moment<ORDER>[dw][_imgx|_imgd]_lr<LR>[_res<RES>]
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
SPACE      ?= invariant
DATA       ?= exact
CHECK_DATA ?=
NB3        ?= 8
Q3         ?= 16
C3_CHUNK   ?= 256
MAP        ?=
MAPREF     ?=
MTAG       ?=
DTAG       ?=
S3STRIDE   ?= 0
M12        ?= plain
M3         ?= block
M12_DROP   ?= 2
M3_RCOND   ?= 1e-6
M3B_RCOND  ?= 1e-9
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
SPACE_TAG  = $(if $(filter image,$(SPACE)),$(if $(filter exact,$(DATA)),_imgx,_imgd),)
C3_TAG     = $(if $(and $(filter 0,$(S3STRIDE)),$(filter 16,$(Q3))),,_c3s$(S3STRIDE)q$(Q3))
W_TAG      = $(if $(filter white,$(M12)),_w12,)$(if $(filter 3,$(ORDER)),$(if $(filter block,$(M3)),,_m3$(M3)),)
VTAG      ?= $(if $(filter image,$(SPACE)),$(if $(filter 3,$(ORDER)),$(C3_TAG),)$(W_TAG),)
DATA_SRC   = $(if $(filter exact,$(1)),exact,$(abspath $(if $(filter images,$(1)),out/$(SYSTEM)/image_moments_N1000000$(C3_TAG).npz,$(1))))
RUN_NAME   = $(if $(filter base,$(METHOD)),base,$(if $(filter map,$(METHOD)),map$(MTAG)_lr$(LR),moment$(ORDER)$(if $(filter 1,$(DW)),dw,)$(SPACE_TAG)$(DTAG)$(VTAG)_lr$(LR)$(RES_TAG)))$(TAG)
RUN_CFG    = $(BENCH)/stage3a_cfgs/$(SYSTEM)_$(RUN_NAME).json
PARTIAL_ENV = $(if $(INIT),PARTIAL_INIT=$(abspath $(INIT)) PARTIAL_SIGMA=$(SIGMA0),)
MOMENT_ENV = $(PARTIAL_ENV) MOMENT_DW=$(DW) MOMENT_LB=$(LB) MOMENT_LAMBDA3=$(LAMBDA3) MOMENT_NKAPPA=$(NKAPPA) \
             MOMENT_SPACE=$(SPACE) MOMENT_DATA=$(call DATA_SRC,$(DATA)) MOMENT_NB3=$(NB3) MOMENT_Q3=$(Q3) MOMENT_C3_CHUNK=$(C3_CHUNK) \
             MOMENT_S3STRIDE=$(S3STRIDE) MOMENT_M12=$(M12) MOMENT_M3=$(M3) MOMENT_M12_DROP=$(M12_DROP) \
             MOMENT_M3_RCOND=$(M3_RCOND) MOMENT_M3B_RCOND=$(M3B_RCOND)
ENTRY_base   = MOMENT_ORDER=2 MOMENT_DW=0 MOMENT_SPACE=invariant MOMENT_LB=$(LB) MOMENT_NKAPPA=$(NKAPPA) python -m moments.reward
ENTRY_moment = MOMENT_ORDER=$(ORDER) $(MOMENT_ENV) python -m moments.reward
ENTRY_map    = $(PARTIAL_ENV) python -m moments.map_entry

RUN_ARGS = --dump_dir out/$(SYSTEM)/$(RUN_NAME) --model_name $(MODEL) \
           --load_checkpoint_dir $(CKPT_DIR) --use_template false \
           --enable_diffusion_shared_vars_cache false \
           --sample_diffusion.N_step $(N_STEP) --seeds $(SEEDS)

.PHONY: cfg maps run check imagechecks imagetest

# Unit test of the image-space reward maps (no Protenix): port vs image_checks, floors, invariance, gradients, cost
imagetest:
	python -m moments.image_moments --systems $(IC_SYSTEMS) --nb3 $(NB3) --q3 $(Q3) --s3-stride $(S3STRIDE) \
	       --chunk $(C3_CHUNK) 2>&1 | tee out/image_moments_test$(C3_TAG).log

run:
	python -c "import json; j = json.load(open('$(CFG)'))[0]; j['name'] = '$(SYSTEM)_$(RUN_NAME)'; \
	j['algorithm'] = {'type': 'base', 'params': {}} if '$(METHOD)' == 'base' else \
	{'type': 'embedopt', 'params': {'reward_scale_s': $(LR), 'reward_scale_z': $(LR)}}; \
	j['reward']['params']['resolution'] = 5.0 if '$(METHOD)' == 'map' else $(RES); \
	rp = j['reward']['params']; \
	rp.update({'ref_map_path': '$(abspath $(MAP))'} if '$(MAP)' and '$(METHOD)' == 'map' else {}); \
	rp.update({'ref_pdb_path': '$(abspath $(MAPREF))'} if '$(MAPREF)' and '$(METHOD)' == 'map' else {}); \
	p = j['algorithm']['params']; \
	p.update({'override_embeddings_path': '$(abspath $(WARM))'} if '$(WARM)' else {}); \
	p.update({'save_embeddings_every_n_steps': -1} if '$(SAVE_EMB)' == '1' else {}); \
	json.dump([j], open('$(RUN_CFG)', 'w'), indent=2)"
	$(if $(WARM),test -f $(WARM) || (echo 'missing WARM embedding: $(WARM)'; exit 1),true)
	$(if $(INIT),$(if $(findstring {seed},$(INIT)),true,test -f $(INIT) || (echo 'missing INIT structure: $(INIT)'; exit 1)),true)
	$(ENTRY_$(METHOD)) --input_json_path $(RUN_CFG) $(RUN_ARGS)

check:
	MOMENT_NB3=$(NB3) MOMENT_Q3=$(Q3) MOMENT_S3STRIDE=$(S3STRIDE) MOMENT_C3_CHUNK=$(C3_CHUNK) MOMENT_M3_RCOND=$(M3_RCOND) MOMENT_M3B_RCOND=$(M3B_RCOND) \
	python -m moments.check --ref $(BENCH)/pdb_clean/$(SYSTEM).pdb \
	       --runs $(if $(RUNS),$(addprefix out/$(SYSTEM)/,$(RUNS)),$(sort $(wildcard out/$(SYSTEM)/base out/$(SYSTEM)/map_lr* out/$(SYSTEM)/moment*_lr*))) \
	       --out out/$(SYSTEM)/$(FIG).png --csv out/$(SYSTEM)/$(if $(RUNS),$(FIG),summary).csv \
	       $(if $(CHECK_DATA),--data $(call DATA_SRC,$(CHECK_DATA)),)

maps:
	python $(BENCH)/04_generate_map.py

# Unit checks for image-based moments (oracle particle, uniform poses): out/<SYS>/image_checks.{json,png}
IC_SYSTEMS ?= 8AHU_A 8K9Z_A 8H1I_A
NMAX       ?= 1000000
imagechecks:
	python -m moments.image_checks --systems $(IC_SYSTEMS) --nmax $(NMAX) --nb3 $(NB3) --q3 $(Q3) --s3-stride $(S3STRIDE) \
	       2>&1 | tee out/image_checks$(C3_TAG).log

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
