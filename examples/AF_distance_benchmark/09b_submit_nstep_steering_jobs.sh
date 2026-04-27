#!/bin/bash -l

# Run N_step robustness sweep: each job in benchmark_cfgs_nstep_k${K}/*.json carries
# its own n_step field, parsed via jq and forwarded as --sample_diffusion.N_step.
# One inference call per job (not per system) since N_step varies within a system.
#
# Usage:
#   sbatch 09b_submit_nstep_steering_jobs.sh                    # all systems
#   sbatch 09b_submit_nstep_steering_jobs.sh 4EGL_A 1IXC_A      # specific systems
#   sbatch --array=0-4 09b_submit_nstep_steering_jobs.sh        # parallelize across jobs
#   K=5 sbatch 09b_submit_nstep_steering_jobs.sh                # K=5 sweep instead of default 20
#   SEEDS=101 sbatch 09b_submit_nstep_steering_jobs.sh 1IXC_A   # single-seed smoke test
#
# Reads:  benchmark_cfgs_nstep_k${K}/{system}_benchmark.json (output of step 08b)
# Output: steering_output_nstep_k${K}/{system}/{job_name}/seed_{N}/...

# SLURM directives — partition not set; supply via SBATCH_PARTITION env var or `sbatch -p ...`
#SBATCH -C h100|a100-80gb|a100-40gb
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 1-00:00:00
#SBATCH --output=slurm_logs/%x_%j_%a.out
#SBATCH -J "AF_dist_nstep"

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
BENCHMARK_DIR="${REPO_ROOT}/examples/AF_distance_benchmark"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the directory containing protenix_base_default_v0.5.0.pt}"

K="${K:-20}"
CONFIG_FOLDER="benchmark_cfgs_nstep_k${K}"
OUTPUT_FOLDER="steering_output_nstep_k${K}"
MODEL_NAME="protenix_base_default_v0.5.0"
SEEDS="${SEEDS:-101,42,16}"
N_CYCLE=10

cd "$BENCHMARK_DIR"
mkdir -p slurm_logs

if ! command -v jq &> /dev/null; then
    echo "ERROR: jq not found. Install with: pixi add jq  (or: mamba install -c conda-forge jq)"
    exit 1
fi

# Build list of systems to run
systems=()
if [ $# -eq 0 ]; then
    echo "No systems specified, running all systems in ${CONFIG_FOLDER}/"
    for json_file in "${CONFIG_FOLDER}"/*_benchmark.json; do
        if [ -f "$json_file" ]; then
            basename=$(basename "$json_file" _benchmark.json)
            systems+=("$basename")
        fi
    done
else
    systems=("$@")
    echo "Running specified systems: ${systems[@]}"
fi

if [ ${#systems[@]} -eq 0 ]; then
    echo "ERROR: No systems found to run"
    exit 1
fi

# Flatten into per-job entries: "system:job_idx:n_step:job_name"
all_jobs=()
for system in "${systems[@]}"; do
    json_file="${BENCHMARK_DIR}/${CONFIG_FOLDER}/${system}_benchmark.json"
    if [ ! -f "$json_file" ]; then
        echo "WARNING: missing config $json_file"
        continue
    fi
    num_jobs=$(jq 'length' "$json_file")
    for ((j=0; j<num_jobs; j++)); do
        n_step=$(jq -r ".[$j].n_step" "$json_file")
        job_name=$(jq -r ".[$j].name" "$json_file")
        if [ "$n_step" = "null" ] || [ -z "$n_step" ]; then
            echo "WARNING: job $j in $system has no n_step field, skipping"
            continue
        fi
        all_jobs+=("${system}:${j}:${n_step}:${job_name}")
    done
done

if [ ${#all_jobs[@]} -eq 0 ]; then
    echo "ERROR: No jobs found"
    exit 1
fi

echo "K = ${K}, total jobs: ${#all_jobs[@]} (from ${#systems[@]} systems)"
echo "Configuration: N_cycle=${N_CYCLE}, seeds=${SEEDS}, N_step varies per job"

# Determine which subset of jobs (not systems) this array task processes
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    start_idx=0
    end_idx=${#all_jobs[@]}
    echo "Sequential mode: processing all ${#all_jobs[@]} jobs"
else
    task_id=$SLURM_ARRAY_TASK_ID
    num_tasks=$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))
    total=${#all_jobs[@]}
    base_count=$((total / num_tasks))
    remainder=$((total % num_tasks))
    if [ $task_id -lt $remainder ]; then
        count=$((base_count + 1))
        start_idx=$((task_id * count))
    else
        count=$base_count
        start_idx=$((remainder * (base_count + 1) + (task_id - remainder) * base_count))
    fi
    end_idx=$((start_idx + count))
    echo "Array task $task_id of $num_tasks: processing jobs $start_idx to $((end_idx - 1))"
fi

for ((i=start_idx; i<end_idx && i<${#all_jobs[@]}; i++)); do
    job_entry="${all_jobs[$i]}"
    IFS=':' read -r system job_idx n_step job_name <<< "$job_entry"

    echo ""
    echo "===== Job $((i - start_idx + 1))/$((end_idx - start_idx)): $job_name (N_step=$n_step) ====="

    input_json="${BENCHMARK_DIR}/${CONFIG_FOLDER}/${system}_benchmark.json"
    output_dir="${BENCHMARK_DIR}/${OUTPUT_FOLDER}/${system}"

    if [ ! -f "$input_json" ]; then
        echo "ERROR: input JSON not found: $input_json"
        continue
    fi

    # Slice out this single job into a temp config (inference takes a list-of-jobs JSON)
    temp_json="/tmp/${system}_job${job_idx}_$$_${RANDOM}.json"
    jq ".[$job_idx] | [.]" "$input_json" > "$temp_json"

    LAYERNORM_TYPE=torch_layernorm pixi run python -m embedopt.inference.inference \
        --input_json_path "$temp_json" \
        --dump_dir "$output_dir" \
        --model_name "$MODEL_NAME" \
        --load_checkpoint_dir "$CHECKPOINT_DIR" \
        --use_template false \
        --enable_diffusion_shared_vars_cache false \
        --seeds "$SEEDS" \
        --sample_diffusion.N_step "$n_step" \
        --model.N_cycle "$N_CYCLE"

    rm -f "$temp_json"
    echo "Completed: $job_name"
done

echo ""
echo "===== Done ====="
