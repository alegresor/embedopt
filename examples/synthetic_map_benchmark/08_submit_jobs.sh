#!/bin/bash -l

# Must be submitted from within the synthetic_map_benchmark/ directory:
#
#   cd examples/synthetic_map_benchmark
#   sbatch 08_submit_jobs.sh                   # all systems
#   sbatch 08_submit_jobs.sh 8AHU_A 8ANE_A    # specific systems
#   sbatch --array=0-4 08_submit_jobs.sh       # parallel job array
#   sbatch 08_submit_jobs.sh --test            # 3 systems only

# SLURM directives — partition not set; supply via SBATCH_PARTITION env var or `sbatch -p ...`
#SBATCH -C h100|a100-80gb|a100-40gb
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH -t 1-00:00:00
#SBATCH --output=slurm_logs/log_%x_%j.out
#SBATCH -J "map_benchmark"

# Must be submitted from within the synthetic_map_benchmark/ directory:
#   cd examples/synthetic_map_benchmark && sbatch 08_submit_jobs.sh
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "${SCRIPT_DIR}/slurm_logs"

export LAYERNORM_TYPE=torch_layernorm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

model_name="protenix_base_default_v0.5.0"
# Checkpoint is auto-downloaded here on first run if not already present.
# Override by setting CHECKPOINT_DIR in your environment before calling this script.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/.cache/protenix/checkpoint}"
config_folder="benchmark_cfgs_5A"
output_folder="benchmark_output"

DIFFUSION_STEPS=100
SEEDS="101,42,16"
N_CYCLE=10

# Build list of systems to run
systems=()

if [ "${1:-}" == "--test" ]; then
    echo "Test mode: running on 3 systems"
    systems=("8AHU_A" "8ANE_A" "8AU1_A")
elif [ $# -eq 0 ]; then
    echo "No systems specified, running all systems in ${config_folder}/"
    for json_file in "${SCRIPT_DIR}/${config_folder}"/*_benchmark.json; do
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

echo "Total systems to run: ${#systems[@]}"
echo "Configuration: N_step=${DIFFUSION_STEPS}, N_cycle=${N_CYCLE}, seeds=${SEEDS}"

run_system() {
    local system="$1"
    local input_json="${SCRIPT_DIR}/${config_folder}/${system}_benchmark.json"
    local output_dir="${SCRIPT_DIR}/${output_folder}/${system}"

    if [ ! -f "$input_json" ]; then
        echo "ERROR: Config file not found: $input_json"
        return 1
    fi

    echo "Input:  $input_json"
    echo "Output: $output_dir"

    cd "$REPO_ROOT"
    pixi run python -m embedopt.inference.inference \
        --model_name ${model_name} \
        --load_checkpoint_dir "$CHECKPOINT_DIR" \
        --seeds $SEEDS \
        --dump_dir "$output_dir" \
        --input_json_path "$input_json" \
        --use_template false \
        --enable_diffusion_shared_vars_cache false \
        --sample_diffusion.N_step $DIFFUSION_STEPS \
        --model.N_cycle $N_CYCLE
}

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Running in sequential mode"
    for i in "${!systems[@]}"; do
        system="${systems[$i]}"
        echo ""
        echo "===== Processing system $((i+1))/${#systems[@]}: $system ====="
        run_system "$system"
        echo "Completed: $system"
    done
else
    task_id=$SLURM_ARRAY_TASK_ID
    num_tasks=$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))
    total_systems=${#systems[@]}

    base_count=$((total_systems / num_tasks))
    remainder=$((total_systems % num_tasks))

    if [ $task_id -lt $remainder ]; then
        systems_for_this_task=$((base_count + 1))
        start_idx=$((task_id * (base_count + 1)))
    else
        systems_for_this_task=$base_count
        start_idx=$((remainder * (base_count + 1) + (task_id - remainder) * base_count))
    fi

    end_idx=$((start_idx + systems_for_this_task))

    echo ""
    echo "===== Job Array Task $task_id ====="
    echo "Processing systems: $start_idx to $((end_idx - 1)) ($systems_for_this_task systems)"
    echo ""

    for ((i=start_idx; i<end_idx; i++)); do
        system="${systems[$i]}"
        echo "--- System $((i - start_idx + 1))/$systems_for_this_task: $system ---"
        run_system "$system"
        echo "Completed: $system"
        echo ""
    done

    echo "===== Task $task_id completed ====="
fi
