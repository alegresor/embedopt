#!/bin/bash -l

# Run base sampling on each system in json_minimal/.
#
# Usage:
#   sbatch 05_submit_base_jobs.sh                     # all systems
#   sbatch 05_submit_base_jobs.sh 1IXC_A 2BJ7_A       # specific systems
#   sbatch --array=0-2 05_submit_base_jobs.sh         # parallelize across array tasks
#   SEEDS=101 sbatch 05_submit_base_jobs.sh 1IXC_A    # single-seed smoke test
#
# Output: base_output/{system}/{system}_minimal/seed_{N}/predictions/...

# SLURM directives — partition not set; supply via SBATCH_PARTITION env var or `sbatch -p ...`
#SBATCH -C h100|a100-80gb|a100-40gb
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 1-00:00:00
#SBATCH --output=slurm_logs/%x_%j_%a.out
#SBATCH -J "AF_dist_base"

# Project root and config — override via env vars
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
BENCHMARK_DIR="${REPO_ROOT}/examples/AF_distance_benchmark"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the directory containing protenix_base_default_v0.5.0.pt}"

JSON_FOLDER="json_minimal"
OUTPUT_FOLDER="base_output"
MODEL_NAME="protenix_base_default_v0.5.0"
DIFFUSION_STEPS=200
SEEDS="${SEEDS:-101,42,16}"
N_CYCLE=10

cd "$BENCHMARK_DIR"
mkdir -p slurm_logs

# Build list of systems to run
systems=()
if [ $# -eq 0 ]; then
    echo "No systems specified, running all systems in ${JSON_FOLDER}/"
    for json_file in "${JSON_FOLDER}"/*_minimal-update-msa.json; do
        if [ -f "$json_file" ]; then
            basename=$(basename "$json_file" _minimal-update-msa.json)
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

echo "Total systems: ${#systems[@]}"
echo "Configuration: N_step=${DIFFUSION_STEPS}, N_cycle=${N_CYCLE}, seeds=${SEEDS}"

# Determine which subset this task processes
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    start_idx=0
    end_idx=${#systems[@]}
    echo "Sequential mode: processing all ${#systems[@]} systems"
else
    task_id=$SLURM_ARRAY_TASK_ID
    num_tasks=$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))
    total_systems=${#systems[@]}
    base_count=$((total_systems / num_tasks))
    remainder=$((total_systems % num_tasks))
    if [ $task_id -lt $remainder ]; then
        count=$((base_count + 1))
        start_idx=$((task_id * count))
    else
        count=$base_count
        start_idx=$((remainder * (base_count + 1) + (task_id - remainder) * base_count))
    fi
    end_idx=$((start_idx + count))
    echo "Array task $task_id of $num_tasks: processing systems $start_idx to $((end_idx - 1))"
fi

for ((i=start_idx; i<end_idx; i++)); do
    system="${systems[$i]}"
    echo ""
    echo "===== Processing $((i - start_idx + 1))/$((end_idx - start_idx)): $system ====="

    input_json="${BENCHMARK_DIR}/${JSON_FOLDER}/${system}_minimal-update-msa.json"
    output_dir="${BENCHMARK_DIR}/${OUTPUT_FOLDER}/${system}"

    if [ ! -f "$input_json" ]; then
        echo "ERROR: input JSON not found: $input_json"
        continue
    fi

    LAYERNORM_TYPE=torch_layernorm pixi run python -m embedopt.inference.inference \
        --input_json_path "$input_json" \
        --dump_dir "$output_dir" \
        --model_name "$MODEL_NAME" \
        --load_checkpoint_dir "$CHECKPOINT_DIR" \
        --use_template false \
        --enable_diffusion_shared_vars_cache false \
        --seeds "$SEEDS" \
        --sample_diffusion.N_step "$DIFFUSION_STEPS" \
        --model.N_cycle "$N_CYCLE"

    echo "Completed: $system"
done

echo ""
echo "===== Done ====="
