#!/bin/bash -l
#SBATCH --job-name=molprobity
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/molprobity_%A_%a.out
# Partition not set — supply via SBATCH_PARTITION env var or `sbatch -p ...`

# Run phenix.molprobity on prediction structures from steps 05/09.
#
# Usage examples:
#   # Default: relaxed PDBs in trajectory/
#   sbatch --array=0-23 10_run_molprobity.sh steering_output_k20
#
#   # Unrelaxed CIFs in predictions/
#   sbatch --array=0-23 10_run_molprobity.sh steering_output_k20 \
#       --subdir predictions --file-pattern '*sample_0.cif' --output-prefix molprobity_unrelaxed
#
#   # Specific system
#   sbatch --array=0 10_run_molprobity.sh steering_output_k20 1IXC_A
#
#   # Force re-run (overwrite existing logs)
#   sbatch --array=0-23 10_run_molprobity.sh steering_output_k20 --force
#
# Output files: {seed_dir}/{output-prefix}.{out,log}
#
# Prerequisites: phenix on PATH. Set PHENIX_ENV_SCRIPT to the path of phenix_env.sh
# and the script will source it (e.g. PHENIX_ENV_SCRIPT=/path/to/phenix-X.Y.Z/phenix_env.sh).
# Alternatively, source phenix_env.sh manually before sbatch.

# Configuration — BASE_DIR resolved from REPO_ROOT (SLURM copies the script
# into a spool dir, so BASH_SOURCE can't be used to locate it).
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
BASE_DIR="${REPO_ROOT}/examples/AF_distance_benchmark"
SEEDS=(101 42 16)

# Defaults — relaxed PDBs from the in-loop reward energy minimization
OUTPUT_DIR=""
SYSTEM_NAME=""
FORCE_RERUN=false
SUBDIR="trajectory"
FILE_PATTERN="step_*_relaxed.pdb"
OUTPUT_PREFIX="molprobity_relaxed"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force|-f)
            FORCE_RERUN=true
            shift
            ;;
        --subdir)
            SUBDIR="$2"
            shift 2
            ;;
        --file-pattern)
            FILE_PATTERN="$2"
            shift 2
            ;;
        --output-prefix)
            OUTPUT_PREFIX="$2"
            shift 2
            ;;
        *)
            if [[ -z "$OUTPUT_DIR" ]]; then
                OUTPUT_DIR="${BASE_DIR}/$1"
            elif [[ -z "$SYSTEM_NAME" ]]; then
                SYSTEM_NAME="$1"
            fi
            shift
            ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "ERROR: must pass an output dir name (e.g. steering_output_k20) as first arg"
    exit 1
fi
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "ERROR: Output directory not found: $OUTPUT_DIR"
    exit 1
fi

mkdir -p slurm_logs

# Activate Phenix from PHENIX_ENV_SCRIPT if provided; sanity-check it's on PATH.
if [[ -n "$PHENIX_ENV_SCRIPT" ]]; then
    source "$PHENIX_ENV_SCRIPT"
fi
command -v phenix.molprobity >/dev/null || {
    echo "ERROR: phenix.molprobity not on PATH."
    echo "Set PHENIX_ENV_SCRIPT=/path/to/phenix_env.sh and re-submit, or activate Phenix in your shell before sbatch."
    exit 1
}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
TOTAL_TASKS=${SLURM_ARRAY_TASK_COUNT:-1}

echo "========================================"
echo "MolProbity Validation"
echo "Output Dir:    $OUTPUT_DIR"
echo "Subdir:        $SUBDIR"
echo "File Pattern:  $FILE_PATTERN"
echo "Output Prefix: $OUTPUT_PREFIX"
echo "Task ${TASK_ID}/${TOTAL_TASKS}"
[ "$FORCE_RERUN" = true ] && echo "Mode: FORCE RERUN"
echo "========================================"

run_molprobity() {
    local input_file="$1"
    local seed_dir="$2"
    local output_prefix="${seed_dir}/${OUTPUT_PREFIX}"
    local log_file="${output_prefix}.log"

    if [ ! -f "$input_file" ]; then
        echo "    WARNING: Input file not found: $input_file"
        return 1
    fi
    if [ -f "$log_file" ] && [ "$FORCE_RERUN" = false ]; then
        echo "    skip (exists): $(basename $log_file)"
        return 0
    fi

    echo "    Running MolProbity on $(basename $input_file)..."
    phenix.molprobity "$input_file" output.prefix="$output_prefix" >& "$log_file" \
        && echo "    OK" \
        || echo "    FAILED (see $log_file)"
}

process_config() {
    local system="$1"
    local config_name="$2"
    local config_dir="${OUTPUT_DIR}/${system}/${config_name}"

    [ -d "$config_dir" ] || { echo "  WARNING: missing config dir: $config_dir"; return 1; }
    echo ""
    echo "  Processing: ${config_name}"

    for seed in "${SEEDS[@]}"; do
        local seed_dir="${config_dir}/seed_${seed}"
        local search_dir="${seed_dir}/${SUBDIR}"
        if [ ! -d "$search_dir" ]; then
            echo "    seed ${seed}: ${SUBDIR}/ not found"
            continue
        fi
        local structure_file=$(find "$search_dir" -name "$FILE_PATTERN" -type f | head -1)
        if [ -z "$structure_file" ]; then
            echo "    seed ${seed}: no file matching '${FILE_PATTERN}' in ${SUBDIR}/"
            continue
        fi
        echo "    seed ${seed}:"
        run_molprobity "$structure_file" "$seed_dir"
    done
}

# Build system list
if [ -n "$SYSTEM_NAME" ]; then
    SYSTEMS=("$SYSTEM_NAME")
else
    SYSTEMS=()
    for sys_dir in "${OUTPUT_DIR}"/*; do
        [ -d "$sys_dir" ] && SYSTEMS+=($(basename "$sys_dir"))
    done
fi
[ ${#SYSTEMS[@]} -eq 0 ] && { echo "ERROR: No systems found"; exit 1; }
echo "Total systems: ${#SYSTEMS[@]}"

# Collect (system, config) pairs
ALL_CONFIGS=()
for system in "${SYSTEMS[@]}"; do
    for config_dir in "${OUTPUT_DIR}/${system}"/*; do
        [ -d "$config_dir" ] || continue
        config_name=$(basename "$config_dir")
        [[ "$config_name" == "${system}_"* ]] || continue
        ALL_CONFIGS+=("${system}|${config_name}")
    done
done
total_configs=${#ALL_CONFIGS[@]}
[ ${total_configs} -eq 0 ] && { echo "ERROR: No configurations found"; exit 1; }
echo "Total configurations: ${total_configs}"

# Distribute across array tasks
cpt=$((total_configs / TOTAL_TASKS))
rem=$((total_configs % TOTAL_TASKS))
if [ $TASK_ID -lt $rem ]; then
    start_idx=$((TASK_ID * (cpt + 1)))
    end_idx=$((start_idx + cpt + 1))
else
    start_idx=$((rem * (cpt + 1) + (TASK_ID - rem) * cpt))
    end_idx=$((start_idx + cpt))
fi
echo "Task ${TASK_ID} processing configs ${start_idx} to $((end_idx - 1))"

processed=0
for ((i=start_idx; i<end_idx && i<total_configs; i++)); do
    config_entry="${ALL_CONFIGS[$i]}"
    system="${config_entry%%|*}"
    config_name="${config_entry##*|}"
    echo ""
    echo "[$((i+1))/${total_configs}] ${system} / ${config_name}"
    process_config "$system" "$config_name"
    processed=$((processed + 1))
done

echo ""
echo "========================================"
echo "Task ${TASK_ID}: ${processed} configs processed"
echo "========================================"
