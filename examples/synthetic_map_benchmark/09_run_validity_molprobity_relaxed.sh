#!/bin/bash -l
#SBATCH --job-name=map_benchmark_validation
# SLURM directives — partition not set; supply via SBATCH_PARTITION env var or `sbatch -p ...`
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/validation_%A_%a.out

# Runs phenix.molprobity and phenix.validation_cryoem on the energy-relaxed
# trajectory files produced by MapReward during inference.
# Before validation, B-factors in step_99_relaxed.pdb are set to a uniform
# value (default: 50) since inference outputs zeros which affect CC scoring.
#
# Requires Phenix (license required): https://phenix-online.org/download
# Set PHENIX_ENV_SCRIPT to the phenix_env.sh path before submitting:
#   export PHENIX_ENV_SCRIPT=/path/to/phenix-X.Y.Z/phenix_env.sh
#
# Must be submitted from within the synthetic_map_benchmark/ directory:
#
#   cd examples/synthetic_map_benchmark
#   sbatch --array=0-12 09_run_validity_molprobity_relaxed.sh           # all systems
#   sbatch --array=0-5  09_run_validity_molprobity_relaxed.sh --test    # 3 systems only
#   sbatch --array=0-12 09_run_validity_molprobity_relaxed.sh 8AHU_A    # specific system
#   sbatch --array=0-12 09_run_validity_molprobity_relaxed.sh --force   # rerun existing

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

mkdir -p "${SCRIPT_DIR}/slurm_logs"

# Activate Phenix from PHENIX_ENV_SCRIPT if provided; sanity-check it's on PATH.
if [[ -n "$PHENIX_ENV_SCRIPT" ]]; then
    source "$PHENIX_ENV_SCRIPT"
fi
command -v phenix.molprobity >/dev/null || {
    echo "ERROR: phenix.molprobity not on PATH."
    echo "Set PHENIX_ENV_SCRIPT=/path/to/phenix_env.sh and re-submit, or activate Phenix in your shell before sbatch."
    exit 1
}

echo "phenix --version"
phenix.version
echo ""

# Configuration
OUTPUT_DIR="${SCRIPT_DIR}/benchmark_output"
MAP_DIR="${SCRIPT_DIR}/synthetic_maps"
TARGET_PDB="trajectory/step_99_relaxed.pdb"
SEEDS=(16 42 101)
RESOLUTIONS=(5.0)
BFACTOR=50

# Parse arguments
SYSTEM_NAME=""
FORCE_RERUN=false
TEST_MODE=false
LEARNING_RATES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force|-f)
            FORCE_RERUN=true
            shift
            ;;
        --test)
            TEST_MODE=true
            shift
            ;;
        --resolutions)
            IFS=',' read -ra RESOLUTIONS <<< "$2"
            shift 2
            ;;
        --lrs|--learning_rates)
            IFS=',' read -ra LEARNING_RATES <<< "$2"
            shift 2
            ;;
        --bfactor)
            BFACTOR="$2"
            shift 2
            ;;
        *)
            if [[ -z "$SYSTEM_NAME" ]]; then
                SYSTEM_NAME="$1"
            fi
            shift
            ;;
    esac
done

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
TOTAL_TASKS=${SLURM_ARRAY_TASK_COUNT:-1}

echo "========================================"
echo "Map Benchmark Validation"
echo "Task ${TASK_ID}/${TOTAL_TASKS}"
echo "B-factor: ${BFACTOR}"
[ "$TEST_MODE" = true ] && echo "Mode: TEST (3 systems)"
[ "$FORCE_RERUN" = true ] && echo "Mode: FORCE RERUN"
echo "========================================"
echo ""

get_resolution() {
    local config_name="$1"
    if [[ $config_name =~ _([0-9]+\.?[0-9]*)A$ ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo ""
    fi
}

get_learning_rate() {
    local config_name="$1"
    if [[ $config_name =~ _([0-9]+\.?[0-9]*)_[0-9]+\.?[0-9]*A$ ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo ""
    fi
}

# Set all B-factors to a uniform value in-place (backs up original as .bak)
fix_bfactors() {
    local pdb_file="$1"
    local bfac="$2"

    if [ ! -f "$pdb_file" ]; then
        return 1
    fi

    # Skip if already set
    local first_bfac
    first_bfac=$(awk '/^ATOM|^HETATM/{printf "%.2f", substr($0,61,6)+0; exit}' "$pdb_file")
    if [ "$first_bfac" = "$(printf '%.2f' "$bfac")" ]; then
        echo "    B-factors already ${bfac}, skipping"
        return 0
    fi

    cp "$pdb_file" "${pdb_file}.bak"
    awk -v bfac="$bfac" '
        /^ATOM|^HETATM/ {
            printf "%s%6.2f%s\n", substr($0,1,60), bfac, substr($0,67)
            next
        }
        { print }
    ' "${pdb_file}.bak" > "$pdb_file"

    echo "    B-factors set to ${bfac} (backup: $(basename "${pdb_file}").bak)"
}

run_molprobity() {
    local pdb_file="$1"
    local output_prefix="$2"
    local log_file="${output_prefix}_molprobity.log"

    if [ ! -f "$pdb_file" ]; then
        echo "    WARNING: PDB not found: $pdb_file"
        return 1
    fi

    if [ -f "$log_file" ] && [ "$FORCE_RERUN" = false ]; then
        echo "    MolProbity log exists, skipping"
        return 0
    fi

    echo "    Running MolProbity..."
    phenix.molprobity "$pdb_file" output.prefix="$output_prefix" >& "$log_file"
    [ $? -eq 0 ] && echo "    MolProbity completed" || echo "    MolProbity failed"
}

run_validation() {
    local pdb_file="$1"
    local map_file="$2"
    local resolution="$3"
    local output_log="$4"

    if [ ! -f "$pdb_file" ]; then
        echo "    WARNING: PDB not found: $pdb_file"
        return 1
    fi

    if [ ! -f "$map_file" ]; then
        echo "    WARNING: Map not found: $map_file"
        return 1
    fi

    if [ -f "$output_log" ] && [ "$FORCE_RERUN" = false ]; then
        echo "    Validation log exists, skipping"
        return 0
    fi

    echo "    Running validation_cryoem (${resolution}A)..."
    phenix.validation_cryoem \
        "$pdb_file" "$map_file" \
        run="*model_vs_data" \
        resolution="$resolution" \
        >& "$output_log"

    if [ $? -eq 0 ]; then
        echo "    Validation completed"
        if grep -q "CC_mask" "$output_log"; then
            CC=$(grep "CC_mask" "$output_log" | awk '{print $NF}')
            echo "      CC_mask: $CC"
        fi
    else
        echo "    Validation failed"
    fi
}

process_config() {
    local system="$1"
    local config_name="$2"
    local config_dir="${OUTPUT_DIR}/${system}/${config_name}"

    if [ ! -d "$config_dir" ]; then
        echo "  WARNING: Config directory not found: $config_dir"
        return 1
    fi

    echo ""
    echo "  Processing: ${config_name}"

    local is_base=false
    [[ $config_name == *"base"* ]] && is_base=true

    local resolution
    resolution=$(get_resolution "$config_name")

    for seed in "${SEEDS[@]}"; do
        local seed_dir="${config_dir}/seed_${seed}"
        local pdb_file="${seed_dir}/${TARGET_PDB}"

        if [ ! -f "$pdb_file" ]; then
            echo "    Seed ${seed}: PDB not found"
            continue
        fi

        echo "    Seed ${seed}:"

        # Fix B-factors before validation
        fix_bfactors "$pdb_file" "$BFACTOR"

        # MolProbity
        run_molprobity "$pdb_file" "${seed_dir}/molprobity_relaxed"

        # validation_cryoem
        if [ "$is_base" = true ]; then
            for map_path in "${MAP_DIR}"/*/"${system}.map"; do
                if [ -f "$map_path" ]; then
                    local map_res_dir
                    map_res_dir=$(basename "$(dirname "$map_path")")
                    if [[ $map_res_dir =~ ^([0-9]+\.?[0-9]*)A$ ]]; then
                        local res="${BASH_REMATCH[1]}"
                        local res_int=${res%.*}
                        run_validation "$pdb_file" "$map_path" "$res" "${seed_dir}/validation_relaxed_${res_int}A.log"
                    fi
                fi
            done
        else
            if [ -n "$resolution" ]; then
                if [[ $resolution =~ ^[0-9]+$ ]]; then
                    local map_res_dir="${resolution}.0A"
                else
                    local map_res_dir="${resolution}A"
                fi
                run_validation "$pdb_file" "${MAP_DIR}/${map_res_dir}/${system}.map" "$resolution" "${seed_dir}/validation_relaxed.log"
            else
                echo "    WARNING: Could not determine resolution from config name"
            fi
        fi
    done
}

# Collect systems
if [ "$TEST_MODE" = true ]; then
    SYSTEMS=("8AHU_A" "8ANE_A" "8AU1_A")
    echo "Test mode: running on ${#SYSTEMS[@]} systems"
elif [ -n "$SYSTEM_NAME" ]; then
    SYSTEMS=("$SYSTEM_NAME")
    echo "Processing specified system: $SYSTEM_NAME"
else
    SYSTEMS=()
    for sys_dir in "${OUTPUT_DIR}"/*; do
        [ -d "$sys_dir" ] && SYSTEMS+=("$(basename "$sys_dir")")
    done
    echo "Processing all systems (${#SYSTEMS[@]} found)"
fi

if [ ${#SYSTEMS[@]} -eq 0 ]; then
    echo "ERROR: No systems found to process"
    exit 1
fi

echo "Total systems: ${#SYSTEMS[@]}"
echo ""

# Collect matching configs
ALL_CONFIGS=()
for system in "${SYSTEMS[@]}"; do
    local_system_dir="${OUTPUT_DIR}/${system}"
    [ ! -d "$local_system_dir" ] && continue

    for config_dir in "${local_system_dir}"/*; do
        [ ! -d "$config_dir" ] && continue
        config_name=$(basename "$config_dir")
        should_process=false

        if [ ${#RESOLUTIONS[@]} -eq 0 ]; then
            should_process=true
        elif [[ $config_name == *"base"* ]]; then
            should_process=true
        else
            config_res=$(get_resolution "$config_name")
            if [ -n "$config_res" ]; then
                for target_res in "${RESOLUTIONS[@]}"; do
                    config_res_f=$(echo "$config_res" | awk '{printf "%.1f", $1}')
                    target_res_f=$(echo "$target_res" | awk '{printf "%.1f", $1}')
                    [ "$config_res_f" = "$target_res_f" ] && { should_process=true; break; }
                done
            fi
        fi

        if [ "$should_process" = true ] && [ ${#LEARNING_RATES[@]} -gt 0 ] && [[ $config_name != *"base"* ]]; then
            should_process=false
            config_lr=$(get_learning_rate "$config_name")
            if [ -n "$config_lr" ]; then
                for target_lr in "${LEARNING_RATES[@]}"; do
                    config_lr_f=$(echo "$config_lr" | awk '{printf "%.4f", $1}')
                    target_lr_f=$(echo "$target_lr" | awk '{printf "%.4f", $1}')
                    [ "$config_lr_f" = "$target_lr_f" ] && { should_process=true; break; }
                done
            fi
        fi

        [ "$should_process" = true ] && ALL_CONFIGS+=("${system}|${config_name}")
    done
done

total_configs=${#ALL_CONFIGS[@]}
echo "Total configurations to process: ${total_configs}"
echo ""

if [ "$total_configs" -eq 0 ]; then
    echo "ERROR: No configurations found"
    exit 1
fi

# Distribute across job array
configs_per_task=$((total_configs / TOTAL_TASKS))
remainder=$((total_configs % TOTAL_TASKS))

if [ $TASK_ID -lt $remainder ]; then
    start_idx=$((TASK_ID * (configs_per_task + 1)))
    end_idx=$((start_idx + configs_per_task + 1))
else
    start_idx=$((remainder * (configs_per_task + 1) + (TASK_ID - remainder) * configs_per_task))
    end_idx=$((start_idx + configs_per_task))
fi

echo "Task ${TASK_ID} processing configs ${start_idx} to $((end_idx - 1))"
echo ""

processed=0
for ((i=start_idx; i<end_idx && i<total_configs; i++)); do
    config_entry="${ALL_CONFIGS[$i]}"
    system="${config_entry%%|*}"
    config_name="${config_entry##*|}"
    echo "[$((i+1))/${total_configs}] System: ${system}, Config: ${config_name}"
    process_config "$system" "$config_name"
    processed=$((processed + 1))
done

echo ""
echo "========================================"
echo "Task ${TASK_ID}: processed ${processed} configs"
echo "========================================"
