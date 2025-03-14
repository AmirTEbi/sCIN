#!/bin/bash

run_command() {
    echo "Running: $*"
    "$@" 2>&1 | tee last_command.log
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        echo "Error in command: $*"
        echo "Check 'last_command.log' for details."
        exit 1
    fi
}

if [[ "$(basename "$PWD")" != "sc-cool" ]]; then
    cd sc-cool 
fi

if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment not activated. Activating..."
    
    if [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
    else
        echo "Virtual environment not found! Make sure '.venv' exists."
        exit 1
    fi
fi

# scBridge preprocessing
run_command python scripts/prep_scBridge.py

# Run the scBridge model
run_command python sCIN/models/scBridge/main.py --data_path="data/10x/scBridge" --source_data="rna_bridge.h5ad" --target_data="atac_bridge.h5ad"

# Ask the user to choose one mode
echo "Select one mode for evaluation:"
echo "1) two_sided"
echo "2) atac_to_rna"
echo "3) rna_to_atac"
read -p "Enter the number (1/2/3): " mode_choice

case "$mode_choice" in
    1) extra_arg="--two_sided" ;;
    2) extra_arg="--atac_to_rna" ;;
    3) extra_arg="--rna_to_atac" ;;
    *) echo "Invalid choice! Exiting..."; exit 1 ;;
esac

# Run the eval script
run_command python scripts/eval_scBridge.py --embs_dir data/10x/scBridge --save_dir results/scBridge $extra_arg

echo "Pipeline completed successfully!"