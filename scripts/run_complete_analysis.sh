#!/bin/bash
# run_complete_analysis.sh
# Downloads GNoME data and runs all XAI analysis scripts

# Exit on error
set -e

# Configuration variables
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
MODELS_DIR="${PROJECT_ROOT}/models"
BASE_OUTPUT_DIR="${PROJECT_ROOT}/experiments/results"

# Create a timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

# Create a log file
LOG_FILE="${OUTPUT_DIR}/analysis_log.txt"
touch "${LOG_FILE}"

echo "Starting GNoME data download and XAI analysis at $(date)" | tee -a "${LOG_FILE}"

# ===== STEP 0: INSTALL OR UPDATE DEPENDENCIES =====
echo "Step 0: Installing or updating dependencies..." | tee -a "${LOG_FILE}"
pip install --upgrade certifi>=2024.2.2 --break-system-packages 2>&1 | tee -a "${LOG_FILE}"
pip install -r "${PROJECT_ROOT}/requirements.txt" --break-system-packages 2>&1 | tee -a "${LOG_FILE}"
pip install -e . --break-system-packages
echo "Dependencies installation complete." | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"

# Create required directories
mkdir -p "${DATA_DIR}" "${MODELS_DIR}" "${BASE_OUTPUT_DIR}"
mkdir -p "${MODELS_DIR}/gnome_model" "${MODELS_DIR}/mlip_model"

# Create output subdirectories
ANALYSIS_DIR="${OUTPUT_DIR}/representation_analysis"
COUNTERFACTUAL_DIR="${OUTPUT_DIR}/counterfactuals"
REFINEMENT_DIR="${OUTPUT_DIR}/candidate_refinement"
BENCHMARK_DIR="${OUTPUT_DIR}/benchmark"

mkdir -p "${ANALYSIS_DIR}" "${COUNTERFACTUAL_DIR}" "${REFINEMENT_DIR}" "${BENCHMARK_DIR}"

# ===== STEP 1: DOWNLOAD GNOME DATA =====
echo "Step 1: Downloading GNoME dataset using wget..." | tee -a "${LOG_FILE}"
python3 "${PROJECT_ROOT}/scripts/download_data_wget.py" --data_dir="${DATA_DIR}" 2>&1 | tee -a "${LOG_FILE}"
echo "GNoME data download complete." | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"

# ===== STEP 2: CONVERT DATA FORMAT =====
echo "Step 2: Converting GNoME data to crystal_data.pkl format..." | tee -a "${LOG_FILE}"
python3 "${PROJECT_ROOT}/scripts/prepare_crystal_data.py" \
  --input_dir="${DATA_DIR}/gnome_data" \
  --output_file="${DATA_DIR}/crystal_data.pkl" \
  --max_structures=50 \
  2>&1 | tee -a "${LOG_FILE}"
echo "Data conversion complete." | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"

# ===== STEP 3: PRETRAIN GNoME MODEL =====
echo "Step 3: Pretraining GNoME model..." | tee -a "${LOG_FILE}"
python3 "${PROJECT_ROOT}/scripts/pretrain_model.py" \
  --data_file="${DATA_DIR}/crystal_data.pkl" \
  --output_dir="${MODELS_DIR}/gnome_model" \
  --epochs=5 \
  2>&1 | tee -a "${LOG_FILE}"
echo "Model pretraining complete." | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"

# Define data file and model paths for XAI scripts
CRYSTAL_DATA="${DATA_DIR}/crystal_data.pkl"
GNOME_MODEL="${MODELS_DIR}/gnome_model"
MLIP_MODEL="${MODELS_DIR}/mlip_model"
NUM_EXAMPLES=5

# ===== STEP 4: ANALYZE REPRESENTATIONS =====
echo "Step 4: Analyzing learned representations" | tee -a "${LOG_FILE}"
python3 "${PROJECT_ROOT}/scripts/analyze_representations.py" \
  --model_dirs "${GNOME_MODEL}" \
  --data_file "${CRYSTAL_DATA}" \
  --output_dir "${ANALYSIS_DIR}" \
  --xai_method "integrated_gradients" \
  --track_evolution \
  --test_concepts \
  --search_novel \
  --analyze_examples \
  --num_examples "${NUM_EXAMPLES}" \
  2>&1 | tee -a "${LOG_FILE}"
echo "Representation analysis complete." | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"

# ===== STEP 5: GENERATE COUNTERFACTUALS =====
echo "Step 5: Generating counterfactual explanations" | tee -a "${LOG_FILE}"
python3 "${PROJECT_ROOT}/scripts/generate_counterfactuals.py" \
  --model_dir "${GNOME_MODEL}" \
  --data_file "${CRYSTAL_DATA}" \
  --output_dir "${COUNTERFACTUAL_DIR}" \
  --method "gradient" \
  --transformation "substitution" \
  --learning_rate 0.01 \
  --max_iterations 50 \
  --num_examples "${NUM_EXAMPLES}" \
  --mlip_dir "${MLIP_MODEL}" \
  2>&1 | tee -a "${LOG_FILE}"
echo "Counterfactual generation complete." | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"

# ===== STEP 6: REFINE CANDIDATES =====
echo "Step 6: Refining crystal structure prediction" | tee -a "${LOG_FILE}"
python3 "${PROJECT_ROOT}/scripts/refine_candidates.py" \
  --model_dir "${GNOME_MODEL}" \
  --data_file "${CRYSTAL_DATA}" \
  --output_dir "${REFINEMENT_DIR}" \
  --method "saps" \
  --xai_method "integrated_gradients" \
  --num_examples "${NUM_EXAMPLES}" \
  --num_candidates 20 \
  2>&1 | tee -a "${LOG_FILE}"
echo "Candidate refinement complete." | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"

# ===== STEP 7: BENCHMARK XAI METHODS =====
echo "Step 7: Benchmarking XAI methods" | tee -a "${LOG_FILE}"
python3 "${PROJECT_ROOT}/scripts/benchmark_xai.py" \
  --model_dir "${GNOME_MODEL}" \
  --data_file "${CRYSTAL_DATA}" \
  --output_dir "${BENCHMARK_DIR}" \
  --methods "gnnexplainer" "integrated_gradients" \
  --num_examples 3 \
  --compare_example 0 \
  2>&1 | tee -a "${LOG_FILE}"
echo "XAI benchmarking complete." | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"

echo "All steps completed successfully at $(date)" | tee -a "${LOG_FILE}"
echo "Results saved to ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"

# Generate a summary report
echo "Generating summary report..." | tee -a "${LOG_FILE}"

cat > "${OUTPUT_DIR}/summary_report.md" << EOF
# XAI Analysis Summary Report

**Date:** $(date)
**Model:** ${GNOME_MODEL}
**Data:** ${CRYSTAL_DATA}

## Steps Performed
1. Downloaded GNoME dataset using wget
2. Converted data to crystal_data.pkl format
3. Pretrained GNoME model
4. Analyzed learned representations
5. Generated counterfactual explanations
6. Refined crystal structure prediction
7. Benchmarked XAI methods

## Results Overview

### 1. Representation Analysis
- Location: ${ANALYSIS_DIR}
- Analysis of learned representations across active learning rounds
- Chemical concept testing results
- Novel stability drivers identified

### 2. Counterfactual Explanations
- Location: ${COUNTERFACTUAL_DIR}
- Gradient-based substitution counterfactuals
- DFT feasibility evaluations

### 3. Crystal Structure Prediction Refinement
- Location: ${REFINEMENT_DIR}
- XAI-guided SAPS results

### 4. XAI Method Benchmarking
- Location: ${BENCHMARK_DIR}
- Comparison of GNNExplainer and Integrated Gradients
- Performance metrics and best practices

## Next Steps
- Review visualization results in each directory
- Validate identified patterns with experimental data
- Apply insights to guide new material synthesis
EOF

echo "Analysis complete! See ${OUTPUT_DIR}/summary_report.md for a summary of results." | tee -a "${LOG_FILE}"