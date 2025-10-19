#!/bin/bash
# ============================================================================
# Script 2: Inference and Evaluation
# ============================================================================
# This script runs inference on fine-tuned models and evaluates them.
# It loops over all prompts in prompts.yml and evaluates each corresponding
# fine-tuned model. Can also load models from HuggingFace.
# ============================================================================

set -e  # Stop on any error

# ============================================================================
# CONFIGURATION VARIABLES
# ============================================================================

# Model configuration
BASE_MODEL="unsloth/gemma-3-12b-it"
TEST_DATASET="./kie_splits/test"
MODEL_DIR_BASE="./kie_finetuned"
OUTPUT_DIR="./kie_results"

# Load from HuggingFace (set to true if models are on HF)
LOAD_FROM_HF=false
HF_REPO_BASE=""  # Base repo ID, e.g., "username/kie-gemma3-12b"

# Inference parameters
MAX_NEW_TOKENS=256
TEMPERATURE=1.0  # Use 1.0 for greedy decoding
MIN_P=0.1
SAMPLE_INDICES="0,1,2,3,4,5,6,7,8,9"  # Comma-separated list of sample indices

# Evaluation parameters
MODEL_NAME="gemma-3-12b-it"  # Model name for logging
USE_WANDB=false
WANDB_PROJECT="kie-eval"

# Prompts file
PROMPTS_FILE="prompts.yml"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Function to extract prompts from YAML file
extract_prompts() {
    python3 << 'EOF'
import yaml
import sys

try:
    with open("prompts.yml", "r") as f:
        prompts_data = yaml.safe_load(f)

    prompts = []
    for key, value in prompts_data.items():
        if isinstance(value, dict) and "text" in value:
            prompt_text = value["text"].strip()
            prompts.append((key, prompt_text))

    # Output as JSON for easier parsing in bash
    import json
    print(json.dumps(prompts))
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
EOF
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

echo "============================================================================"
echo "🚀 KIE Inference and Evaluation Script"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  - Base Model: $BASE_MODEL"
echo "  - Test Dataset: $TEST_DATASET"
echo "  - Model Directory Base: $MODEL_DIR_BASE"
echo "  - Output Directory: $OUTPUT_DIR"
echo "  - Load from HuggingFace: $LOAD_FROM_HF"
if [ "$LOAD_FROM_HF" = true ]; then
    echo "  - HuggingFace Base Repo: $HF_REPO_BASE"
fi
echo ""
echo "Inference Parameters:"
echo "  - Max New Tokens: $MAX_NEW_TOKENS"
echo "  - Temperature: $TEMPERATURE (1.0 = greedy decoding)"
echo "  - Min P: $MIN_P"
echo "  - Sample Indices: $SAMPLE_INDICES"
echo ""
echo "Evaluation Parameters:"
echo "  - Model Name: $MODEL_NAME"
echo "  - Use WandB: $USE_WANDB"
if [ "$USE_WANDB" = true ]; then
    echo "  - WandB Project: $WANDB_PROJECT"
fi
echo ""
echo "============================================================================"
echo ""

# Check if test dataset exists
if [ ! -d "$TEST_DATASET" ]; then
    echo "❌ Error: Test dataset not found at $TEST_DATASET"
    echo "Please run './0_prepare_data.sh' first to create the dataset splits."
    exit 1
fi

# Check if prompts file exists
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "❌ Error: Prompts file not found at $PROMPTS_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Extract prompts from YAML
echo "🔄 Extracting prompts from $PROMPTS_FILE..."
PROMPTS_JSON=$(extract_prompts)

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to extract prompts from $PROMPTS_FILE"
    exit 1
fi

# Parse prompts count
PROMPTS_COUNT=$(echo "$PROMPTS_JSON" | python3 -c "import sys, json; data = json.load(sys.stdin); print(len(data))")
echo "✅ Found $PROMPTS_COUNT prompt(s) to evaluate"
echo ""

# Loop over each prompt
echo "============================================================================"
echo "🔄 Starting inference and evaluation for all prompts..."
echo "============================================================================"
echo ""

PROMPT_INDEX=0
echo "$PROMPTS_JSON" | python3 -c "
import sys
import json

data = json.load(sys.stdin)
for i, (key, prompt) in enumerate(data):
    print(f'{i}|||{key}|||{prompt}')
" | while IFS='|||' read -r INDEX KEY PROMPT_TEXT; do
    PROMPT_INDEX=$((INDEX + 1))
    PROMPT_NUM=$(printf "%02d" $PROMPT_INDEX)

    echo ""
    echo "------------------------------------------------------------------------"
    echo "🔄 Evaluating prompt $PROMPT_INDEX/$PROMPTS_COUNT: $KEY"
    echo "------------------------------------------------------------------------"
    echo ""

    # Determine model path
    if [ "$LOAD_FROM_HF" = true ] && [ -n "$HF_REPO_BASE" ]; then
        MODEL_PATH="${HF_REPO_BASE}-prompt${PROMPT_NUM}"
        LOAD_FROM_HF_FLAG="--load-from-hf"
    else
        MODEL_PATH="${MODEL_DIR_BASE}/prompt_${PROMPT_NUM}_${KEY}"
        LOAD_FROM_HF_FLAG=""
    fi

    # Check if model exists (only for local models)
    if [ "$LOAD_FROM_HF" = false ] && [ ! -d "$MODEL_PATH" ]; then
        echo "⚠️  Warning: Model not found at $MODEL_PATH"
        echo "Skipping prompt: $KEY"
        echo ""
        continue
    fi

    INFERENCE_OUTPUT_DIR="${OUTPUT_DIR}/prompt_${PROMPT_NUM}_${KEY}"
    INFERENCE_RESULTS="${INFERENCE_OUTPUT_DIR}/inference_results.json"
    EVAL_EXCEL="${OUTPUT_DIR}/KIE_prompt_${PROMPT_NUM}_${KEY}.xlsx"

    mkdir -p "$INFERENCE_OUTPUT_DIR"

    echo "Prompt: $KEY"
    echo "Model Path: $MODEL_PATH"
    echo "Inference Output: $INFERENCE_OUTPUT_DIR"
    echo "Evaluation Excel: $EVAL_EXCEL"
    echo ""

    # ========================================================================
    # Step 1: Run Inference
    # ========================================================================

    echo "🔄 Running inference..."

    python inference.py \
        --model-path "$MODEL_PATH" \
        $LOAD_FROM_HF_FLAG \
        --base-model "$BASE_MODEL" \
        --test-dataset "$TEST_DATASET" \
        --prompt "$PROMPT_TEXT" \
        --sample-indices "$SAMPLE_INDICES" \
        --max-new-tokens $MAX_NEW_TOKENS \
        --temperature $TEMPERATURE \
        --min-p $MIN_P \
        --output-dir "$INFERENCE_OUTPUT_DIR"

    echo "✅ Inference complete!"
    echo ""

    # ========================================================================
    # Step 2: Run Evaluation
    # ========================================================================

    echo "🔄 Running evaluation..."

    EVAL_CMD="python evaluate.py \
        --inference-results \"$INFERENCE_RESULTS\" \
        --test-dataset \"$TEST_DATASET\" \
        --output-excel \"$EVAL_EXCEL\" \
        --model-name \"$MODEL_NAME\""

    # Add WandB flag if enabled
    if [ "$USE_WANDB" = true ]; then
        EVAL_CMD="$EVAL_CMD --use-wandb --wandb-project \"$WANDB_PROJECT\""
    fi

    eval $EVAL_CMD

    echo "✅ Evaluation complete!"
    echo "   Results saved to: $EVAL_EXCEL"
    echo ""
done

echo ""
echo "============================================================================"
echo "🎉 All inference and evaluation complete!"
echo "============================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Summary of outputs:"
ls -lh "$OUTPUT_DIR"/*.xlsx 2>/dev/null || echo "No Excel files found."
echo ""
echo "You can now review the results in the Excel files."
echo ""
