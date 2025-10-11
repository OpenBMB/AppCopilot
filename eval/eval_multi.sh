#!/bin/bash

if [ ! -f "run_predict_minicpm.py" ]; then
    echo "Error: run_predict_minicpm.py script not found"
    exit 1
fi

if [ ! -f "run_eval_agent.py" ]; then
    echo "Error: run_eval_agent.py script not found"
    exit 1
fi

# Configure basic parameters
data_name="aitz_test"
model_name="AppCopilot_test"
base_output_dir="./eval_results/AppCopilot/${data_name}/${model_name}"

# List of models to process
models_base_path=(
    "/home/dangyf/AgentCPM-GUI/sft/output/ckpt_rft_4"
)


start_time=$(date +%s)
echo "===== Start model inference and evaluation ====="
echo "Start time: $(date)"

# Iterate over models to run inference and evaluation
for model in "${models_base_path[@]}"; do
    echo "===== Processing model: ${model} ====="
    
    model_root_dir="${base_output_dir}/${model}"
    mkdir -p "${model_root_dir}"
    
    inference_output_dir="${model_root_dir}/inference" 
    mkdir -p "${inference_output_dir}"

    echo "[Step 1/2] Running inference for model ${model}..."
    python run_predict_appcopilot_multi.py \
        --model_path "${model}" \
        --output_dir "${inference_output_dir}" \
        --data_name "${data_name}"
    
    inference_exit_code=$?
    if [ $inference_exit_code -ne 0 ]; then
        echo "Error: Inference for model ${model} failed, exit code: ${inference_exit_code}"
        continue 
    fi
    
    all_jsonl_path="${inference_output_dir}/all.jsonl"
    if [ ! -f "${all_jsonl_path}" ]; then
        echo "Error: Inference result file ${all_jsonl_path} not found"
        continue 
    fi
    
    eval_output_dir="${model_root_dir}/results" 
    mkdir -p "${eval_output_dir}"
    
    echo "[Step 2/2] Running evaluation for model ${model}..."
    python run_eval_agent.py \
        --input_path "${all_jsonl_path}" \
        --output_dir "${eval_output_dir}" \
        --data_name "${data_name}"
    
    eval_exit_code=$?
    if [ $eval_exit_code -ne 0 ]; then
        echo "Error: Evaluation for model ${model} failed, exit code: ${eval_exit_code}"
    else
        echo "Inference and evaluation for model ${model} completed!"
        echo "Result path for model ${model}: ${model_root_dir}"  
    fi

done

# Calculate and print total time and end info
end_time=$(date +%s)
total_time=$((end_time - start_time))
echo "===== Model inference and evaluation completed ====="
echo "End time: $(date)"
echo "Total time: ${total_time} seconds (approximately $(echo "scale=2; $total_time/60" | bc) minutes)"
echo "Root directory for all model results: ${base_output_dir}"