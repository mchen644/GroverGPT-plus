# GroverGPT+
GroverGPT+: Simulating Grover's Algorithm via Chain-of-Thought Reasoning and Quantum-Native Tokenization

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) [![Arxiv](https://img.shields.io/badge/arXiv-2505.04880-B31B1B.svg)][#arxiv-paper-package]

[#arxiv-paper-package]: https://arxiv.org/abs/2505.04880

## Getting Started

### Step 1: Download the Data

First, download the data from the following link:  
[Download Data](https://huggingface.co/datasets/mchen644/Grover_MMS/resolve/main/data_MMS.rar?download=truep)

Extract the downloaded `data_MMS.rar` to the current directory.

### Step 2: Extend the tokenizer
Extend the tokenizer by following the instructions in the `extend_tokenizer.ipynb`

### Step 3: Generate Dataset for SFT (Supervised Fine-Tuning)
To generate the dataset for CoT Training, use the `dataset_generate_MMS.py` script. This script allows you to customize the dataset by specifying the range of qubits (`n_min` and `n_max`) and the type of input (`Oracle` or `FullCircuit`).

#### Example Commands

1. ​**Generate a dataset with only Oracle definition as input for 2-10 qubits:**
   ```bash
   python dataset_generate_MMS.py --n_min 2 --n_max 10 --input_type Oracle

2. ​**Generate a dataset with FullCircuit definition as input for 2-5 qubits:**
   ```bash
   python dataset_generate_MMS.py --n_min 2 --n_max 7 --input_type FullCircuit

### Step 4: Install LLaMAFactory for Parameter Efficient Supervised Fine-tuning with LoRA
We use the ​**Meta-Llama-3-8B-Instruct** version of the LLaMA model. Before SFT, we need to put all the dataset in the `LLaMA-Factory/data/.`, and add the corresponding info in the dataset_info.json. Below is the training script to reproduce the results:

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /home/cm/model/Meta-Llama-3-8B-Instruct  \
    --dataset Grover_FullCircuit_2_7_MMS,Grover_Oracle_2_10_MMS \
    --dataset_dir /home/cm/LLaMA-Factory/data \
    --template llama3 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir /home/cm/saves/Meta-Llama-3-8B-Instruct/lora/GroverGPT+\
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 3000 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --warmup_steps 20 \
    --eval_steps 50 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 2e-5 \
    --num_train_epochs 10 \
    --val_size 0.1 \
    --plot_loss \
    --fp16 \
    --resize_vocab True \
    --max_samples 10000 \
    --disable_shuffling True \
