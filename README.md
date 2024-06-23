
# Llama-2-7b Customer Support Agent Fine-tuning

This repository contains scripts and instructions for fine-tuning the Llama-2-7b model on the Kaludi Customer-Support-Responses dataset. The goal is to create an automated customer support agent capable of generating relevant and coherent responses to customer queries.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project fine-tunes the Llama-2-7b model using the [Kaludi Customer-Support-Responses dataset](https://huggingface.co/datasets/Kaludi/Customer-Support-Responses). The objective is to train a model that can generate appropriate customer support responses.

## Dataset

The dataset used for training consists of customer queries and corresponding support responses. It can be loaded from Hugging Face:

```python
from datasets import load_dataset
dataset = load_dataset("Kaludi/Customer-Support-Responses")
```

## Model

We use the [NousResearch/Llama-2-7b-chat-hf](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf) as the base model and fine-tune it using LoRA (Low-Rank Adaptation) techniques to efficiently adjust the model weights.

## Training

The training script `train.py` performs the following steps:
1. Load the dataset.
2. Format the data for Llama-2.
3. Set up the model and tokenizer.
4. Configure training parameters.
5. Fine-tune the model using the `SFTTrainer` from the `trl` library.

To train the model, run:

```bash
python train.py
```

### Training Parameters

- **LoRA parameters**:
  - `lora_r = 64`: The rank of the LoRA matrices. Higher values may capture more information but require more computation.
  - `lora_alpha = 16`: Scaling factor for the LoRA layers. Adjusts the importance of the LoRA modifications.
  - `lora_dropout = 0.1`: Dropout rate for the LoRA layers. Helps prevent overfitting by randomly dropping units during training.

- **TrainingArguments parameters**:
  - `num_train_epochs = 5`: Number of times the model will iterate over the entire training dataset.
  - `per_device_train_batch_size = 4`: Number of samples processed before updating the model's parameters.
  - `gradient_accumulation_steps = 1`: Number of steps to accumulate gradients before performing a backpropagation pass.
  - `learning_rate = 2e-4`: Step size at each iteration while moving towards a minimum of the loss function.
  - `weight_decay = 0.001`: Regularization technique to reduce overfitting by penalizing large weights.
  - `lr_scheduler_type = "cosine"`: Learning rate schedule. Cosine decay can help the model converge more smoothly.

The model was trained on Google Colab using a T4 GPU. For better performance, more powerful GPUs such as V100 or A100 could be used, and the batch size could be increased. Gradient checkpointing can be disabled on machines with more memory to speed up training.

## Evaluation

Evaluation is performed using `evaluate_LLM.ipynb`, which calculates the perplexity score of the fine-tuned model:

```python
perplexity = calculate_perplexity(model, tokenizer, validation_texts)
print(f'Perplexity: {perplexity}')
```

### Evaluation Metrics

- **Perplexity**: Measures how well the model predicts the next token in a sequence. Lower perplexity indicates better performance.


## Usage

You can interact with the fine-tuned model using `run_agent.py`. This script allows you to input queries and get responses from the model in a loop:

```bash
python run_agent.py
```

or you can ** run the run_agent.ipynb ** file in colab or jupyter environment - tested. 

## Requirements

Install the required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### List of Dependencies

- `accelerate==0.21.0`
- `peft==0.4.0`
- `bitsandbytes==0.40.2`
- `transformers==4.31.0`
- `trl==0.4.7`
- `torch==2.3.0+cu121`

## Results

After fine-tuning, the model achieves a perplexity score of less than 100 on the sample validation set, indicating good performance in generating coherent and relevant responses.

