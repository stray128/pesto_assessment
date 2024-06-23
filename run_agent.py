import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel



# The model that you want to train from the Hugging Face hub
model_name = "NousResearch/Llama-2-7b-chat-hf"

device_map = {"": 0}

# Fine-tuned model name
finetuned_model = "Llama-2-7b-customer-agent-finetune-5_epoch"

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map = device_map
)
model = PeftModel.from_pretrained(base_model, finetuned_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"



## LLM Interactions
def generate_response(input_text):
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt").to(model.device)

    # Generate the response
    output = model.generate(input_ids=input_ids, max_new_tokens=100)

    # Decode the output and return it
    return tokenizer.decode(output[0], skip_special_tokens=True)



while True:
    # Get user input
    input_text = input("Enter your query: ")
    # Check if user wants to exit
    if input_text.lower() == "exit":
        break
    # Generate and print the response
    response = generate_response(input_text)
    print(f"Chat Agent: {response.split('</s>')[-1].strip()}")
