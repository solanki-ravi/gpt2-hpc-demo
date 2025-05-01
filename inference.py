import torch
import deepspeed
import argparse
import os # Import os module
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

def main():
    parser = argparse.ArgumentParser(description="Inference script for DeepSpeed trained GPT-2 model.")
    parser.add_argument("checkpoint_dir", type=str, help="Directory containing the DeepSpeed checkpoint")
    parser.add_argument("--prompt", type=str, default="DeepSpeed is", help="Prompt for text generation")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Base model name for tokenizer")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run inference on (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate")
    args = parser.parse_args()

    # --- Model Configuration (must match training) ---
    # Known issue: tokenizer.vocab_size might need adjustment depending on how it was saved/loaded.
    # Using standard gpt2 size (50257) and pad_token_id (50256) for now.
    config = GPT2Config(
        vocab_size=50257, # Standard GPT-2 size
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        pad_token_id=50256 # Standard GPT-2 pad_token_id (same as eos_token_id)
    )

    # --- Load Tokenizer ---
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_name)
    # Ensure pad token is set for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Instantiate Model ---
    print(f"Instantiating base model...")
    # Note: Using low_cpu_mem_usage=True with DeepSpeed ZeRO can sometimes cause issues
    # if parameters are not properly gathered before loading. We load on CPU first.
    model = GPT2LMHeadModel(config)
    print(f"Base model instantiated on CPU.")

    # --- Load Checkpoint ---
    print(f"Loading model weights from checkpoint directory: {args.checkpoint_dir}")
    
    # Construct path to the specific state dict file
    # Adjust filename if necessary based on your checkpoint structure 
    # (e.g., pytorch_model.bin, mp_rank_XX_model_states.pt)
    state_dict_path = os.path.join(args.checkpoint_dir, 'mp_rank_00_model_states.pt') 
    
    if not os.path.exists(state_dict_path):
        # Fallback for non-ZeRO or different saving formats
        state_dict_path_alt = os.path.join(args.checkpoint_dir, 'pytorch_model.bin')
        if os.path.exists(state_dict_path_alt):
            state_dict_path = state_dict_path_alt
        else:
            raise FileNotFoundError(f"Could not find model state dict file at {state_dict_path} or {state_dict_path_alt}")

    print(f"Loading state dict from: {state_dict_path}")
    state_dict = torch.load(state_dict_path, map_location="cpu")

    # Extract the model state dict - DeepSpeed often saves it under 'module'
    if 'module' in state_dict:
        model_state_dict = state_dict['module']
    elif 'model' in state_dict: # Another common key
        model_state_dict = state_dict['model']
    else:
        model_state_dict = state_dict # Assume it's the top-level object if no key matches

    # Load the state dict into the base model
    # set strict=False if there are mismatches (e.g., unexpected keys)
    model.load_state_dict(model_state_dict, strict=False) 
    print(f"Successfully loaded model weights from {state_dict_path}")

    # --- Prepare Model for Inference ---
    model.eval()
    model.to(args.device)
    print(f"Model moved to device: {args.device} and set to evaluation mode.")

    # --- Tokenize Prompt ---
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    print(f"Tokenized prompt: {inputs['input_ids']}")

    # --- Generate Text ---
    print(f"Generating text...")
    with torch.no_grad(): # Disable gradient calculations for inference
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'], # Pass attention mask
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id, # Set pad token id for generation
            eos_token_id=tokenizer.eos_token_id, # Set eos token id
            do_sample=True, # Enable sampling for more varied output
            top_k=50,
            top_p=0.95
        )

    # --- Decode and Print ---
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n--- Prompt ---")
    print(args.prompt)
    print("\n--- Generated Text ---")
    print(generated_text)

if __name__ == "__main__":
    main() 