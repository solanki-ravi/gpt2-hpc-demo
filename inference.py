import torch
import deepspeed
import argparse
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
import os # Import os for environment variables
import torch.distributed as dist # Import torch.distributed
# import torch.optim as optim # Remove this import
from deepspeed.ops.adam import DeepSpeedCPUAdam # Import DeepSpeedCPUAdam

def main():
    parser = argparse.ArgumentParser(description="Inference script for DeepSpeed trained GPT-2 model.")
    parser.add_argument("checkpoint_dir", type=str, help="Directory containing the DeepSpeed checkpoint")
    parser.add_argument("--prompt", type=str, default="DeepSpeed is", help="Prompt for text generation")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Base model name for tokenizer")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run inference on (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate")
    args = parser.parse_args()

    # --- Manually Initialize Distributed Backend (for single process) ---
    # DeepSpeed requires this even for single process if using ZeRO features,
    # and doing it manually prevents the mpi4py auto-detect issue.
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9994' # Default port, modify if needed
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_RANK'] = '0' # Add LOCAL_RANK for DeepSpeed
    dist.init_process_group(backend='gloo')
    print("Manually initialized torch.distributed (rank 0, world 1, local_rank 0)")
    # ---------------------------------------------------------------

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
    model = GPT2LMHeadModel(config)
    print(f"Base model instantiated on CPU.")

    # --- Create Optimizer Instance (required by ZeRO Stage 2 init) ---
    # Use DeepSpeedCPUAdam as expected by the config with offloading
    optimizer = DeepSpeedCPUAdam(model.parameters(), lr=1e-5)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-5) # Use same LR as config just in case
    print("Created DeepSpeedCPUAdam optimizer for DeepSpeed initialization.")

    # --- Initialize DeepSpeed Engine ---
    # Pass the DeepSpeedCPUAdam optimizer 
    print("Initializing DeepSpeed engine for loading...")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer, # Pass DeepSpeedCPUAdam
        config="deepspeed_config.json" # Provide path to config used during training
    )
    print("DeepSpeed engine initialized.")

    # --- Load Checkpoint using the Engine ---
    # Checkpoint dir is the base (e.g., ./my_gpt2_checkpoint)
    # The tag is the specific sub-directory (e.g., global_step65110)
    checkpoint_tag = args.checkpoint_dir.split('/')[-1] # Assumes last part is tag
    base_checkpoint_dir = "/".join(args.checkpoint_dir.split('/')[:-1]) # Get base dir
    if not base_checkpoint_dir:
        base_checkpoint_dir = "." # Handle case where only tag is given

    print(f"Loading model weights from checkpoint tag: {checkpoint_tag} in dir: {base_checkpoint_dir}")
    load_path = model_engine.load_checkpoint(
        base_checkpoint_dir,
        tag=checkpoint_tag,
        load_optimizer_states=False, # Don't load optimizer state for inference
        load_lr_scheduler_states=False # Don't load scheduler state for inference
    )

    if load_path is None:
        raise ValueError(f"Failed to load checkpoint {checkpoint_tag} from {base_checkpoint_dir}")

    print(f"Successfully loaded model weights from {load_path}")

    # --- Prepare Model for Inference ---
    # Use the model extracted from the DeepSpeed engine
    model_to_use = model_engine.module
    model_to_use.eval()
    model_to_use.to(args.device)
    print(f"Model moved to device: {args.device} and set to evaluation mode.")

    # --- Tokenize Prompt ---
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    print(f"Tokenized prompt: {inputs['input_ids']}")

    # --- Generate Text ---
    print(f"Generating text...")
    with torch.no_grad(): # Disable gradient calculations for inference
        outputs = model_to_use.generate(
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