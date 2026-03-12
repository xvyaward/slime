import os
import torch
from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load_common_state_dict, load_tensors_metadata

def main():
    # Initialize basic parallel state for single GPU
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    
    if parallel_state.is_unitialized():
        parallel_state.initialize_model_parallel(1, 1)

    ckpt_dir = "/root/changhun.lee/models/EXAONE-4.0-1.2B_torch_dist/release"
    
    print(f"Inspecting checkpoint at {ckpt_dir}...")
    
    # Try to load common state dict (metadata like args, iteration)

    try:
        common_sd = load_common_state_dict(ckpt_dir)
        print("Common State Dict keys:", common_sd.keys())
        for k in common_sd.keys():
            if k != 'model':
                print(f"  {k}: {type(common_sd[k])}")
    except Exception as e:
        print(f"Failed to load common state dict: {e}")

    # Try to load tensors metadata (the actual parameter names)
    try:
        tensors_metadata = load_tensors_metadata(ckpt_dir)
        print("\nTensor keys in checkpoint:")
        # keys = sorted(list(tensors_metadata.keys()))
        keys = tensors_metadata.keys()
        for k in keys:
            print(f"  {k}")

    except Exception as e:
        print(f"Failed to load tensors metadata: {e}")

if __name__ == "__main__":
    main()
