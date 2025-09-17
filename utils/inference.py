import queue
import torch
from typing import Literal
import gc

def get_gpu() -> Literal["mps", "cuda", "cpu"]:
    """
    Get the gpu device, if gpu is not available, return cpu.
    Returns:
        str: mps|cuda|cpu
    """
    if torch.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
    
def delete_model(model):
    """
    Delete the model from the gpu memory.
    Args:
        model: The model to delete.
    """
    del model
    torch.cuda.empty_cache()
    gc.collect()

