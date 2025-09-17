from typing import Dict, List
from utils.data import read_json_from_file, get_all_files
def split_batches(data: List[Dict], batch_size: int) -> List[List[Dict]]:
    """Split a list of dictionaries into batches of a given size."""
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def merge_batches(batches: List[List[Dict]]) -> List[Dict]:
    """Merge a list of batches into a single list of dictionaries."""
    merged = []
    for batch in batches:
        merged.extend(batch)
    return merged

def create_llm_batch(data: List[Dict], batch_size: int):
    """Create batches for LLM processing."""

    
