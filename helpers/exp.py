from itertools import product
from queue import Queue


def create_tasks(model:list,baselines:list,attackers:list) -> list:
    """
    Create tasks for LLM processing.
    """
    task = []
    for model, baseline, attacker in product(model, baselines, attackers):
        task.append({
            "model": model,
            "baseline": baseline,
            "attacker": attacker
        })
    return task