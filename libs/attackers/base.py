from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset
from typing import TypedDict, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class DatasetConfig(TypedDict, total=False):
    path: str
    data_files: dict
    split: str

class BaseAttacker(ABC):
    """
    Base class for all attackers.

    Args:
        model: The target model to be attacked.
            - Hugging Face model name or path.
            - If None: Do not need to load a model.
        dataset (Union[str, DatasetConfig, None]): The dataset to store the attack result.
            - If str: The name of a dataset to load from Hugging Face datasets.
            - If DatasetConfig: A dictionary containing parameters for `load_dataset`.
            - If None: No dataset is provided.
    """

    def __init__(self, dataset: Union[str, DatasetConfig, None] = None):
            try:
                if isinstance(dataset, str):
                    self.dataset = load_dataset(dataset,split=dataset.get("split",None))
                elif isinstance(dataset, dict):   
                    self.dataset = load_dataset(dataset["path"], data_files=dataset.get("data_files",None))
                elif dataset is None:
                    self.dataset = Dataset.from_dict({})
                else:
                    raise ValueError("Dataset must be a string (dataset name) or a list (custom dataset).")
            except Exception as e:
                print(f"Error loading dataset: {e}")
                self.dataset = Dataset.from_dict({})
           
    def push_to_hub(self, repo_name):
        """
        Push the dataset to Hugging Face Hub.
        Args:
            repo_name (str): The name of the repository to push to.
        """
        self.dataset.push_to_hub(repo_name)

    @abstractmethod
    def attack(self, *args, **kwargs):
        """
        Perform the attack.
        """
        pass

    def save(self, **kwargs):
        """
        Save the attack results to the dataset.
        Args:
            kwargs: pass keyword arguments to save to the dataset.

        override this method to customize the saving process.
        """
        self.dataset = self.dataset.add_item(dict(**kwargs))
        print(f"Saved {kwargs} to dataset.")