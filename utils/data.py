import csv
import json
import os
from typing import Dict, List
import uuid
import datasets

def get_all_files(dir_path: str, file_extension: str="json") -> List[str]:
    """
    Get all files in a directory with a specific file extension.
    
    Args:
        dir_path (str): The path to the directory.
        file_extension (str): The file extension to search for.
        
    Returns:
        List[str]: A list of file paths.
    """
    files = []
    for root, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith(file_extension):
                files.append(os.path.join(root, filename))
    return files

def load_csv_to_list(file_path):
    """
    Load a CSV file and convert it to a list of lists.
    """   
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data = [list(row.values()) for row in reader]
    return data

def save_list_to_csv(data, file_path):
    """
    Save a list of lists to a CSV file.

    Args:
        data (list of lists): The data to save.
            the first row is the header.
            [["column1", "column2"], ["value1", "value2"],["value3", "value4"],...]
        file_path (str): The path to the CSV file.
    """
    dir = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(dir):
        print(f"Directory {dir} does not exist. Creating it.")
        os.makedirs(os.path.dirname(file_path))
    if not data:
        return
    with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data[0])
        for row in data[1:]:
            writer.writerow(row)
    print(f"Data saved to {file_path}")

def write_json_to_file(file_path, journal):
    """
    Write a json object to a json file.
    """
    dir = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(dir):
        print(f"Directory {dir} does not exist. Creating it.")
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
    with open(file_path, mode='w', newline='', encoding='utf-8') as journal_file:
       journal = json.dumps(journal, indent=4)
       journal_file.write(journal)

def read_json_from_file(file_path):
    """
    Read a json obj from a json file.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return None
    with open(file_path, mode='r', encoding='utf-8') as journal_file:
        journal = json.load(journal_file)
    return journal

def get_dataset(name, split, type=None, **target_column:dict):
    """
    Get the dataset either from huggingface or local.
    """
    try:
        if isinstance(name, str):
            if os.path.exists(name) and type is None:
                dataset = datasets.load_dataset(name,split=split)
            elif type is not None:
                dataset = datasets.load_dataset(type, data_files=name, split=split)
                
        elif isinstance(name, dict):
            dataset = datasets.load_dataset(name["repo_name"], name["split"])
        else:
            raise ValueError("name must be a string or a dictionary.")
        if target_column:
            for k,v in target_column.items():
                dataset = dataset.filter(lambda x: x[k] == v)
        return dataset
    except Exception as e:
        print(f"Error loading dataset {name}: {e}")
        return None
    
def parse_unknown_args(unknown_args):
    unknown_args_dict = {}
    for i in range(0, len(unknown_args), 2):
        key = unknown_args[i].lstrip("--")
        value = unknown_args[i + 1]
        unknown_args_dict[key] = value
    return unknown_args_dict

def add_ids(tasks:List[Dict],id_names:str="id")-> List[Dict]:
    """
    Add ids to the task.
    """
    return [{id_names: str(uuid.uuid4()), **task} for task in tasks]

def filter_by_key(data: List[Dict], key: str, value: str) -> List[Dict]:
    """Filter a list of dictionaries by a given key and value."""
    return [item for item in data if item.get(key) == value]