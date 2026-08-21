"""Shared path constants and JSON read/write helpers used across the pipeline.

DATA_BASE_PATH / OUTPUT_PATH are the default locations for raw platform
exports and generated analysis artifacts, respectively; most modules read
them through `from src.utils.common_io import *` and interpolate them into
their own more specific paths (e.g. f"{OUTPUT_PATH}/<module>/...").
"""

import json
import os

DATA_BASE_PATH = "data/chatgpt"
OUTPUT_PATH = "./outputs"


def load_json(file_path):
    """Load and parse a JSON file, returning None (and logging) on failure."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return None


def to_json(data, file_path, indent=4):
    """
    Save data to a .json or .jsonl file.

    - .json  → writes the whole object
    - .jsonl → writes one JSON object per line
    """

    try:
        ext = os.path.splitext(file_path)[1].lower()
        # Create parent directory if needed
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        if ext == ".json":
            with open(file_path, "w") as file:
                json.dump(data, file, indent=indent, ensure_ascii=False)

        elif ext == ".jsonl":
            with open(file_path, "w") as file:
                # Expect iterable of dicts or dict-like values
                if isinstance(data, dict):
                    iterable = data.values()
                else:
                    iterable = data

                for item in iterable:
                    file.write(json.dumps(item, ensure_ascii=False) + "\n")

        else:
            raise ValueError(f"Unsupported file type: {ext}")

    except Exception as e:
        print(f"Error saving JSON file {file_path}: {e}")
