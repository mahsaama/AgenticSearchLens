import json
import os

DATA_BASE_PATH = "data/chatgpt"
OUTPUT_PATH = "./outputs"

def load_json(file_path):
    """Helper function to load JSON files."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return None

def load_jsonl(file_path):
    """Helper function to load JSONL files."""
    data = {}
    try:
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                data[str(i)] = json.loads(line)
        return data
    except Exception as e:
        print(f"Error loading JSONL file {file_path}: {e}")
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


