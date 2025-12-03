import json
import yaml
import random

import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(path: str):
    if not path:
        raise ValueError("A configuration file path must be provided")
    lower = path.lower()
    try:
        with open(path, 'r') as f:
            if lower.endswith('.json'):
                return json.load(f)
            # Otherwise prefer YAML (supports both .yml and .yaml)
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration file {path}: {e}")