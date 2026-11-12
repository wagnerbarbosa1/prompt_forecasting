import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description="Prompt generating for LLM's forecast\n -few (few-shot prompts) \n -zero (zero-shot prompts) \n \\"
        "-similar(similar-shor)) \n -guaratuba (guaratuba dataset parameters) \n -joinville (joinville dataset parameters)"
)

parser.add_argument(
    "few",
)