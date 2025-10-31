from collections import Counter

import pandas as pd 
import numpy as np
import statistics as stats
from typing import Dict, Any

from preprocess.krippendorff import alpha_per_item



def raw_to_dist(score_list:list)->list:
    counter = Counter(score_list)
    dist = [counter[i]/5 for i in range(6)]
    return dist 


def _dictify_line(line: str) -> Dict[str, Any]:
    """
    Convert a line from the dataset into a dictionary.
    The line is expected to be tab-separated and contain the following fields:
    1. gs_score
    2. n_annotators
    3. origin
    4. score_list (space-separated)
    5. sentence1
    6. sentence2
    """
    fields = line.split('\t') if isinstance(line, str) else line
    score_list = [int(score) for score in fields[3].split(' ')]
    stdev = stats.stdev(score_list)
    mode = stats.multimode(score_list)
    if len(mode) == 1:
        mode = mode[0]
        dev_from_mode = sum(sc - mode for sc in score_list if sc != mode)
    else:
        mode = None
        dev_from_mode = None

    return {
        'gs_score': float(fields[0]),
        'n_annotators': int(fields[1]),
        'origin': fields[2],
        'score_list': score_list,
        'mode': mode,
        'stdev': stdev,
        'devfrommode': dev_from_mode,
        'sentence1': fields[4],
        'sentence2': fields[5].strip("\n")
    }

def split_data_holdout(df: pd.DataFrame, holdout_percent: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to split.
    test_size : float
        The proportion of the dataset to include in the test split (between 0 and 1).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the train and test DataFrames.
    """
    if not 0.1 <= holdout_percent <= 1:
        raise ValueError("Percentage must be between 0 and 1.")
    
    holdout_size = int(len(df) * holdout_percent)

    test = df.sample(n=holdout_size)
    train = df.drop(test.index)
    
    return train, test

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a dataset from a file and convert it into a DataFrame.
    """
    with open(path, 'r', encoding='utf-8') as f:
        dataset = [_dictify_line(line) for line in f.readlines()]

    df = pd.DataFrame(dataset)

    
    df['labels'] = df['score_list'].apply(raw_to_dist)

    return df