import os
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

PARENT_DIR = Path(os.getcwd())
IRIS_DATA_PATH = Path(PARENT_DIR, 'data', 'iris', 'iris_csv.csv')


def load_iris_data():
    df = pd.read_csv(IRIS_DATA_PATH)
    return df
