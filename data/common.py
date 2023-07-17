import os
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

PARENT_DIR = Path(os.getcwd()).parent


def load_iris_data():
    IRIS_DATA = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    urlretrieve(IRIS_DATA)
    df = pd.read_csv(IRIS_DATA, sep=',')
    return df


