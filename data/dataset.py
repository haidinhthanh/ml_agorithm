from data.common import *


class Dataset:
    def __init__(self):
        self.dataset_dict = dict()

    def init(self):
        self.dataset_dict = {
            "iris": load_iris_data()
        }

    def load(self, dataset_name):
        try:
            return self.dataset_dict[dataset_name]
        except Exception:
            raise Exception(f"Not support dataset name {dataset_name}")


