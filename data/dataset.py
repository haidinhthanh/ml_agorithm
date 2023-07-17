from data.common import *


class Dataset:
    dataset_dict = {
        "iris": load_iris_data()
    }

    @classmethod
    def load(cls, dataset_name):
        try:
            return cls.dataset_dict[dataset_name]
        except Exception:
            raise Exception(f"Not support dataset name {dataset_name}")
