import os

from utils.daps_explorer import DapsExplorer, DataSetType
from utils.dataset_creator import DatasetCreator, DataSetType

dir_path = os.path.dirname(os.path.realpath(__file__))

def get_data_groups(data_set_type: DataSetType):
    group_1_data = []
    group_0_data = []

    set_0 = DapsExplorer.get_data_set(data_set_type, True)
    for recording in set_0:
        try:
            group_0_data.append(recording)
        except Exception as e:
            print(f"Failed to load {recording}: {e}")

    set_1 = DapsExplorer.get_data_set(data_set_type, False)
    for recording in set_1:
        try:
            group_1_data.append(recording)
        except Exception as e:
            print(f"Failed to load {recording}: {e}")

    return group_0_data, group_1_data


def create_train_data():
    [group0, group1] = get_data_groups(DataSetType.Training)

    dc = DatasetCreator(
        group0, group1,
        dataset_path=os.path.join(dir_path, "..", "dataset"),
        dataset_type=DataSetType.Training,
    )
    dc.export_dataset()


def create_test_data():
    [group0, group1] = get_data_groups(DataSetType.Test)
    dc = DatasetCreator(
        group0, group1,
        dataset_path=os.path.join(dir_path, "..", "dataset"),
        dataset_type=DataSetType.Test,
    )
    dc.export_dataset()


