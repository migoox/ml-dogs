import os

from utils.daps_explorer import DapsExplorer, DataSetType
from utils.dataset_creator import DatasetCreator, DataSetType, SpecgramsSilentFilter

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

def create_train_data(parent_path: str = "datasets", dataset_path: str = "dataset", filters: list = [ SpecgramsSilentFilter() ]):
    [group0, group1] = get_data_groups(DataSetType.Training)
    dc = DatasetCreator(
        group0, group1,
        parent_path=os.path.join(dir_path, "..", parent_path),
        dataset_type=DataSetType.Training,
        specgram_filters=filters
    )
    dc.export_dataset(dataset_path)

def create_test_data(parent_path: str = "datasets", dataset_path: str = "dataset", filters: list = [  SpecgramsSilentFilter() ]):
    [group0, group1] = get_data_groups(DataSetType.Test)
    dc = DatasetCreator(
        group0, group1,
        parent_path=os.path.join(dir_path, "..", parent_path),
        dataset_type=DataSetType.Test,
        specgram_filters=filters
    )
    dc.export_dataset(dataset_path)

def create_validation_data(parent_path: str = "datasets", dataset_path: str = "dataset", filters: list = [  SpecgramsSilentFilter() ]):
    [group0, group1] = get_data_groups(DataSetType.Validation)
    dc = DatasetCreator(
        group0, group1,
        parent_path=os.path.join(dir_path, "..", parent_path),
        dataset_type=DataSetType.Validation,
        specgram_filters=filters
    )
    dc.export_dataset(dataset_path)

def create_datasets(parent_path: str = "datasets", dataset_path: str = "dataset", filters: list = [ SpecgramsSilentFilter() ]):
    create_train_data(parent_path, dataset_path, filters)
    create_validation_data(parent_path, dataset_path, filters)
    create_test_data(parent_path, dataset_path, filters)