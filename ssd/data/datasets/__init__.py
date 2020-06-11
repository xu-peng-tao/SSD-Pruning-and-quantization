from torch.utils.data import ConcatDataset

from ssd.config.path_catlog import DatasetCatalog
from .voc import VOCDataset
from .coco import COCODataset
from .txt_dataset import TxtDataset
_DATASETS = {
    'VOCDataset': VOCDataset,
    'COCODataset': COCODataset,
}


def build_dataset(dataset_list, transform=None, target_transform=None, is_train=True):
    assert len(dataset_list) > 0
    datasets = []
    for dataset_name in dataset_list:
        if dataset_name in DatasetCatalog.DATASETS:
            data = DatasetCatalog.get(dataset_name)
            args = data['args']
            factory = _DATASETS[data['factory']]
            args['transform'] = transform
            args['target_transform'] = target_transform
            if factory == VOCDataset:
                args['keep_difficult'] = not is_train
            elif factory == COCODataset:
                args['remove_empty'] = is_train
            dataset = factory(**args)
        else:
            if is_train is True:
                dataset_type = 'train'
            else:
                dataset_type = 'valid'
            dataset = TxtDataset( transform=transform, target_transform=target_transform,
                                  dataset_type=dataset_type,dataset_name=dataset_name)
        datasets.append(dataset)
    # for testing, return a list of datasets
    if not is_train:
        return datasets
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return [dataset]
