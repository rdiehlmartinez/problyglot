# meta datasets and dataloader  for training 
from .metadataset import MetaDataset
from .metadataloader import MetaDataLoader

# nlu dataloader for evaluation
from .nludataloader import NLUDataLoader

# nlu datasets for evaluation
from .xnlidataset import XNLIDatasetGenerator

NLU_DATASET_GENERATOR_MAPPING = {
    "xnli": XNLIDatasetGenerator
}