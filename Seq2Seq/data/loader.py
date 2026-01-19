from torch.utils.data import DataLoader
from Seq2Seq.data.dataset import EnglishFrenchDataset, CollateFunctor
from Seq2Seq.utils.config import load_config


def create_loader(dataset: EnglishFrenchDataset, batch_size: int = 64):
    loader = DataLoader(
        dataset,
        batch_size,
        shuffle= True,
        collate_fn = CollateFunctor(),
    )
    return loader

