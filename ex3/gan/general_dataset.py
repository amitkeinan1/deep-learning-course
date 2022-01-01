from torch.utils.data import Dataset


class GeneralDataset(Dataset):
    def __init__(self, encoded_images):
        self.samples = encoded_images

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
