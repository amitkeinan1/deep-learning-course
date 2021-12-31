from torch.utils.data import Dataset


class EncodedImagesDataset(Dataset):
    def __init__(self, encoded_images):
        self.encoded_images = encoded_images

    def __len__(self):
        return len(self.encoded_images)

    def __getitem__(self, idx):
        return self.encoded_images[idx]
