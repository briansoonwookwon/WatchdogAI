import os
from PIL import Image
from torch.utils.data import Dataset

class PosterDataset(Dataset):
    def __init__(self, root_dir="images", transform=None):
        self.samples = []
        poster_dir = os.path.join(root_dir, "poster")
        for img in os.listdir(poster_dir):
            if img.endswith('.jpg'):
                self.samples.append((os.path.join(poster_dir, img), 1))
        nonposter_dir = os.path.join(root_dir, "nonposter")
        for img in os.listdir(nonposter_dir):
            if img.endswith('.jpg'):
                self.samples.append((os.path.join(nonposter_dir, img), 0))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label