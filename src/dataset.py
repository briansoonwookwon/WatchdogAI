import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class PosterDataset(Dataset):
    def __init__(self, root_dir="images", transform="default"):
        self.samples = []
        poster_dir = os.path.join(root_dir, "poster")
        for img in os.listdir(poster_dir):
            if img.endswith('.jpg'):
                self.samples.append((os.path.join(poster_dir, img), 1))
        nonposter_dir = os.path.join(root_dir, "nonposter")
        for img in os.listdir(nonposter_dir):
            if img.endswith('.jpg'):
                self.samples.append((os.path.join(nonposter_dir, img), 0))

        if transform == "default":
            print("Using default transform")
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label