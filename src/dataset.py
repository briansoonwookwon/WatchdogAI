import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

class PosterDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.train = train  # Store the train parameter
        self.root_dir = root_dir  # Store the root directory
        self.samples = []
        true_dir = os.path.join(root_dir, "true")
        false_dir = os.path.join(root_dir, "false")
        
        # Load true images
        true_images = []
        for img in os.listdir(true_dir):
            if img.endswith(('.jpg', '.png', '.jpeg')):
                true_images.append((os.path.join(true_dir, img), 1))
        
        # Load false images
        false_images = []
        for img in os.listdir(false_dir):
            if img.endswith(('.jpg', '.png', '.jpeg')):
                false_images.append((os.path.join(false_dir, img), 0))
        
        # Balance the dataset by sampling the same number of false images as true images
        if len(false_images) > len(true_images):
            false_images = random.sample(false_images, len(true_images))
        
        self.samples = true_images + false_images
        
        if transform is not None:
            self.transform = transform
        else:
            if train:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])
            else:  # val/test
                self.transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label