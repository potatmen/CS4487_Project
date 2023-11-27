import torch
from glob import glob
from random import sample
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __vectorize(self, image_path, label):
        #label = label - (label - 0.5) * 0.2
        label = torch.tensor([float(label), 1 - float(label)])
        additional_transforms = [] if self.is_validation else [
            T.RandomHorizontalFlip(), T.RandomRotation(degrees=15)]
        transform = T.Compose(
            [T.Resize(256), T.CenterCrop(224), *additional_transforms, T.ToTensor()])
        with Image.open(image_path) as image:
            tensor_image = transform(image).to(self.device)
        additional_transforms = [] if self.is_validation else [
            T.RandomErasing(scale=(0.1, 0.2))]
        transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            *additional_transforms
        ])
        return transform(tensor_image).detach().cpu(), label

    def __init__(self, is_validation=False, device=None, max_samples=None):
        self.is_validation = is_validation
        self.device = device
        folder = f"{'val' if is_validation else 'train'}"
        fake_imgs = [f'FakeManipulation-{i+1}/**/*.jpg' for i in range(5)]
        fake_paths = [glob(f'{folder}/{img}', recursive=True) for img in fake_imgs]
        real_imgs = [f'Real-{i+1}/**/*.jpg' for i in range(4)]
        real_paths = [glob(f'{folder}/{img}', recursive=True) for img in real_imgs]
        data = []
        for label, paths in enumerate([fake_paths, real_paths]):
            for path in paths:
                for image_path in path:
                    vectorized_data = self.__vectorize(image_path, label)
                    data.append(vectorized_data)        
        self.data = sample(data, k=len(data))[:max_samples]

    def __getitem__(self, index):
        return self.__vectorize(*self.data[index])

    def __len__(self):
        return len(self.data)