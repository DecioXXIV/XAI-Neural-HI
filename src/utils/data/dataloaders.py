import torch
from abc import ABC, abstractmethod
from typing import List, Tuple
from torchvision import datasets
from torchvision import transforms as T
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from src.utils.data.custom_transforms import Invert, AddGaussianNoise

class BaseDataLoader(ABC):
    def __init__(self, directory: str, classes: List[str], batch_size: int, model_input_size: int, mean_: List[float], std_: List[float], device: str):
        self.directory = directory
        self.classes = classes
        self.batch_size = batch_size
        self.model_input_size = model_input_size
        self.mean = mean_
        self.std = std_
        self.device = device
    
    def generate_dataset(self) -> datasets.ImageFolder:
        dataset = datasets.ImageFolder(root=self.directory, transform=self.compose_transform())
        dataset.samples = sorted(dataset.samples, key=lambda x: x[0])
        dataset.imgs = dataset.samples
        
        class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.classes))}
        dataset.class_to_idx = class_to_idx
        return dataset
    
    @abstractmethod
    def load_data(self) -> Tuple[datasets.ImageFolder, DataLoader]: pass
    
    @abstractmethod
    def compose_transform(self) -> T.Compose: pass
    
class TrainDataLoader(BaseDataLoader):
    def __init__(self, directory: str, classes: List[str], batch_size: int, model_input_size: int, mean_: List[float], std_: List[float], device: str, weighted_sampling: bool=True, shuffle: bool=True):
        super().__init__(directory, classes, batch_size, model_input_size, mean_, std_, device)
        self.weighted_sampling = weighted_sampling
        self.shuffle = shuffle
    
    def _make_weights_for_balanced_classes(self, images) -> List[float]:
        nclasses = len(self.classes)
        count = [0] * nclasses                                                      
        for item in images: count[item[1]] += 1
        
        weight_per_class = [0.] * nclasses                                      
        N = float(sum(count))   
                                                        
        for i in range(nclasses): weight_per_class[i] = N/float(count[i])                                 
        weight = [0] * len(images)     
                                                 
        for idx, val in enumerate(images): weight[idx] = weight_per_class[val[1]]
        return weight

    def load_data(self) -> Tuple[datasets.ImageFolder, DataLoader]:
        dataset = self.generate_dataset()
        
        num_workers = 4
        pin_memory = self.device == "cuda"
        persistent_workers = num_workers > 0
        prefetch_factor = 2
        
        if self.weighted_sampling:
            weights = torch.DoubleTensor(self._make_weights_for_balanced_classes(dataset.samples))
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)
        else: loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)
        
        return dataset, loader
    
    def compose_transform(self) -> T.Compose:
        mean_int = tuple(int(m * 255) for m in self.mean)
        cjitter = {'brightness': [0.4, 1.3], 'contrast': 0.6, 'saturation': 0.6, 'hue': (-0.4, 0.4)}
        randaffine = {'degrees': [-10,10], 'translate': [0.2, 0.2], 'scale': [1.3, 1.4], 'shear': 1, 'interpolation': v2.InterpolationMode.BILINEAR, 'fill': mean_int}
        randpersp = {'distortion_scale': 0.1, 'p': 0.2, 'interpolation': v2.InterpolationMode.BILINEAR, 'fill': mean_int}
        gray_p = 0.2
        gaussian_blur = {'kernel_size': 3, 'sigma': [0.1, 0.5]}
        rand_eras = {'p': 0.5, 'scale': [0.02, 0.33], 'ratio': [0.3, 3.3], 'value': self.mean}
        invert_p = 0.05
        gaussian_noise = {'mean': 0., 'std': 0.004}
        gn_p = 0.0
        
        return T.Compose([
            T.Resize((self.model_input_size, self.model_input_size)),
            T.ColorJitter(**cjitter),
            T.RandomAffine(**randaffine),
            T.RandomPerspective(**randpersp),
            T.GaussianBlur(**gaussian_blur),
            T.RandomGrayscale(p=gray_p),
            T.ToTensor(),
            T.RandomErasing(**rand_eras),
            T.RandomApply([Invert()], p=invert_p),
            T.Normalize(mean=self.mean, std=self.std),
            T.RandomApply([AddGaussianNoise(**gaussian_noise)], p=gn_p)
        ])

class TestDataLoader(BaseDataLoader):
    def __init__(self, directory: str, classes: List[str], batch_size: int, model_input_size: int, mean_: List[float], std_: List[float], device: str):
        super().__init__(directory, classes, batch_size, model_input_size, mean_, std_, device)
    
    def load_data(self) -> Tuple[datasets.ImageFolder, DataLoader]:
        dataset = self.generate_dataset()
        
        num_workers = 4
        pin_memory = self.device == "cuda"
        persistent_workers = num_workers > 0
        prefetch_factor = 2
        
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)
        
        return dataset, loader
    
    def compose_transform(self) -> T.Compose:
        return T.Compose([
            T.Resize((self.model_input_size, self.model_input_size)),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])