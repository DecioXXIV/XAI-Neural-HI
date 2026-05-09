import torch
import random
from abc import ABC, abstractmethod
from typing import List, Tuple
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Sampler

from src.utils.data.dataset_utils import get_train_transforms
from src.utils.data.custom_transforms import Invert, AddGaussianNoise

# ====================================== #
# SUPPORT FOR DETERMINISTIC AUGMENTATION #
# ====================================== #
class PrecomputedOrderSampler(Sampler):
    """Sampler that yields a fixed, pre-computed sequence of dataset indices."""

    def __init__(self, indices: torch.Tensor): self.indices = indices.tolist()

    def update_indices(self, new_indices: torch.Tensor): self.indices = new_indices.tolist()

    def __iter__(self): return iter(self.indices)

    def __len__(self) -> int: return len(self.indices)

class DeterministicAugmentedDataset(ImageFolder):
    """ImageFolder variant that seeds each random transform individually.

    The seed used before transform_idx for sample at position (batch_idx, pos_in_batch)
    in epoch e of run with random_seed i is:
        int(str(i) + str(e) + str(batch_idx) + str(pos_in_batch) + str(transform_idx))
    where batch_idx = batch index, pos_in_batch = position of the sample within the batch.
    """

    RANDOM_TRANSFORM_INDICES: set = {1, 2, 3, 4, 5, 7, 8, 10}

    def __init__(self, root: str, transforms_list: list, classes_list: List[str]):
        super().__init__(root=root, transform=None)
        self.samples = sorted(self.samples, key=lambda x: x[0])
        self.imgs = self.samples
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(classes_list))}

        self.transforms_list = transforms_list
        # shared_batch_positions[dataset_idx] = [batch_idx, pos_in_batch] — lives in OS shared memory so all DataLoader worker processes (spawned later) see in-place updates.
        self.shared_batch_positions = torch.zeros(len(self.samples), 2, dtype=torch.long).share_memory_()
        # shared_meta[0] = random_seed, shared_meta[1] = epoch — same reasoning.
        self.shared_meta = torch.zeros(2, dtype=torch.long).share_memory_()

    def set_epoch_context(self, indices: torch.Tensor, batch_size: int, random_seed: int, epoch: int):
        """Fill shared tensors in-place. Must be called while workers are idle (between epochs)."""
        self.shared_meta[0] = random_seed
        self.shared_meta[1] = epoch
        n_samples = len(indices)
        self.shared_batch_positions[indices, 0] = torch.arange(n_samples, dtype=torch.long) // batch_size
        self.shared_batch_positions[indices, 1] = torch.arange(n_samples, dtype=torch.long) % batch_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        img = self.loader(path)

        batch_idx = self.shared_batch_positions[index, 0].item()
        pos_in_batch = self.shared_batch_positions[index, 1].item()
        random_seed = self.shared_meta[0].item()
        epoch = self.shared_meta[1].item()

        for transform_idx, transform in enumerate(self.transforms_list):
            if transform_idx in self.RANDOM_TRANSFORM_INDICES:
                seed = int(str(random_seed) + str(epoch) + str(batch_idx) + str(pos_in_batch) + str(transform_idx))
                random.seed(seed)
                torch.manual_seed(seed)
            img = transform(img)

        return img, label

# =========== #
# DATALOADERS #
# =========== #
class BaseDataLoader(ABC):
    def __init__(self, directory: str, classes: List[str], batch_size: int, model_input_size: int, mean_: List[float], std_: List[float], device: str):
        self.directory = directory
        self.classes = classes
        self.batch_size = batch_size
        self.model_input_size = model_input_size
        self.mean = mean_
        self.std = std_
        self.device = device
        
        self.num_workers = 4
        self.pin_memory = self.device == "cuda"
        self.persistent_workers = self.num_workers > 0
        self.prefetch_factor = 2
    
    def generate_dataset(self) -> ImageFolder:
        dataset = ImageFolder(root=self.directory, transform=self.compose_transform())
        dataset.samples = sorted(dataset.samples, key=lambda x: x[0])
        dataset.imgs = dataset.samples
        
        class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.classes))}
        dataset.class_to_idx = class_to_idx
        return dataset
    
    @abstractmethod
    def load_data(self) -> Tuple[ImageFolder, DataLoader]: pass
    
    @abstractmethod
    def compose_transform(self) -> T.Compose: pass
    
class TestDataLoader(BaseDataLoader):
    def __init__(self, directory: str, classes: List[str], batch_size: int, model_input_size: int, mean_: List[float], std_: List[float], device: str):
        super().__init__(directory, classes, batch_size, model_input_size, mean_, std_, device)
    
    def load_data(self) -> Tuple[ImageFolder, DataLoader]:
        dataset = self.generate_dataset()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, 
                            pin_memory=self.pin_memory, persistent_workers=self.persistent_workers, prefetch_factor=self.prefetch_factor)
        
        return dataset, loader
    
    def compose_transform(self) -> T.Compose:
        return T.Compose([
            T.Resize((self.model_input_size, self.model_input_size)),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])

class TrainDataLoader(BaseDataLoader):
    """TrainDataLoader with fully deterministic, per-sample augmentation.
    - no WeightedRandomSampler; epoch order seeded with int(str(random_seed) + str(epoch))
    - every random transform seeded with int(str(random_seed)+str(epoch)+str(batch_idx)+str(pos_in_batch)+str(transform_idx))
    """

    def __init__(self, directory: str, classes: List[str], batch_size: int,
                 model_input_size: int, mean_: List[float], std_: List[float],
                 device: str, random_seed: int, train_transforms: str, epoch: int = 1):
        super().__init__(directory, classes, batch_size, model_input_size, mean_, std_, device)
        self.random_seed = random_seed
        self.train_transforms = train_transforms
        self.epoch = epoch
        self._dataset: DeterministicAugmentedDataset | None = None
        self._sampler: PrecomputedOrderSampler | None = None
        self._loader: DataLoader | None = None

    def _get_transforms_as_list(self) -> list:
        mean_int = tuple(int(m * 255) for m in self.mean)
        transforms_dict = get_train_transforms(self.train_transforms)
        
        cjitter        = transforms_dict.get("cjitter")
        cjitter_p      = transforms_dict.get("cjitter_p")
        randaffine     = transforms_dict.get("randaffine")
        randaffine_p   = transforms_dict.get("randaffine_p")
        randpersp      = transforms_dict.get("randpersp")
        gray_p         = transforms_dict.get("gray_p")
        gaussian_blur  = transforms_dict.get("gaussian_blur")
        blur_p         = transforms_dict.get("blur_p")
        rand_eras      = transforms_dict.get("rand_eras")
        invert_p       = transforms_dict.get("invert_p")
        gaussian_noise = transforms_dict.get("gaussian_noise")
        gn_p           = transforms_dict.get("gn_p")
        
        return [
            T.Resize((self.model_input_size, self.model_input_size)),                                                                   # 0 - deterministic
            T.RandomApply([T.ColorJitter(**cjitter)], p=cjitter_p),                                                                     # 1 - random
            T.RandomApply([T.RandomAffine(**randaffine, interpolation=v2.InterpolationMode.BILINEAR, fill=mean_int)], p=randaffine_p),  # 2 - random
            T.RandomPerspective(**randpersp, interpolation=v2.InterpolationMode.BILINEAR, fill=mean_int),                               # 3 - random
            T.RandomApply([T.GaussianBlur(**gaussian_blur)], p=blur_p),                                                                 # 4 - random
            T.RandomGrayscale(p=gray_p),                                                                                                # 5 - random
            T.ToTensor(),                                                                                                               # 6 - deterministic
            T.RandomErasing(**rand_eras, value=self.mean),                                                                              # 7 - random
            T.RandomApply([Invert()], p=invert_p),                                                                                      # 8 - random
            T.Normalize(mean=self.mean, std=self.std),                                                                                  # 9 - deterministic
            T.RandomApply([AddGaussianNoise(**gaussian_noise)], p=gn_p)                                                                 # 10 - random
        ]

    def compose_transform(self) -> T.Compose: return T.Compose(self._get_transforms_as_list())

    def _generate_epoch_indices(self, n_samples: int) -> torch.Tensor:
        gen = torch.Generator()
        gen.manual_seed(int(str(self.random_seed) + str(self.epoch)))
        return torch.randperm(n_samples, generator=gen)

    def load_data(self) -> Tuple['DeterministicAugmentedDataset', DataLoader]:
        transforms_list = self._get_transforms_as_list()
        self._dataset = DeterministicAugmentedDataset(self.directory, transforms_list, self.classes)
        indices = self._generate_epoch_indices(len(self._dataset))
        self._dataset.set_epoch_context(indices, self.batch_size, self.random_seed, self.epoch)
        self._sampler = PrecomputedOrderSampler(indices)
        
        self._loader = DataLoader(
            self._dataset, batch_size=self.batch_size, sampler=self._sampler,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers, prefetch_factor=self.prefetch_factor
        )
        
        return self._dataset, self._loader

    def update_epoch(self, epoch: int):
        """Update shared tensors and sampler in-place for the next epoch. Workers see changes immediately."""
        assert self._dataset is not None and self._sampler is not None, "Call load_data() before update_epoch()."
        self.epoch = epoch
        indices = self._generate_epoch_indices(len(self._dataset))
        self._dataset.set_epoch_context(indices, self.batch_size, self.random_seed, epoch)
        self._sampler.update_indices(indices)