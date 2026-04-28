import os
import numpy as np
import pickle as pkl
from typing import List, Tuple
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class TrainRGBMeanStdComputer:
    def __init__(self, experiment_ft_dir: str, dataset: str, classes: List[str]):
        self.experiment_ft_dir = experiment_ft_dir
        self.dataset = dataset
        self.classes = classes
    
    def __call__(self) -> Tuple[List[float], List[float]]:
        path = os.path.join(self.experiment_ft_dir, "rgb_train_stats.pkl")
        try:
            with open(path, "rb") as f:
                mean_, std_ = pkl.load(f)
        except FileNotFoundError:
            mean_, std_ = self._compute_train_rgb_mean_std()
        
        return mean_, std_
    
    def _compute_train_rgb_mean_std(self) -> Tuple[List[float], List[float]]:
        training_crops = []
        for cls in self.classes:
            cls_dir = os.path.join(self.experiment_ft_dir, "train_pre_aug", cls)
            training_crops.extend([os.path.join(cls_dir, fname) for fname in os.listdir(cls_dir) if fname.endswith(".png")])
        
        def process_image(img_path):
            img_array = np.array(Image.open(img_path)).astype(np.float32) / 255.0
            pixel_count = img_array.shape[0] * img_array.shape[1]
            channel_sum = np.sum(img_array, axis=(0, 1))
            channel_squared_sum = np.sum(img_array ** 2, axis=(0, 1))
            return pixel_count, channel_sum, channel_squared_sum

        pixel_num, channel_sum, channel_squared_sum = 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

        # max_workers=None -> min(32, os.cpu_count() + 4) as per Python 3.8+
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_image, training_crops), 
                                desc="Computing train RGB mean and std",
                                total=len(training_crops),
                                position=0, 
                                leave=True, 
                                dynamic_ncols=True))

        for pixel_count, sum_, squared_sum in results:
            pixel_num += pixel_count
            channel_sum = [x + y for x, y in zip(channel_sum, sum_)]
            channel_squared_sum = [x + y for x, y in zip(channel_squared_sum, squared_sum)]

        mean_ = [x / pixel_num for x in channel_sum]
        std_ = [np.sqrt(x / pixel_num - (m ** 2)) for x, m in zip(channel_squared_sum, mean_)]

        savepath = os.path.join(self.experiment_ft_dir, "rgb_train_stats.pkl")
        with open(savepath, "wb") as f:
            pkl.dump((mean_, std_), f)
        
        return mean_, std_