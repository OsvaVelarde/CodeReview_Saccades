from torch import load, stack, cat, float32
from torch.utils.data import Dataset

import os
import pandas as pd

class SaccadesParraLab(Dataset):
    def __init__(self, pathdata, transform=None):
        self.path = pathdata
        self.list_filenames = [name for name in os.listdir(pathdata)]
        self.transform = transform

    def __len__(self):
        return len(self.list_filenames)

    def __getitem__(self, idx):
        name_scene = self.list_filenames[idx]
        seq_path = self.path + name_scene
        seq_list = load(seq_path)
        seq_tensor = [stack(tensores, dim=0) for tensores in seq_list]    
        seq_tensor = cat(seq_tensor,dim=0).to(float32)/255.0

        return name_scene, seq_tensor

        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        # return image, label