import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import SimpleITK as sitk
import torchvision.transforms as transforms
import torch


class SpineAgeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: Path, nifti_file: str, transform=transforms.ToTensor()):
        self.df = df
        self.transform = transform
        self.df['image_filepath'] = self.df['Study_ID'].apply(lambda x: data_root / x / nifti_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # read image
        sitk_image = sitk.ReadImage(self.df['image_filepath'].iloc[index])

        # get result in the form of a numpy array
        np_image = sitk.GetArrayFromImage(sitk_image)

        # preprocessing for transforming to tensor and training
        np_image = np_image.transpose(1, 2, 0) # to have DxHxW from WxDxH
        # normalize image
        if np.max(np_image) != 0:
            np_image = np_image / np.max(np_image)
        else:
            np_image = np_image / 1.0
        
        # convert to torch image with the option of augmentation
        image = self.transform(np_image).float()
        image = image.unsqueeze(0) # 1xDxHxW

        # read label
        y = torch.tensor(self.df['Age'].iloc[index]).float() # fix the dimension and Float issue

        # finding study id
        study_id = self.df['Study_ID'].iloc[index]
        
        return {
            "t2_whole_spine": {"mask": image},
            "age": y,
            "study_id": study_id
        }

def make_loader(stage: str, batch_size: int, shuffle: bool, data_root: Path, df_folder: Path, nifti_file: str, num_workers: int=0):
    assert stage in ['train', 'val', 'test'], 'stage should be train, val, or test'
    df = pd.read_csv(df_folder / f'{stage}.csv')
    dataset = SpineAgeDataset(df, data_root, nifti_file)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
