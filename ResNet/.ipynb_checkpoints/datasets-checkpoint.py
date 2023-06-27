from collections import defaultdict
from pathlib import Path
import random

import pandas as pd
from PIL import Image
import torch
import torchvision

import common


class PatchTrainDataset:
    def __init__(self, dataset_index: Path, experimentType:str) -> None:
        df = pd.read_csv(dataset_index, index_col=0)
        df = self.get_subset(df)

        patches = defaultdict(list)
        labels = {}

        for path, row in df.iterrows():
            patches[row.Image].append(path)
            labels[row.Image] = row.Profile

        sorted_keys = sorted(patches.keys())
        self.patches = [patches[k] for k in sorted_keys]
        self.labels = [labels[k] for k in sorted_keys]
        if(experimentType=="Single" or experimentType=="Cross"):
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(common.mean,common.std)
            ])
        elif(experimentType=="Multi"):
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                #torchvision.transforms.Normalize(common.mean,common.std)
            ])

    def get_subset(self, df):
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        path = random.choice(self.patches[idx])
        profile = self.labels[idx]
        image = self.transform(Image.open(path))
        return image, common.PROFILES_IDS[profile]


class PatchTestDataset:
    def __init__(self, dataset_index: Path,experimentType:str) -> None:
        df = pd.read_csv(dataset_index, index_col=0)
        df = self.get_subset(df)

        self.data = [(path, row.Profile, row.Image) for path, row in df.iterrows()]
        if(experimentType=="Single" or experimentType=="Cross"):
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(common.mean,common.std)
            ])
        elif(experimentType=="Multi"):
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                #torchvision.transforms.Normalize(common.mean,common.std)
            ])

    def get_subset(self, df):
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        path, profile, img = self.data[idx]
        image = self.transform(Image.open(path))
        return image, common.PROFILES_IDS[profile],img


class UnknownProfileTrainDataset(PatchTrainDataset):
    def __init__(self, dataset_index: Path,experimentType:str, fold: int, test_profile: str) -> None:
        self.fold = fold
        self.test_profile = test_profile

        super().__init__(dataset_index,experimentType)

    def get_subset(self, df):
        return df[(df.Fold != self.fold) & (df.Profile != self.test_profile)]


class UnknownProfileTestDataset(PatchTestDataset):
    def __init__(self, dataset_index: Path,experimentType:str, fold: int, test_profile: str) -> None:
        self.fold = fold
        self.test_profile = test_profile

        super().__init__(dataset_index,experimentType)

    def get_subset(self, df):
        return df[(df.Fold == self.fold) | (df.Profile == self.test_profile)]


class CrossSocialTrainDataset(PatchTrainDataset):
    def __init__(self, dataset_index: Path,experimentType:str, test_social: str) -> None:
        self.test_social = test_social

        super().__init__(dataset_index,experimentType)

    def get_subset(self, df):
        return df[df.Social != self.test_social]


class CrossSocialTestDataset(PatchTestDataset):
    def __init__(self, dataset_index: Path,experimentType:str, test_social: str) -> None:
        self.test_social = test_social

        super().__init__(dataset_index,experimentType)

    def get_subset(self, df):
        return df[df.Social == self.test_social]

    
#Single-social Train   
class OneSocialDatasetTrain(PatchTrainDataset):
    def __init__(self, dataset_index: Path, experimentType:str, fold: int, one_social: str) -> None:
        self.fold = fold
        self.one_social = one_social

        super().__init__(dataset_index,experimentType)

    def get_subset(self, df):
        return df[(df.Fold != self.fold) & (df.Social == self.one_social)]
#Single-social Test   
class OneSocialDatasetTest(PatchTestDataset):
    def __init__(self, dataset_index: Path,experimentType:str, fold: int, one_social: str) -> None:
        self.fold = fold
        self.one_social = one_social

        super().__init__(dataset_index,experimentType)

    def get_subset(self, df):
        return df[(df.Fold == self.fold) & (df.Social == self.one_social)]
    
class AllSocialDatasetTrain(PatchTrainDataset):
    def __init__(self, dataset_index: Path,experimentType:str, fold: int) -> None:
        self.fold = fold
        
        super().__init__(dataset_index,experimentType)

    def get_subset(self, df):
        return df[(df.Fold != self.fold)]
    
    
class AllSocialDatasetTest(PatchTestDataset):
    def __init__(self, dataset_index: Path,experimentType:str, fold: int) -> None:
        self.fold = fold
        

        super().__init__(dataset_index,experimentType)

    def get_subset(self, df):
        return df[(df.Fold == self.fold)]