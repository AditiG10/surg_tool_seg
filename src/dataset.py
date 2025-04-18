import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SurgicalDataset(Dataset):
    def __init__(self, img_dir, msk_dir=None, tfm=None, img_sz=(512, 512), n_cls=9):
        self.img_pth = sorted(glob.glob(os.path.join(img_dir, '*')))
        self.msk_pth = sorted(glob.glob(os.path.join(msk_dir, '*'))) if msk_dir else None
        self.tfm = tfm
        self.img_sz = img_sz
        self.n_cls = n_cls

    def __len__(self):
        return len(self.img_pth)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_pth[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_sz)

        if self.msk_pth:
            msk = cv2.imread(self.msk_pth[idx], cv2.IMREAD_GRAYSCALE)
            msk = cv2.resize(msk, self.img_sz, interpolation=cv2.INTER_NEAREST)

            if self.tfm:
                aug = self.tfm(image=img, mask=msk)
                img, msk = aug['image'], aug['mask'].long()

            else:
                img = transforms.ToTensor()(img)
                msk = torch.from_numpy(msk).long()

            return img, msk

        else:
            if self.tfm:
                aug = self.tfm(image=img)
                img = aug['image']
            else:
                img = transforms.ToTensor()(img)

            return img


def get_training_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.95, 1.05), translate_percent=0.05, rotate=(-15, 15), p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_validation_augmentations():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
