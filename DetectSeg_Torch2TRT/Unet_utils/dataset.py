from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_head, masks_head, imgs_vac, masks_vac, scale=1):
        self.imgs_head = imgs_head
        self.masks_head = masks_head
        self.imgs_vac = imgs_vac
        self.masks_vac = masks_vac
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids_head = [splitext(file)[0] for file in listdir(imgs_head)
                    if not file.startswith('.')]
        self.ids_vac = [splitext(file)[0] for file in listdir(imgs_vac)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids_head)} examples')
        logging.info(f'Creating dataset with {len(self.ids_vac)} examples')

    def __len__(self):
        return min(len(self.ids_head), len(self.ids_vac))

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        # normalization
        mean1 = np.mean(img_trans)
        std1 = np.std(img_trans)
        img_trans = (img_trans - mean1 + 3.0 * std1) / 6 / std1
        img_trans = np.clip(img_trans, 0, 1)

        return img_trans

    def __getitem__(self, i):
        # get head image
        idx_head = self.ids_head[i]
        head_mask_file = glob(self.masks_head + idx_head + '*')
        head_img_file = glob(self.imgs_head + idx_head + '*')

        assert len(head_mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx_head}: {head_mask_file}'
        assert len(head_img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx_head}: {head_img_file}'
        head_mask = Image.open(head_mask_file[0])
        head_img = Image.open(head_img_file[0])

        assert head_img.size == head_mask.size, \
            f'Image and mask {idx_head} should be the same size, but are {head_img.size} and {head_mask.size}'

        head_img = self.preprocess(head_img, self.scale)
        head_mask = np.array(head_mask)
        head_mask[head_mask > 0] = 1
        head_mask = np.expand_dims(head_mask, axis=0)

        # get vac image
        idx_vac = self.ids_vac[i]
        vac_mask_file = glob(self.masks_vac + idx_vac + '*')
        vac_img_file = glob(self.imgs_vac + idx_vac + '*')

        assert len(vac_mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx_vac}: {vac_mask_file}'
        assert len(vac_img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx_vac}: {vac_img_file}'
        vac_mask = Image.open(vac_mask_file[0])
        vac_img = Image.open(vac_img_file[0])

        assert vac_img.size == vac_mask.size, \
            f'Image and mask {idx_vac} should be the same size, but are {vac_img.size} and {vac_mask.size}'

        vac_img = self.preprocess(vac_img, self.scale)
        vac_mask = np.array(vac_mask)
        vac_mask[vac_mask > 0] = 1
        vac_mask = np.expand_dims(vac_mask, axis=0)

        return {'head_image': torch.from_numpy(head_img), 'head_mask': torch.from_numpy(head_mask),
                'vac_image': torch.from_numpy(vac_img), 'vac_mask': torch.from_numpy(vac_mask)}
