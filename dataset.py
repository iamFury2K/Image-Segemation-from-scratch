import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        super(ImageDataset, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = None
        self.num_image = os.listdir(image_dir)

    def __len__(self):
        return len(self.num_image)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.num_image[index])
        mask_path = os.path.join(self.mask_dir, self.num_image[index]).replace(".jpg", ".png")
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask[mask==255.0] = 1

        if self.transform is not None:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation['image']
            mask = augmentation['mask']

        return image, mask

