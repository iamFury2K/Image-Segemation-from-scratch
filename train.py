import torch
import torch.nn as nn
import  torch.optim as optim
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from architecture.UNET import UNET
from utils.utils import *

# Setting up the Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PIN_MEMORY = True
NUM_WORKERS = 2
LOAD_MODEL = False
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
TRAIN_IMG_DIR = 'data/train_images'
TRAIN_MASK_DIR = 'data/train_masks'
VAL_IMG_DIR = 'data/val_images'
VAL_MASK_DIR = 'data/val_masks'

def train_fn(loader, model, loss_fn, optimizer, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, target) in enumerate(loop):
        data = data.permute(0,3,1,2).to(device=DEVICE).float()
        target = target.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, target)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=[0.0, 0.0, 0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0),

    ToTensorV2()
    ]
    )

    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0),
    ToTensorV2()
    ]
    )
    model = UNET(in_channels=3, out_channels=1).to(device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY
    )

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        # saving checkpoint
        checkpoint = {
            "state_dict" : model.state_dict(),
            "optimizer" : optimizer.state_dict()
        }
        save_checkpoint(checkpoint)

        check_accuracy(val_loader, model, device=DEVICE)

        save_predictions_as_img(val_loader, model, dir="saved_images/", device=DEVICE)


    

if __name__ == '__main__':
    main()