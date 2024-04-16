import torch
from torch.utils.data import DataLoader
from dataloader.dataset import ImageDataset
import torch.nn as nn
import torchvision
def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    print("=> Saving checkpoint {} ".format(filename))
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])


def get_loaders(
        train_dir,
        train_mask_dir,
        val_dir,
        val_mask_dir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True):

    train_ds = ImageDataset(
        image_dir=train_dir,
        mask_dir=train_mask_dir,
        transform=train_transform
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )
    val_ds = ImageDataset(
        image_dir=val_dir,
        mask_dir=val_mask_dir,
        transform=val_transform
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )
    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixel = 0
    model.eval()
    dice_score = 0
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device).unsqueeze(1)
            preds = nn.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixel += torch.numel(num_correct)
            dice_score += 2 * (preds * y).sum() / (preds + y).sum() + 1e-8

    print(f"Accuracy: {num_correct/num_pixel*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_img(loader, model, dir="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.inference_mode():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds,  f"{dir}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), "{dir}/tar_{idx}.png")

    model.train()


