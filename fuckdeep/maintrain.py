import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from configs.cfg import Cfg
from utils.device import get_device
from data.dataset import FruitHFDataset
from data.transforms import get_transforms
from models.small_cnn import SmallCNN
from engine.train import train_loop

def main():
    device = get_device()

    dataset = load_dataset("Densu341/Fresh-rotten-fruit")  # 이미 캐시되어 있으면 빨라짐
    train_tf, val_tf = get_transforms(Cfg.IMAGE_SIZE)

    train_loader = DataLoader(
        FruitHFDataset(dataset["train"], transform=train_tf),
        batch_size=Cfg.BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=Cfg.NUM_WORKERS, pin_memory=(device.type=="cuda"),
    )
    val_loader = DataLoader(
        FruitHFDataset(dataset["test"], transform=val_tf),
        batch_size=Cfg.BATCH_SIZE, shuffle=False,
        num_workers=Cfg.NUM_WORKERS, pin_memory=(device.type=="cuda"),
    )

    model = SmallCNN(num_classes=Cfg.NUM_CLASSES, multitask=Cfg.MULTITASK).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Cfg.LR, weight_decay=5e-4)

    train_loop(model, train_loader, val_loader, optimizer, Cfg.EPOCHS, device,
               multitask=Cfg.MULTITASK, scheduler=None, grad_clip=1.0)

if __name__ == "__main__":
    main()