"""Training entrypoint
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import wandb

from datasets.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss


def train_classifier(model, dataloader, device, epochs=100, lr=1e-4):

    print("Starting classifier training...")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):

            if batch_idx % 50 == 0:
                print(f"Processing batch {batch_idx}/{len(dataloader)}")

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        wandb.log({
            "classification_loss": avg_loss,
            "epoch": epoch + 1
        })

        print(f"[Classifier] Epoch {epoch+1} Loss: {avg_loss:.4f}")


def train_localizer(model, dataloader, device, epochs=100, lr=1e-4):

    print("Starting localization training...")

    mse_loss = nn.MSELoss()
    iou_loss = IoULoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):

            if batch_idx % 50 == 0:
                print(f"Processing batch {batch_idx}/{len(dataloader)}")

            images = batch["image"].to(device)
            bbox = batch["bbox"].to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = mse_loss(outputs, bbox) + iou_loss(outputs, bbox)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        wandb.log({
            "localization_loss": avg_loss,
            "epoch": epoch + 1
        })

        print(f"[Localizer] Epoch {epoch+1} Loss: {avg_loss:.4f}")


def dice_loss(pred, target, eps=1e-6):

    pred = torch.softmax(pred, dim=1)

    target_one_hot = torch.nn.functional.one_hot(
        target, num_classes=pred.shape[1]
    ).permute(0, 3, 1, 2).float()

    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

    dice = (2 * intersection + eps) / (union + eps)

    return 1 - dice.mean()


def train_segmenter(model, dataloader, device, epochs=100, lr=1e-4):

    print("Starting segmentation training...")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):

            if batch_idx % 50 == 0:
                print(f"Processing batch {batch_idx}/{len(dataloader)}")

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = dice_loss(outputs, masks)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        wandb.log({
            "segmentation_loss": avg_loss,
            "epoch": epoch + 1
        })

        print(f"[Segmentation] Epoch {epoch+1} Loss: {avg_loss:.4f}")


def main(args):

    print("Initializing training...")

    wandb.init(
        project="Oxford-IIIT-Pet-Multitask",
        config={
            "batch_size": args.batch_size,
            "epochs": 5,
            "lr": 1e-4
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")

    dataset = OxfordIIITPetDataset(
        root=args.data_root,
        split="train"
    )

    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )

    print("DataLoader ready")

    if args.task == "classification":

        print("Initializing classification model...")
        model = VGG11Classifier().to(device)

        train_classifier(model, dataloader, device)

        print("Saving classifier model...")
        torch.save(model.state_dict(), "classifier.pth")

    elif args.task == "localization":

        print("Initializing localization model...")
        model = VGG11Localizer().to(device)

        train_localizer(model, dataloader, device)

        print("Saving localizer model...")
        torch.save(model.state_dict(), "localizer.pth")

    elif args.task == "segmentation":

        print("Initializing segmentation model...")
        model = VGG11UNet().to(device)

        train_segmenter(model, dataloader, device)

        print("Saving segmentation model...")
        torch.save(model.state_dict(), "unet.pth")

    wandb.finish()

    print("Training completed!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "localization", "segmentation"],
    )
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    main(args)