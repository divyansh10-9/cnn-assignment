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


def train_classifier(model, dataloader, val_loader, device, epochs=30, lr=1e-4):

    print("Starting classifier training...")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(epochs):

        # --- Train ---
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):

            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}")

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        # --- Validation ---
        model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step()

        wandb.log({
            "classification_train_loss": avg_loss,
            "classification_val_loss": avg_val_loss,
            "classification_val_acc": val_acc,
            "epoch": epoch + 1
        })

        print(f"[Classifier] Epoch {epoch+1}/{epochs}  "
              f"Train Loss: {avg_loss:.4f}  Val Loss: {avg_val_loss:.4f}  Val Acc: {val_acc:.4f}")


def train_localizer(model, dataloader, val_loader, device, epochs=30, lr=1e-4):

    print("Starting localization training...")

    iou_loss_fn = IoULoss()
    smooth_l1 = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):

            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}")

            images = batch["image"].to(device)
            # bbox is normalized [0,1] (x_center, y_center, width, height)
            bbox = batch["bbox"].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Combined SmoothL1 + IoU loss on normalized coords
            loss = smooth_l1(outputs, bbox) + iou_loss_fn(outputs, bbox)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        # Validation IoU
        model.eval()
        val_iou = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                bbox = batch["bbox"].to(device)
                outputs = model(images)
                val_iou += (1 - iou_loss_fn(outputs, bbox)).item()
                val_batches += 1

        avg_val_iou = val_iou / val_batches
        scheduler.step()

        wandb.log({
            "localization_train_loss": avg_loss,
            "localization_val_iou": avg_val_iou,
            "epoch": epoch + 1
        })

        print(f"[Localizer] Epoch {epoch+1}/{epochs}  "
              f"Train Loss: {avg_loss:.4f}  Val IoU: {avg_val_iou:.4f}")


def dice_loss(pred, target, eps=1e-6):

    pred = torch.softmax(pred, dim=1)

    target_one_hot = torch.nn.functional.one_hot(
        target, num_classes=pred.shape[1]
    ).permute(0, 3, 1, 2).float()

    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

    dice = (2 * intersection + eps) / (union + eps)

    return 1 - dice.mean()


def train_segmenter(model, dataloader, val_loader, device, epochs=30, lr=1e-4):

    print("Starting segmentation training...")

    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):

            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}")

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Combined Dice + CE loss for better convergence
            loss = dice_loss(outputs, masks) + ce_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        # Validation Dice
        model.eval()
        val_dice = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                outputs = model(images)
                val_dice += (1 - dice_loss(outputs, masks)).item()
                val_batches += 1

        avg_val_dice = val_dice / val_batches
        scheduler.step()

        wandb.log({
            "segmentation_train_loss": avg_loss,
            "segmentation_val_dice": avg_val_dice,
            "epoch": epoch + 1
        })

        print(f"[Segmenter] Epoch {epoch+1}/{epochs}  "
              f"Train Loss: {avg_loss:.4f}  Val Dice: {avg_val_dice:.4f}")


def main(args):

    print("Initializing training...")

    wandb.init(
        project="Oxford-IIIT-Pet-Multitask",
        config={
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "task": args.task,
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")

    train_dataset = OxfordIIITPetDataset(root=args.data_root, split="train")
    val_dataset   = OxfordIIITPetDataset(root=args.data_root, split="val")

    print(f"Train size: {len(train_dataset)}  Val size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    if args.task == "classification":

        print("Initializing classification model...")
        model = VGG11Classifier().to(device)
        train_classifier(model, train_loader, val_loader, device,
                         epochs=args.epochs, lr=args.lr)
        torch.save(model.state_dict(), "checkpoints/classifier.pth")
        print("Saved checkpoints/classifier.pth")

    elif args.task == "localization":

        print("Initializing localization model...")
        model = VGG11Localizer().to(device)
        train_localizer(model, train_loader, val_loader, device,
                        epochs=args.epochs, lr=args.lr)
        torch.save(model.state_dict(), "checkpoints/localizer.pth")
        print("Saved checkpoints/localizer.pth")

    elif args.task == "segmentation":

        print("Initializing segmentation model...")
        model = VGG11UNet().to(device)
        train_segmenter(model, train_loader, val_loader, device,
                        epochs=args.epochs, lr=args.lr)
        torch.save(model.state_dict(), "checkpoints/unet.pth")
        print("Saved checkpoints/unet.pth")

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
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--lr",         type=float, default=1e-4)

    args = parser.parse_args()
    main(args)
