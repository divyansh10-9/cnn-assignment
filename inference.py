"""Inference and evaluation
"""

import torch
from torch.utils.data import DataLoader
import argparse

from datasets.pets_dataset import OxfordIIITPetDataset
from models.multitask import MultiTaskPerceptionModel


def run_inference(model, dataloader, device):

    model.eval()

    results = []

    with torch.no_grad():

        for batch in dataloader:

            images = batch["image"].to(device)

            outputs = model(images)

            classification = outputs["classification"]
            localization = outputs["localization"]
            segmentation = outputs["segmentation"]

            preds = torch.argmax(classification, dim=1)

            results.append({
                "classification": preds.cpu(),
                "bbox": localization.cpu(),
                "segmentation": segmentation.cpu()
            })

    return results


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = OxfordIIITPetDataset(
        root=args.data_root,
        split="test"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    model = MultiTaskPerceptionModel()
    model.to(device)

    results = run_inference(model, dataloader, device)

    print("Inference completed")
    print(f"Total batches processed: {len(results)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    main(args)