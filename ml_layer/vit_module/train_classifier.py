"""
Fine-tunes a LINEAR CLASSIFICATION HEAD on top of CLIP's frozen ViT image
encoder. This is real training: forward pass through frozen CLIP to get
image embeddings, then a trainable nn.Linear layer classifies those
embeddings, updated via backpropagation over several epochs.

Why this approach: training the full ViT would need far more data and
compute than a laptop can handle. Linear probing on frozen features is
a standard, well-established transfer learning technique that gets
strong results with a tiny dataset and CPU-only training.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.dirname(__file__)

EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE = 0.001


class FoodImageDataset(Dataset):
    def __init__(self, root_dir, processor):
        self.samples = []
        self.processor = processor
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return image, label


def collate_fn(batch, processor):
    images, labels = zip(*batch)
    inputs = processor(images=list(images), return_tensors="pt")
    return inputs["pixel_values"], torch.tensor(labels)


def extract_features(clip_model, pixel_values):
    # Bypass get_image_features() entirely - its return type has changed
    # across transformers versions (plain tensor vs wrapped object vs
    # tuple), causing repeated breakage. Instead we call the underlying
    # vision tower + projection layer directly - this is exactly what
    # get_image_features() does internally, so results are identical,
    # but the return shape here is stable and always a plain tensor.
    with torch.no_grad():
        vision_outputs = clip_model.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output
        image_features = clip_model.visual_projection(pooled_output)
    return image_features


def train():
    print("Loading CLIP (frozen feature extractor)...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False  # freeze CLIP - only the head trains

    train_dataset = FoodImageDataset(os.path.join(DATA_DIR, "train"), processor)
    val_dataset = FoodImageDataset(os.path.join(DATA_DIR, "val"), processor)

    num_classes = len(train_dataset.classes)
    print(f"Classes ({num_classes}): {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               collate_fn=lambda b: collate_fn(b, processor))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, processor))

    feature_dim = clip_model.config.projection_dim  # 512 for this CLIP variant
    classifier_head = nn.Linear(feature_dim, num_classes)

    optimizer = torch.optim.Adam(classifier_head.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    best_state_dict = None

    print("Starting training (only the linear head has trainable weights)...")
    for epoch in range(EPOCHS):
        classifier_head.train()
        epoch_loss = 0.0

        for pixel_values, labels in train_loader:
            features = extract_features(clip_model, pixel_values)
            logits = classifier_head(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        classifier_head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for pixel_values, labels in val_loader:
                features = extract_features(clip_model, pixel_values)
                logits = classifier_head(features)
                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total > 0 else 0
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.clone() for k, v in classifier_head.state_dict().items()}

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Val Accuracy: {val_acc:.4f}")

    # Save the trained head + class labels
    torch.save(best_state_dict, os.path.join(OUTPUT_DIR, "classifier_head.pt"))
    print(f"Saved BEST checkpoint (val accuracy: {best_val_acc:.4f}), not the final epoch.")
    with open(os.path.join(OUTPUT_DIR, "labels.json"), "w") as f:
        json.dump(train_dataset.classes, f)

    print("Saved classifier_head.pt and labels.json")

    # Plot and save training curves - THIS is your evidence for the report
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, marker='o')
    ax1.set_title("Training Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2.plot(val_accuracies, marker='o', color='green')
    ax2.set_title("Validation Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"))
    print("Saved training_curves.png - use this in your project report")


if __name__ == "__main__":
    train()

