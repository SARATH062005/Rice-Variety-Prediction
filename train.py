# ====================================================
# Imports
# ====================================================
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm  # PyTorch Image Models

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
# --- Extra for metrics ---
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support
)


# ====================================================
# Configuration
# ====================================================
class CFG:
    TRAIN_DIR = 'data/train/'
    TRAIN_CSV = 'data/train.csv'

    MODEL_NAME = 'efficientnet_b0'
    IMG_SIZE = 256
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4

    N_SPLITS = 5
    RANDOM_STATE = 42

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 0  # 0 for Windows

# ====================================================
# Dataset
# ====================================================
class PlantDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = self.df['ID'].values
        self.labels = self.df['label_idx'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.randn(3, CFG.IMG_SIZE, CFG.IMG_SIZE), torch.tensor(-1, dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label

# ====================================================
# Transforms
# ====================================================
def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(CFG.IMG_SIZE, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),               # ✅ vertical flip
        transforms.RandomAffine(
            degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
        ),                                                   # ✅ affine distortions
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        ),                                                   # ✅ stronger color jitter
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),  # ✅ occlusion
    ])

    val_tf = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE + 32, CFG.IMG_SIZE + 32)),
        transforms.CenterCrop(CFG.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf
# ====================================================
# Model
# ====================================================
def create_model(num_classes):
    model = timm.create_model(
        CFG.MODEL_NAME,
        pretrained=True,
        num_classes=num_classes
    )
    return model.to(CFG.DEVICE)

# ====================================================
# Training helpers
# ====================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, all_preds, all_labels = 0.0, [], []
    pbar = tqdm(loader, desc="Training")
    for x, y in pbar:
        if -1 in y: continue
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        pbar.set_postfix(loss=loss.item())
    return running_loss/len(loader.dataset), f1_score(all_labels, all_preds, average='micro')

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating")
        for x, y in pbar:
            if -1 in y: continue
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            running_loss += loss.item() * x.size(0)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            pbar.set_postfix(loss=loss.item())
    return running_loss/len(loader.dataset), f1_score(all_labels, all_preds, average='micro')

# ====================================================
# Main
# ====================================================
def main():
    print(f"Using device: {CFG.DEVICE}")
    df = pd.read_csv(CFG.TRAIN_CSV)

    class_names = sorted(df['TARGET'].unique())
    class_to_idx = {n: i for i, n in enumerate(class_names)}
    idx_to_class = {i: n for n, i in class_to_idx.items()}
    num_classes = len(class_names)
    df['label_idx'] = df['TARGET'].map(class_to_idx)

    skf = StratifiedKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=CFG.RANDOM_STATE)
    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df['label_idx'])):
        df.loc[val_idx, 'fold'] = fold

    for fold in range(CFG.N_SPLITS):
        print(f"\n{'='*40}\n FOLD {fold} / {CFG.N_SPLITS-1}\n{'='*40}")
        train_df = df[df.fold != fold].reset_index(drop=True)
        val_df   = df[df.fold == fold].reset_index(drop=True)

        train_tf, val_tf = get_transforms()
        train_ds = PlantDataset(train_df, CFG.TRAIN_DIR, train_tf)
        val_ds   = PlantDataset(val_df,   CFG.TRAIN_DIR, val_tf)

        # Weighted sampler and loss
        counts = train_df['label_idx'].value_counts().sort_index().values
        sampler_weights = 1. / torch.tensor(counts, dtype=torch.float)
        sample_weights = sampler_weights[train_df['label_idx']]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        class_weights = compute_class_weight('balanced', classes=np.unique(train_df['label_idx']), y=train_df['label_idx'])
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(CFG.DEVICE))

        # train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, sampler=sampler, num_workers=CFG.NUM_WORKERS)
        train_loader = DataLoader(train_ds,
                          batch_size=CFG.BATCH_SIZE,
                          shuffle=True,
                          num_workers=8,
                          pin_memory=True,
                          prefetch_factor=4)

        val_loader   = DataLoader(val_ds,   batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)

        model = create_model(num_classes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)

        best_val_f1 = 0.0
        train_loss_hist, val_loss_hist = [], []

        # ---- Training Loop ----
        for epoch in range(CFG.NUM_EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{CFG.NUM_EPOCHS} ---")
            tr_loss, tr_f1 = train_one_epoch(model, train_loader, optimizer, criterion, CFG.DEVICE)
            va_loss, va_f1 = validate_one_epoch(model, val_loader, criterion, CFG.DEVICE)
            train_loss_hist.append(tr_loss)
            val_loss_hist.append(va_loss)
            scheduler.step(va_f1)
            print(f"Train -> Loss: {tr_loss:.4f} | Micro F1: {tr_f1:.4f}")
            print(f"Valid -> Loss: {va_loss:.4f} | Micro F1: {va_f1:.4f}")
            if va_f1 > best_val_f1:
                best_val_f1 = va_f1
                torch.save(model.state_dict(), f'best_model_fold_{fold}.pth')
                print(f"⭐ Validation F1 improved to {best_val_f1:.4f}")

        # ---- Analysis/Plots AFTER training ----
        print("[INFO] Generating analysis plots for fold", fold)
        os.makedirs("runs/metrics", exist_ok=True)

        # Load best model for final validation predictions
        model.load_state_dict(torch.load(f'best_model_fold_{fold}.pth'))
        model.eval()
        final_preds, final_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                if -1 in y: continue
                x, y = x.to(CFG.DEVICE), y.to(CFG.DEVICE)
                out = model(x)
                final_preds.extend(out.argmax(1).cpu().numpy())
                final_labels.extend(y.cpu().numpy())
        final_preds = np.array(final_preds)
        final_labels = np.array(final_labels)

        # Confusion Matrix
        cm = confusion_matrix(final_labels, final_preds, normalize='true')
        fig_cm, ax_cm = plt.subplots(figsize=(6,6))
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap='Blues', values_format=".2f", ax=ax_cm, colorbar=False)
        ax_cm.set_title(f"Fold {fold} – Normalized Confusion Matrix")
        plt.tight_layout()
        fig_cm.savefig(f"runs/metrics/fold{fold}_confusion_matrix.png", dpi=300)
        plt.close(fig_cm)

        # Per-Class Metrics
        prec, rec, f1, _ = precision_recall_fscore_support(final_labels, final_preds, average=None)
        classes = [idx_to_class[i] for i in range(len(prec))]
        x = np.arange(len(classes))
        width = 0.25
        fig_prf, ax_prf = plt.subplots(figsize=(8,4))
        ax_prf.bar(x - width, prec, width, label='Precision')
        ax_prf.bar(x,       rec,  width, label='Recall')
        ax_prf.bar(x + width, f1, width, label='F1')
        ax_prf.set_xticks(x)
        ax_prf.set_xticklabels(classes, rotation=45, ha='right')
        ax_prf.set_ylim(0, 1)
        ax_prf.set_ylabel("Score")
        ax_prf.set_title(f"Fold {fold} – Per-Class Metrics")
        ax_prf.legend()
        plt.tight_layout()
        fig_prf.savefig(f"runs/metrics/fold{fold}_per_class_prf.png", dpi=300)
        plt.close(fig_prf)

        # Training vs Validation Loss Curves
        fig_curve, ax_curve = plt.subplots()
        ax_curve.plot(train_loss_hist, label='Train Loss')
        ax_curve.plot(val_loss_hist, label='Validation Loss')
        ax_curve.set_xlabel("Epoch")
        ax_curve.set_ylabel("Loss")
        ax_curve.set_title(f"Fold {fold} – Training vs Validation Loss")
        ax_curve.legend()
        plt.tight_layout()
        fig_curve.savefig(f"runs/metrics/fold{fold}_loss_curves.png", dpi=300)
        plt.close(fig_curve)
        print(f"[INFO] Saved analysis plots to runs/metrics/ for fold {fold}")

    print("\n" + "="*40)
    print("      TRAINING COMPLETE      ")
    print("Best models saved as 'best_model_fold_*.pth'")
    print("="*40)

if __name__ == '__main__':
    main()
        