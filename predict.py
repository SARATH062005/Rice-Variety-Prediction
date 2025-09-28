# ====================================================
# Imports
# ====================================================
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import torch.nn.functional as F

# ====================================================
# Configuration
# ====================================================
class CFG:
    TEST_DIR = 'test/'
    TRAIN_CSV = 'train.csv'
    SUBMISSION_CSV = 'submission.csv'
    PLOT_DIR = 'prediction_plots/'

    MODEL_PATHS = ['best_model_fold_0.pth', 'best_model_fold_1.pth']
    MODEL_NAME = 'efficientnet_b0'
    IMG_SIZE = 256
    BATCH_SIZE = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 0  # set >0 if Linux

# create plot directory if not exists
os.makedirs(CFG.PLOT_DIR, exist_ok=True)

# ====================================================
# Dataset
# ====================================================
class PlantTestDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = self.df['ID'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.randn(3, CFG.IMG_SIZE, CFG.IMG_SIZE), "error"

        if self.transform:
            image = self.transform(image)
        return image, img_name

# ====================================================
# Augmentations (for TTA)
# ====================================================
def get_tta_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.RandomResizedCrop(CFG.IMG_SIZE, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAutocontrast(0.3),
        transforms.RandomGrayscale(0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])

# ====================================================
# Model
# ====================================================
def create_model(num_classes):
    model = timm.create_model(
        CFG.MODEL_NAME,
        pretrained=False,
        num_classes=num_classes
    )
    return model.to(CFG.DEVICE)

# ====================================================
# Plotting function
# ====================================================
def plot_prediction(image_name, model_probs, best_model_idx):
    models = [f"Model {i+1}" for i in range(len(model_probs))]
    probs = [p.item() for p in model_probs]
    colors = ['green' if i == best_model_idx else 'skyblue' for i in range(len(probs))]

    plt.figure(figsize=(6,4))
    plt.bar(models, probs, color=colors)
    plt.ylim(0, 1)
    plt.title(f"Image: {image_name}\nBest model: {models[best_model_idx]}")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.PLOT_DIR, f"{image_name}_prediction.png"))
    plt.close()

# ====================================================
# Ensemble Prediction with logging & plotting
# ====================================================
def predict_with_ensemble(models, loader, class_names):
    all_probs, all_ids, best_models = [], [], []

    with torch.no_grad():
        for images, filenames in tqdm(loader, desc="Predicting (ensemble)"):
            if "error" in filenames:
                continue
            images = images.to(CFG.DEVICE)
            batch_probs = []
            for model in models:
                outputs = model(images)
                batch_probs.append(F.softmax(outputs, dim=1))
            batch_probs_stack = torch.stack(batch_probs)  # [num_models, batch_size, num_classes]

            avg_prob = torch.mean(batch_probs_stack, dim=0)
            all_probs.append(avg_prob.cpu())
            all_ids.extend(filenames)

            pred_classes = torch.argmax(avg_prob, dim=1)
            for i, p_class in enumerate(pred_classes):
                model_probs = batch_probs_stack[:, i, p_class]
                best_model_idx = torch.argmax(model_probs).item()
                best_model_name = CFG.MODEL_PATHS[best_model_idx]
                best_models.append(best_model_name)

                # Print log
                print(f"Image: {filenames[i]} | Predicted: {class_names[p_class]} | "
                      f"Best Model: {best_model_name} | "
                      f"Model probs: {[round(p.item(),4) for p in model_probs]}")

                # Plot for first few images (or all)
                plot_prediction(filenames[i], model_probs, best_model_idx)

    all_probs = torch.cat(all_probs)
    preds = torch.argmax(all_probs, dim=1).numpy()
    return preds, all_ids, best_models

# ====================================================
# Main
# ====================================================
def main():
    print(f"Using device: {CFG.DEVICE}")
    print(f"Models: {CFG.MODEL_PATHS}")

    # Label mapping
    train_df = pd.read_csv(CFG.TRAIN_CSV)
    class_names = sorted(train_df['TARGET'].unique())
    idx_to_class = {i: name for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    # Data
    test_files = os.listdir(CFG.TEST_DIR)
    test_df = pd.DataFrame({'ID': test_files})
    tta_transform = get_tta_transform()
    test_dataset = PlantTestDataset(test_df, CFG.TEST_DIR, tta_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS
    )

    # Load models
    models = []
    for path in CFG.MODEL_PATHS:
        m = create_model(num_classes)
        m.load_state_dict(torch.load(path, map_location=CFG.DEVICE))
        m.eval()
        models.append(m)
        print(f"Loaded weights: {path}")

    preds, ids, best_model_for_image = predict_with_ensemble(models, test_loader, class_names)
    predicted_labels = [idx_to_class[p] for p in preds]

    # Save submission
    submission_df = pd.DataFrame({
        'ID': ids,
        'TARGET': predicted_labels
    })
    submission_df.to_csv(CFG.SUBMISSION_CSV, index=False)
    print(f"\n✅ Saved predictions to {CFG.SUBMISSION_CSV}")
    print(submission_df.head())

if __name__ == '__main__':
    main()