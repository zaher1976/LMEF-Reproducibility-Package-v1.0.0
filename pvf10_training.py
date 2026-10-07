"""Auto-extracted from the corresponding Colab notebook.
Notebook magics/shell commands are preserved as comments where necessary.
For exact reproduction, use the notebook version.
"""

# %% [Code cell 1]
# 1) Setup
# COLAB SHELL: !pip -q install torch torchvision scikit-learn pillow pandas matplotlib seaborn

import os, re, time, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report)
import matplotlib.pyplot as plt
from PIL import Image

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0))

from google.colab import drive
drive.mount('/content/drive')

# %% [Code cell 2]
ROOT = '/content/drive/MyDrive/Datasets/PVF-10/PVF_10_Ori'
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

def normalize_class_name(raw_name):
    """Strip leading numeric prefix (e.g. '01bottom dirt' -> 'bottom dirt')."""
    return re.sub(r'^\d+', '', raw_name).strip().lower()

records = []
for split in ['train', 'test']:
    split_path = os.path.join(ROOT, split)
    for raw_cls in os.listdir(split_path):
        cls_path = os.path.join(split_path, raw_cls)
        if not os.path.isdir(cls_path):
            continue
        norm_cls = normalize_class_name(raw_cls)
        for fname in os.listdir(cls_path):
            if fname.lower().endswith(IMG_EXTS):
                records.append({
                    'path': os.path.join(cls_path, fname),
                    'class_name': norm_cls,
                    'original_split': split,
                })

df = pd.DataFrame(records)
print('Total images:', len(df))

class_names = sorted(df['class_name'].unique())
print(f'Detected {len(class_names)} unified classes:', class_names)
class_to_idx = {c: i for i, c in enumerate(class_names)}
df['label'] = df['class_name'].map(class_to_idx)

print('\nOverall class distribution:')
print(df['class_name'].value_counts())
print('\nOriginal split distribution:')
print(df['original_split'].value_counts())

# %% [Code cell 4]
sample_paths = df['path'].sample(min(20, len(df)), random_state=SEED).tolist()
modes = set()
dtypes = set()
ranges = []

for p in sample_paths:
    with Image.open(p) as im:
        modes.add(im.mode)
        arr = np.array(im)
        dtypes.add(str(arr.dtype))
        ranges.append((arr.min(), arr.max()))

print('Image modes found:', modes)
print('Numpy dtypes found:', dtypes)
print('Sample value ranges (min, max):', ranges[:10])

# %% [Code cell 6]
IMG_SIZE = 128

def load_and_preprocess(path, img_size=IMG_SIZE):
    with Image.open(path) as im:
        arr = np.array(im)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)  # convert to grayscale if RGB
        arr = arr.astype(np.float32)
        # normalize based on actual bit depth observed in the data, not assumed
        if arr.max() > 255:
            arr = arr / 65535.0   # 16-bit thermal
        else:
            arr = arr / 255.0     # 8-bit
        im2 = Image.fromarray((arr * 255).astype(np.uint8)).resize((img_size, img_size))
        return np.array(im2, dtype=np.float32) / 255.0

print('Caching preprocessed images (this may take a few minutes for 5,579 images)...')
t0 = time.time()
img_cache = {}
for i, p in enumerate(df['path']):
    try:
        img_cache[p] = load_and_preprocess(p)
    except Exception as e:
        print(f'[Error loading] {p}: {e}')
    if (i+1) % 1000 == 0:
        print(f'  {i+1}/{len(df)} cached...')
print(f'Cached {len(img_cache)} images in {time.time()-t0:.1f}s')

# drop any rows that failed to load
df = df[df['path'].isin(img_cache.keys())].reset_index(drop=True)
print('Final usable images:', len(df))

# %% [Code cell 8]
train_full = df[df['original_split'] == 'train'].reset_index(drop=True)
test_df = df[df['original_split'] == 'test'].reset_index(drop=True)

train_df, val_df = train_test_split(
    train_full, test_size=0.15, stratify=train_full['label'], random_state=SEED)

print('Train/Val/Test sizes:', len(train_df), len(val_df), len(test_df))
print('\nTrain distribution:')
print(train_df['class_name'].value_counts())
print('\nTest distribution (original, untouched):')
print(test_df['class_name'].value_counts())

# %% [Code cell 9]
class PVF10Dataset(Dataset):
    def __init__(self, frame, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        arr = img_cache[row['path']].copy()
        if self.augment:
            if np.random.rand() < 0.5:
                arr = np.fliplr(arr).copy()
            if np.random.rand() < 0.5:
                arr = np.flipud(arr).copy()
            k = np.random.choice([0, 1, 2, 3])
            arr = np.rot90(arr, k).copy()
        arr = arr[None, :, :]
        return torch.tensor(arr, dtype=torch.float32), torch.tensor(row['label'], dtype=torch.long)

BATCH = 32
train_loader = DataLoader(PVF10Dataset(train_df, augment=True), batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(PVF10Dataset(val_df,   augment=False), batch_size=BATCH, shuffle=False)
test_loader  = DataLoader(PVF10Dataset(test_df,  augment=False), batch_size=BATCH, shuffle=False)

# %% [Code cell 11]
class LightCNN10(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)
            n_flat = self.features(dummy).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(n_flat, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model = LightCNN10(num_classes=len(class_names)).to(device)
n_params = sum(p.numel() for p in model.parameters())
print('Total trainable parameters:', n_params, f'(~{n_params*4/1e6:.2f} MB fp32)')

# %% [Code cell 13]
counts = train_df['label'].value_counts().sort_index()
weights = torch.tensor([1.0 / counts.get(i, 1) for i in range(len(class_names))], dtype=torch.float32)
weights = (weights / weights.sum() * len(class_names)).to(device)
print('Class weights:', dict(zip(class_names, weights.cpu().numpy().round(2))))

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

EPOCHS = 40
PATIENCE = 8
best_val_loss = float('inf'); best_state = None; patience_ctr = 0
history = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
    train_loss /= len(train_df)

    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
    val_loss /= len(val_df)
    val_acc = correct / len(val_df)
    scheduler.step(val_loss)

    history.append({'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc})
    print(f'Epoch {epoch+1:02d}/{EPOCHS} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_ctr = 0
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f'Early stopping at epoch {epoch+1}')
            break

model.load_state_dict(best_state)
model.eval()

# %% [Code cell 15]
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        out = model(x)
        all_preds.extend(out.argmax(1).cpu().tolist())
        all_labels.extend(y.tolist())

acc = accuracy_score(all_labels, all_preds)
prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
cm = confusion_matrix(all_labels, all_preds)

print('=== TEST SET RESULTS (REAL, PVF-10, original test split) ===')
print('Accuracy:', acc)
print('Macro P/R/F1:', prec, rec, f1)
print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('PVF-10 Confusion Matrix (Real Test Set)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('pvf10_confusion_matrix.png', dpi=150)
plt.show()

# %% [Code cell 17]
def benchmark_latency(model, device, img_size=IMG_SIZE, n_runs=100, warmup=20):
    model.eval()
    dummy = torch.randn(1, 1, img_size, img_size).to(device)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        for _ in range(n_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 95))

lat_mean, lat_p95 = benchmark_latency(model, device)
print(f'Latency on {device.type.upper()}: mean={lat_mean:.2f} ms, p95={lat_p95:.2f} ms')

cpu_model = LightCNN10(num_classes=len(class_names))
cpu_model.load_state_dict(model.state_dict())
cpu_model.to('cpu')
lat_cpu_mean, lat_cpu_p95 = benchmark_latency(cpu_model, torch.device('cpu'))
print(f'Latency on CPU: mean={lat_cpu_mean:.2f} ms, p95={lat_cpu_p95:.2f} ms')

# %% [Code cell 19]
results = {
    'dataset': 'PVF-10 (real, original train/test split, .tif thermal images)',
    'classes': class_names,
    'n_train': len(train_df), 'n_val': len(val_df), 'n_test': len(test_df),
    'n_params': n_params,
    'test_accuracy': acc, 'macro_precision': prec, 'macro_recall': rec, 'macro_f1': f1,
    'confusion_matrix': cm.tolist(),
    'latency_ms_gpu_mean': lat_mean if device.type == 'cuda' else None,
    'latency_ms_cpu_mean': lat_cpu_mean,
    'latency_ms_cpu_p95': lat_cpu_p95,
    'epochs_run': len(history),
    'history': history,
}
with open('pvf10_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps({k: v for k, v in results.items() if k != 'history'}, indent=2))

from google.colab import files as colab_files
colab_files.download('pvf10_results.json')
colab_files.download('pvf10_confusion_matrix.png')
