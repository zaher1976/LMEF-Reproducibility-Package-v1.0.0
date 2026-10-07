"""Auto-extracted from the corresponding Colab notebook.
Notebook magics/shell commands are preserved as comments where necessary.
For exact reproduction, use the notebook version.
"""

# %% [Code cell 1]
# 1) Setup
# COLAB SHELL: !pip -q install torch torchvision scikit-learn pillow pandas matplotlib seaborn

import os, time, json, random, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report)
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0))

# %% [Code cell 3]
# Upload zip from local machine
from google.colab import files
print('Select the ZIP file containing the class folders...')
uploaded = files.upload()
zip_name = list(uploaded.keys())[0]

EXTRACT_DIR = '/content/dataset'
os.makedirs(EXTRACT_DIR, exist_ok=True)
with zipfile.ZipFile(zip_name, 'r') as z:
    z.extractall(EXTRACT_DIR)

print('Extracted to', EXTRACT_DIR)
for root, dirs, fnames in os.walk(EXTRACT_DIR):
    if fnames:
        print(root, '->', len(fnames), 'files')

# %% [Code cell 5]
# Optional: build a simple manifest for unlabeled / pilot thermal images
PILOT_DIR = '/content/pilot_images'  # change if needed
if os.path.isdir(PILOT_DIR):
    pilot_files = [f for f in os.listdir(PILOT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f'Found {len(pilot_files)} pilot thermal images (qualitative use only, not for quantitative metrics).')
else:
    print('No pilot directory found — skip if not using a small thermal pilot set.')

# %% [Code cell 7]
DATA_ROOT = EXTRACT_DIR  # from step 2-a

records = []
class_names = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
print('Detected classes:', class_names)

for cls in class_names:
    cls_dir = os.path.join(DATA_ROOT, cls)
    for fname in os.listdir(cls_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            records.append({'path': os.path.join(cls_dir, fname), 'class_name': cls})

df = pd.DataFrame(records)
class_to_idx = {c: i for i, c in enumerate(class_names)}
df['label'] = df['class_name'].map(class_to_idx)

print('Total images:', len(df))
print(df['class_name'].value_counts())

# %% [Code cell 9]
IMG_SIZE = 128  # adjust based on dataset resolution / GPU memory

from PIL import Image, ImageFilter, ImageOps

def quality_filter(path, min_size=20000):
    """Discard unreadable / near-empty images (quality filtering step)."""
    try:
        with Image.open(path) as im:
            im.verify()
        return os.path.getsize(path) > min_size
    except Exception:
        return False

print('Filtering low-quality images...')
df['ok'] = df['path'].apply(quality_filter)
n_before = len(df)
df = df[df['ok']].reset_index(drop=True)
print(f'Kept {len(df)} / {n_before} images after quality filtering')

def load_and_preprocess(path, img_size=IMG_SIZE):
    img = Image.open(path).convert('L')                  # grayscale (thermal-equivalent)
    img = ImageOps.autocontrast(img, cutoff=1)            # contrast enhancement
    img = img.filter(ImageFilter.MedianFilter(size=3))    # noise reduction
    img = img.resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32) / 255.0         # normalization
    return arr

# %% [Code cell 11]
print('Caching preprocessed images (this runs once)...')
t0 = time.time()
img_cache = {}
for p in df['path']:
    img_cache[p] = load_and_preprocess(p)
print(f'Cached {len(img_cache)} images in {time.time()-t0:.1f}s')

# %% [Code cell 13]
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df['label'], random_state=SEED)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df['label'], random_state=SEED)

print('Train/Val/Test sizes:', len(train_df), len(val_df), len(test_df))
print('\nTrain class distribution:')
print(train_df['class_name'].value_counts())

# %% [Code cell 14]
class ThermalFaultDataset(Dataset):
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
train_ds = ThermalFaultDataset(train_df, augment=True)
val_ds   = ThermalFaultDataset(val_df,   augment=False)
test_ds  = ThermalFaultDataset(test_df,  augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

# %% [Code cell 16]
class LightCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        # infer flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)
            n_flat = self.features(dummy).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = LightCNN(num_classes=len(class_names)).to(device)
n_params = sum(p.numel() for p in model.parameters())
print('Total trainable parameters:', n_params)
print(f'Approx. model size: {n_params * 4 / 1e6:.2f} MB (fp32)')

# %% [Code cell 18]
counts = train_df['label'].value_counts().sort_index()
weights = torch.tensor([1.0 / counts.get(i, 1) for i in range(len(class_names))], dtype=torch.float32)
weights = (weights / weights.sum() * len(class_names)).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

EPOCHS = 40
PATIENCE = 7
best_val_loss = float('inf')
best_state = None
patience_ctr = 0
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
    train_loss /= len(train_ds)

    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
    val_loss /= len(val_ds)
    val_acc = correct / len(val_ds)
    scheduler.step(val_loss)

    history.append({'epoch': epoch + 1, 'train_loss': train_loss,
                     'val_loss': val_loss, 'val_acc': val_acc})
    print(f'Epoch {epoch+1:02d}/{EPOCHS} | train_loss={train_loss:.4f} '
          f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}')

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

# %% [Code cell 20]
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        out = model(x)
        preds = out.argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())

acc = accuracy_score(all_labels, all_preds)
prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
cm = confusion_matrix(all_labels, all_preds)

print('=== TEST SET RESULTS (REAL) ===')
print('Accuracy:', acc)
print('Macro Precision:', prec, '| Macro Recall:', rec, '| Macro F1:', f1)
print('\nPer-class report:')
print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

# Confusion matrix plot
plt.figure(figsize=(8, 6))
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix (Test Set)')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# Binary-style FNR/FPR if a 'Normal' class exists
if 'normal' in [c.lower() for c in class_names] or 'Normal' in class_names:
    normal_idx = [i for i, c in enumerate(class_names) if c.lower() == 'normal'][0]
    y_true_bin = [0 if l == normal_idx else 1 for l in all_labels]
    y_pred_bin = [0 if p == normal_idx else 1 for p in all_preds]
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f'\nBinary fault-vs-normal | FNR: {fnr:.4f} | FPR: {fpr:.4f}')

# %% [Code cell 22]
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
print(f'Latency on {device.type.upper()} (batch=1): mean={lat_mean:.2f} ms | p95={lat_p95:.2f} ms')

# Also benchmark on CPU explicitly for comparison even if GPU is available
cpu_model = LightCNN(num_classes=len(class_names))
cpu_model.load_state_dict(model.state_dict())
cpu_model.to('cpu')
lat_cpu_mean, lat_cpu_p95 = benchmark_latency(cpu_model, torch.device('cpu'))
print(f'Latency on CPU (batch=1): mean={lat_cpu_mean:.2f} ms | p95={lat_cpu_p95:.2f} ms')

# %% [Code cell 24]
results = {
    'classes': class_names,
    'n_train': len(train_df), 'n_val': len(val_df), 'n_test': len(test_df),
    'n_params': n_params,
    'model_size_mb_fp32': n_params * 4 / 1e6,
    'test_accuracy': acc,
    'macro_precision': prec, 'macro_recall': rec, 'macro_f1': f1,
    'confusion_matrix': cm.tolist(),
    'latency_ms_gpu_mean': lat_mean if device.type == 'cuda' else None,
    'latency_ms_gpu_p95': lat_p95 if device.type == 'cuda' else None,
    'latency_ms_cpu_mean': lat_cpu_mean,
    'latency_ms_cpu_p95': lat_cpu_p95,
    'epochs_run': len(history),
    'history': history,
}

with open('thermal_fault_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Saved real results to thermal_fault_results.json')
print(json.dumps({k: v for k, v in results.items() if k != 'history'}, indent=2))

from google.colab import files as colab_files
colab_files.download('thermal_fault_results.json')
colab_files.download('confusion_matrix.png')
