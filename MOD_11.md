# Module 11: Image Data and Computer Vision Research

## TORA CS 336: PyTorch for Research

**Institution:** The Open Research Academy (TORA)

**Instructor:** Alpha Alimamy Kamara

**Module Length:** 2 weeks (4 lectures + 2 lab sessions)

**Prerequisites:** Modules 1–10 completed; familiarity with `nn.Module`, training loops, data pipelines, and regression research; access to an image dataset (public or custom).

---

## 11.0 Module Overview

In Module 10, we studied tabular data — structured rows and columns with heterogeneous features. Now we turn to **image data**, the modality that launched the deep learning revolution. Images are dense, homogeneous, and spatially structured: every pixel is a number, and the arrangement of pixels carries meaning. This structure is what makes convolutional neural networks (CNNs) so effective, and it is what makes image research both accessible and rich.

This module is about **image data and computer vision research**. We will begin with the mathematics of images as tensors, then develop the image data pipeline, then study convolutional neural networks, then work through a complete image classification case study. Along the way, we will encounter the design patterns that underlie every vision model, from the simplest CNN to the largest vision transformer.

The pedagogical approach remains **mathematics first, code second**. For every concept, we will write the mathematical object explicitly — the image tensor, the convolution operation, the pooling operation — and only then translate it into PyTorch. This mirrors how research is conducted: you have a mathematical model of the image in mind, and PyTorch is the instrument that realizes it.

By the end of this module, you will be able to load and preprocess image data; implement convolutional neural networks in PyTorch; apply data augmentation; evaluate classification models with appropriate metrics; visualize model predictions; and articulate why image research is a core domain of deep learning.

The module is self-paced. Work through the mathematics carefully before running the code. The code will make much more sense if you understand what it is computing.

---

## 11.1 Images as Tensors

### 11.1.1 The Mathematical Representation of an Image

A grayscale image is a function $ I: \Omega \to \mathbb{R} $, where $ \Omega \subset \mathbb{R}^2 $ is the image domain. In discrete form, a grayscale image of height $ H $ and width $ W $ is a matrix:

$$
I \in \mathbb{R}^{H \times W}
$$

where $ I_{ij} $ is the intensity at pixel $ (i, j) $, typically in the range $ [0, 255] $ for 8-bit images or $ [0, 1] $ for normalized images.

A color image has three channels: red, green, and blue. It is a 3-tensor:

$$
I \in \mathbb{R}^{3 \times H \times W}
$$

where $ I_{c,i,j} $ is the intensity of channel $ c $ at pixel $ (i, j) $.

A batch of $ N $ color images is a 4-tensor:

$$
\mathcal{X} \in \mathbb{R}^{N \times 3 \times H \times W}
$$

This is the standard input format for convolutional neural networks in PyTorch.

### 11.1.2 Channel Conventions

PyTorch uses the **channels-first** convention: $ (N, C, H, W) $. Other libraries (e.g., TensorFlow, Keras) use **channels-last**: $ (N, H, W, C) $. When porting models between libraries, you must transpose the axes.

```python
import torch

# Channels-first (PyTorch)
x_pytorch = torch.randn(32, 3, 224, 224)  # (N, C, H, W)

# Channels-last (TensorFlow)
x_tf = torch.randn(32, 224, 224, 3)  # (N, H, W, C)

# Convert
x_tf_from_pytorch = x_pytorch.permute(0, 2, 3, 1)  # (N, H, W, C)
x_pytorch_from_tf = x_tf.permute(0, 3, 1, 2)  # (N, C, H, W)
```

### 11.1.3 Pixel Value Conventions

Pixel values are typically stored as integers in the range $ [0, 255] $. For neural networks, they are normalized to $ [0, 1] $ or standardized to have mean 0 and std 1.

**Normalization to $ [0, 1] $:**

$$
I_{\text{norm}} = \frac{I}{255}
$$

**Standardization:**

$$
I_{\text{std}} = \frac{I - \mu}{\sigma}
$$

where $ \mu $ and $ \sigma $ are the per-channel mean and std. For ImageNet, the standard values are:

$$
\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]
$$

### 11.1.4 Why Images Are Special

Images differ from tabular data in several important ways:

**Spatial structure.** Nearby pixels are correlated. Convolutional layers exploit this structure by applying the same filter across the image.

**Translation invariance.** An object is the same object regardless of where it appears in the image. Convolutional layers are translation-equivariant: shifting the input shifts the output.

**Hierarchical features.** Low-level features (edges, textures) combine to form mid-level features (parts, objects) and high-level features (scenes, concepts). Deep CNNs learn this hierarchy.

**Large input dimensionality.** A $ 224 \times 224 \times 3 $ image has 150,528 values. A fully connected layer with 1000 units would have 150 million parameters. Convolutional layers share parameters, reducing this dramatically.

### 11.1.5 Summary: Images as Tensors

| Aspect | Grayscale | Color | Batch |
|--------|-----------|-------|-------|
| Shape | $ (H, W) $ | $ (3, H, W) $ | $ (N, 3, H, W) $ |
| PyTorch | `(H, W)` | `(C, H, W)` | `(N, C, H, W)` |
| Range | [0, 255] or [0, 1] | [0, 255] or [0, 1] | Same |
| Normalization | $ I/255 $ | $ I/255 $ | Per-channel |

> **Exercise 11.1:** For each of the following images, write the shape in PyTorch convention and the number of elements.
>
> 1. A grayscale image of size $ 28 \times 28 $.
> 2. A color image of size $ 224 \times 224 $.
> 3. A batch of 32 color images of size $ 32 \times 32 $.
> 4. A batch of 16 grayscale images of size $ 64 \times 64 $.

---

## 11.2 Loading Image Data

### 11.2.1 The Folder Structure

The standard folder structure for image classification is:

```
data/
  train/
    class_1/
      img1.jpg
      img2.jpg
    class_2/
      img1.jpg
      img2.jpg
  val/
    class_1/
      img1.jpg
    class_2/
      img1.jpg
  test/
    class_1/
      img1.jpg
    class_2/
      img1.jpg
```

Each class is a subfolder containing images of that class.

### 11.2.2 `ImageFolder`

PyTorch provides `ImageFolder` for this structure:

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('data/train', transform=transform)
val_dataset = datasets.ImageFolder('data/val', transform=transform)

print(f"Classes: {train_dataset.classes}")
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
```

### 11.2.3 A Custom Dataset

For more complex scenarios (e.g., multi-label classification, segmentation), define a custom Dataset:

```python
from PIL import Image
import os
import torch
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, labels_df, transform=None):
        self.img_dir = img_dir
        self.labels_df = labels_df
        self.transform = transform

    def __getitem__(self, index):
        img_name = self.labels_df.iloc[index, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.labels_df.iloc[index, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.labels_df)
```

### 11.2.4 DataLoaders

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
```

### 11.2.5 Summary: Loading Image Data

| Method | Use Case | Tool |
|--------|----------|------|
| `ImageFolder` | Standard folder structure | `torchvision.datasets` |
| Custom Dataset | Non-standard structure | `Dataset` |
| DataLoader | Batching | `torch.utils.data` |
| Transforms | Preprocessing | `torchvision.transforms` |

> **Exercise 11.2:** Download a small image dataset (e.g., CIFAR-10, Fashion-MNIST, or a custom dataset) and load it with `ImageFolder` or a custom Dataset. Print the number of samples, the number of classes, and the shape of a sample.

---

## 11.3 Transforms and Augmentation

### 11.3.1 The Purpose of Transforms

Transforms are operations applied to images before they are fed to the model. They serve two purposes:

**Preprocessing:** Resize, normalize, and convert to tensors.

**Augmentation:** Randomly perturb images to increase the diversity of the training set.

### 11.3.2 Common Transforms

**Resize:** Resize the image to a fixed size.

$$
I' = \text{Resize}(I, (H', W'))
$$

**CenterCrop:** Crop the center of the image.

**RandomCrop:** Crop a random region of the image.

**RandomHorizontalFlip:** Flip the image horizontally with probability $ p $.

**ColorJitter:** Randomly change brightness, contrast, saturation, and hue.

**Normalize:** Subtract the mean and divide by the std.

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 11.3.3 Why Augmentation Works

Data augmentation is a form of regularization. It encodes prior knowledge about the task:

**Translation invariance:** Objects are the same regardless of position. Random crop and translation encode this.

**Horizontal flip invariance:** Many objects are symmetric. Horizontal flip encodes this.

**Scale invariance:** Objects can appear at different scales. Random resized crop encodes this.

**Color invariance:** Objects are the same under different lighting. Color jitter encodes this.

By training on augmented data, the model learns to be invariant to these transformations, improving generalization.

### 11.3.4 The Mathematics of Augmentation

Let $ \mathcal{T} $ be a set of transformations. The augmented loss is:

$$
\mathcal{L}_{\text{aug}}(\theta) = \frac{1}{N} \sum_{i=1}^N \mathbb{E}_{t \sim \mathcal{T}}[\ell(y_i, f_\theta(t(\mathbf{x}_i)))]
$$

In practice, we approximate the expectation with a single sample per epoch:

$$
\mathcal{L}_{\text{aug}}(\theta) \approx \frac{1}{N} \sum_{i=1}^N \ell(y_i, f_\theta(t_i(\mathbf{x}_i)))
$$

where $ t_i \sim \mathcal{T} $ is sampled independently for each sample.

### 11.3.5 Summary: Transforms

| Transform | Purpose | When to Use |
|-----------|---------|-------------|
| Resize | Fixed size | Always |
| CenterCrop | Validation | Validation |
| RandomCrop | Augmentation | Training |
| RandomHorizontalFlip | Augmentation | Training |
| ColorJitter | Augmentation | Training |
| Normalize | Standardization | Always |
| ToTensor | Convert to tensor | Always |

> **Exercise 11.3:** Create training and validation transform pipelines for a dataset of your choice. Train a model with and without augmentation. Compare the validation accuracy.

---

## 11.4 Convolutional Neural Networks (CNNs)

### 11.4.1 The Convolution Operation

The **convolution** of an image $ I \in \mathbb{R}^{H \times W} $ with a filter $ K \in \mathbb{R}^{k \times k} $ is:

$$
(I * K)_{ij} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I_{i+m, j+n} K_{mn}
$$

The filter slides across the image, computing a weighted sum at each position. The result is a **feature map** $ (I * K) \in \mathbb{R}^{(H-k+1) \times (W-k+1)} $.

For multi-channel inputs, the convolution sums over channels:

$$
(I * K)_{ij} = \sum_{c=0}^{C-1} \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I_{c, i+m, j+n} K_{c, m, n}
$$

### 11.4.2 Padding and Stride

**Padding** adds zeros around the image to control the output size:

$$
H_{\text{out}} = \frac{H_{\text{in}} + 2p - k}{s} + 1
$$

where $ p $ is the padding and $ s $ is the stride.

**Stride** controls how far the filter moves at each step. Stride 1 preserves the spatial resolution; stride 2 halves it.

```python
import torch.nn as nn

# Convolution with padding=1, stride=1
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1)

# Convolution with padding=0, stride=2
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=0, stride=2)
```

### 11.4.3 Pooling

**Pooling** reduces the spatial resolution of feature maps. Max pooling takes the maximum value in each window:

$$
\text{MaxPool}(I)_{ij} = \max_{m, n} I_{i \cdot s + m, j \cdot s + n}
$$

Average pooling takes the average:

$$
\text{AvgPool}(I)_{ij} = \frac{1}{k^2} \sum_{m, n} I_{i \cdot s + m, j \cdot s + n}
$$

```python
# Max pooling
pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Average pooling
pool = nn.AvgPool2d(kernel_size=2, stride=2)
```

### 11.4.4 The CNN Architecture

A typical CNN for image classification has the following structure:

1. **Convolutional layers:** Extract features.
2. **Activation functions:** Introduce nonlinearity.
3. **Pooling layers:** Reduce spatial resolution.
4. **Fully connected layers:** Classify.

The output of the convolutional layers is a feature map $ F \in \mathbb{R}^{C \times H \times W} $. The fully connected layers flatten this to a vector and map it to class logits.

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=10)
print(model)
```

### 11.4.5 Why Convolutions Work

**Parameter sharing:** The same filter is applied across the image, reducing the number of parameters.

**Translation equivariance:** Shifting the input shifts the output, allowing the model to detect objects regardless of position.

**Local connectivity:** Each output depends only on a local region of the input, exploiting spatial structure.

**Hierarchical features:** Stacking convolutional layers builds a hierarchy from edges to textures to objects.

### 11.4.6 Summary: CNNs

| Component | Mathematical Form | PyTorch |
|-----------|-------------------|---------|
| Convolution | $ (I * K)_{ij} = \sum_{m,n} I_{i+m,j+n} K_{mn} $ | `nn.Conv2d` |
| Padding | Add zeros | `padding=k` |
| Stride | Step size | `stride=s` |
| Max pooling | $ \max_{m,n} I_{i+m,j+n} $ | `nn.MaxPool2d` |
| Avg pooling | $ \frac{1}{k^2} \sum_{m,n} I_{i+m,j+n} $ | `nn.AvgPool2d` |
| Activation | $ \text{ReLU}(x) = \max(0, x) $ | `nn.ReLU` |

> **Exercise 11.4:** Implement a CNN with the following architecture:
>
> 1. Conv2d(3, 32, kernel_size=3, padding=1) + ReLU + MaxPool2d(2)
> 2. Conv2d(32, 64, kernel_size=3, padding=1) + ReLU + MaxPool2d(2)
> 3. Conv2d(64, 128, kernel_size=3, padding=1) + ReLU + MaxPool2d(2)
> 4. Flatten + Linear(128*4*4, 256) + ReLU + Linear(256, 10)
>
> Verify the output shape for an input of shape (32, 3, 32, 32).

---

## 11.5 Case Study: Shape Classification from Generated Images

### 11.5.1 The Problem

We consider a synthetic shape classification problem. The dataset consists of images of geometric shapes (circles, squares, triangles) rendered on a white background. The goal is to classify each image into one of the three shape classes.

This is a controlled setting: the shapes are simple, the backgrounds are clean, and the classes are balanced. It is an ideal starting point for image classification research.

### 11.5.2 The Data

We generate the data synthetically using `PIL` or `matplotlib`:

```python
import numpy as np
from PIL import Image, ImageDraw
import os

def generate_shape(shape, size=64):
    img = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(img)
    margin = 10
    if shape == 'circle':
        draw.ellipse([margin, margin, size-margin, size-margin], fill='blue')
    elif shape == 'square':
        draw.rectangle([margin, margin, size-margin, size-margin], fill='red')
    elif shape == 'triangle':
        draw.polygon([(size//2, margin), (margin, size-margin), (size-margin, size-margin)], fill='green')
    return img

# Generate dataset
os.makedirs('data/train/circle', exist_ok=True)
os.makedirs('data/train/square', exist_ok=True)
os.makedirs('data/train/triangle', exist_ok=True)

for i in range(200):
    for shape in ['circle', 'square', 'triangle']:
        img = generate_shape(shape)
        img.save(f'data/train/{shape}/{i}.png')
```

### 11.5.3 The Mathematical Model

We model the classification problem with a CNN:

$$
\hat{\mathbf{y}} = \text{softmax}(f_\theta(\mathbf{x}))
$$

where $ f_\theta $ is the CNN and $ \hat{\mathbf{y}} \in \mathbb{R}^3 $ is the predicted probability over the three classes.

The loss is cross-entropy:

$$
\ell_{\text{CE}}(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{c=1}^3 y_c \log \hat{y}_c
$$

### 11.5.4 The Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Datasets
train_dataset = datasets.ImageFolder('data/train', transform=train_transform)
val_dataset = datasets.ImageFolder('data/val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Model
class ShapeCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ShapeCNN(num_classes=3).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training
n_epochs = 20
for epoch in range(n_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        yhat = model(x_batch)
        loss = loss_fn(y_batch, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            yhat = model(x_batch)
            _, predicted = yhat.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

    print(f"Epoch {epoch}: val_accuracy={100 * correct / total:.2f}%")
```

### 11.5.5 Evaluation

The model achieves high accuracy on this simple task. We can visualize the predictions:

```python
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    x_batch, y_batch = next(iter(val_loader))
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    yhat = model(x_batch)
    _, predicted = yhat.max(1)

# Visualize
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    img = x_batch[i].cpu().permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)
    ax.imshow(img)
    ax.set_title(f"True: {y_batch[i].item()}, Pred: {predicted[i].item()}")
    ax.axis('off')
plt.show()
```

### 11.5.6 Reflection

The shape classification problem is simple, but it illustrates the full image classification pipeline: data generation, transforms, CNN architecture, training, and evaluation. For your capstone, you will apply the same pipeline to a real dataset with more complex images.

**Research relevance:** The shape dataset is a controlled setting for studying CNN behavior. You can vary the number of classes, the complexity of the shapes, the amount of noise, and the size of the training set to study how CNNs generalize.

> **Exercise 11.5:** Extend the shape classification problem to include:
>
> 1. More shapes (hexagon, star, ellipse).
> 2. Noise (Gaussian noise, salt-and-pepper noise).
> 3. Occlusion (randomly cover part of the shape).
> 4. Rotation (rotate the shape by random angles).
>
> Train a CNN on the extended dataset and report the accuracy.

---

## 11.6 Evaluation: Accuracy, Precision, Recall, F1

### 11.6.1 Classification Metrics

For classification, the standard metrics are:

**Accuracy:**

$$
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{1}[\hat{y}_i = y_i]
$$

**Precision (per class):**

$$
\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}
$$

**Recall (per class):**

$$
\text{Recall}_c = \frac{TP_c}{TP_c + FN_c}
$$

**F1 (per class):**

$$
F1_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}
$$

**Macro F1:** Average of per-class F1.

**Confusion matrix:** A $ C \times C $ matrix where entry $ (i, j) $ is the number of samples with true class $ i $ and predicted class $ j $.

### 11.6.2 Computing Metrics in PyTorch

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Collect predictions
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device)
        yhat = model(x_batch)
        _, predicted = yhat.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.numpy())

# Compute metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
cm = confusion_matrix(all_labels, all_preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")
print(f"Confusion matrix:\n{cm}")
```

### 11.6.3 Why Accuracy Is Not Enough

Accuracy is misleading on imbalanced datasets. If 99% of samples are class 0, a model that always predicts class 0 achieves 99% accuracy but is useless. Precision, recall, and F1 provide a more complete picture.

**Research relevance:** Always report per-class metrics and the confusion matrix, not just overall accuracy.

### 11.6.4 Summary: Classification Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Accuracy | $ \frac{1}{N} \sum \mathbb{1}[\hat{y} = y] $ | Overall correctness |
| Precision | $ \frac{TP}{TP + FP} $ | Fraction of positive predictions correct |
| Recall | $ \frac{TP}{TP + FN} $ | Fraction of positives detected |
| F1 | $ 2 \cdot \frac{P \cdot R}{P + R} $ | Harmonic mean |
| Macro F1 | $ \frac{1}{C} \sum F1_c $ | Average over classes |
| Confusion matrix | $ C \times C $ matrix | Per-class errors |

> **Exercise 11.6:** Compute accuracy, precision, recall, F1, and the confusion matrix for the shape classification model. Which classes are most often confused?

---

## 11.7 Worked Example: Full Image Classification Pipeline

Let us combine everything into a complete pipeline for image classification.

### 11.7.1 The Data

We use CIFAR-10: 60,000 images of size $ 32 \times 32 $, 10 classes.

### 11.7.2 The Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)

# Transforms
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# Model
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN(num_classes=10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# Training
n_epochs = 50
for epoch in range(n_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        yhat = model(x_batch)
        loss = loss_fn(y_batch, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            yhat = model(x_batch)
            _, predicted = yhat.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: test_accuracy={100 * correct / total:.2f}%")
```

### 11.7.3 Results

The model achieves approximately 80–85% accuracy on CIFAR-10, depending on hyperparameters and training time. State-of-the-art models achieve >99%, but they use much larger architectures and longer training.

### 11.7.4 Reflection

This pipeline is the canonical image classification pipeline in PyTorch. It includes data loading, augmentation, a CNN architecture, training, and evaluation. For your capstone, you will modify this pipeline for your specific dataset and task.

**Research relevance:** The choice of architecture, augmentation, and hyperparameters significantly affects performance. Ablation studies can reveal which choices matter most.

> **Exercise 11.7:** Extend the pipeline above to include:
>
> 1. A deeper CNN with 5 convolutional layers.
> 2. A wider CNN with 256 channels.
> 3. Different augmentation strategies (cutout, mixup).
> 4. A comparison with a pre-trained ResNet-18.
> 5. Learning rate scheduling with warmup.

---

## 11.8 Common Errors and Debugging

### 11.8.1 Error: Shape Mismatch

**Symptom:** `RuntimeError: Given groups=1, weight of size [32, 3, 3, 3], expected input[32, 1, 32, 32] to have 3 channels, but got 1 channels instead`

**Cause:** The input has 1 channel (grayscale) but the model expects 3 (RGB).

**Fix:** Convert grayscale to RGB in the transform, or change the first layer to accept 1 channel.

### 11.8.2 Error: CUDA Out of Memory

**Symptom:** `RuntimeError: CUDA out of memory`

**Cause:** The batch is too large for the GPU.

**Fix:** Reduce the batch size, use gradient accumulation, or use mixed precision.

### 11.8.3 Error: Poor Accuracy

**Symptom:** The model performs barely above chance.

**Cause:** Learning rate wrong, insufficient training, or bug in the data pipeline.

**Fix:** Check the learning rate, train longer, visualize the data.

### 11.8.4 Error: Overfitting

**Symptom:** Training accuracy is high but validation accuracy is low.

**Cause:** The model is too large, the dataset is too small, or there is no regularization.

**Fix:** Add dropout, weight decay, or data augmentation. Reduce the model size.

### 11.8.5 Summary: Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| Shape mismatch | Wrong number of channels | Convert or adjust model |
| CUDA OOM | Batch too large | Reduce batch, mixed precision |
| Poor accuracy | Wrong LR, insufficient training | Tune LR, train longer |
| Overfitting | Model too large | Dropout, weight decay, augmentation |

> **Exercise 11.8:** For each of the following scenarios, diagnose the issue and propose a fix:
>
> 1. Training accuracy is 99% but validation accuracy is 50%.
> 2. Loss is NaN after 5 epochs.
> 3. Accuracy is 10% on a 10-class dataset.
> 4. Training is very slow.

---

## 11.9 Research Application: Data Augmentation as Regularization

### 11.9.1 The Research Question

Data augmentation is a form of regularization. But which augmentations are most effective? And how do they interact?

The literature suggests several findings:

**Horizontal flip** is effective for most natural image tasks but not for tasks where orientation matters (e.g., digit recognition).

**Random crop** is effective for most tasks, but the amount of crop matters.

**Color jitter** helps with lighting invariance but can hurt if the task depends on color.

**Cutout** (randomly masking a region) helps with robustness but can hurt if the object is small.

**Mixup** (linearly interpolating between images) helps with calibration and robustness.

### 11.9.2 Designing an Ablation Study

To study augmentation, run the same model with different augmentation strategies:

```python
augmentations = {
    'none': transforms.Compose([transforms.ToTensor(), transforms.Normalize(...)]),
    'flip': transforms.Compose([transforms.RandomHorizontalFlip(), ...]),
    'crop': transforms.Compose([transforms.RandomCrop(32, padding=4), ...]),
    'flip+crop': transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), ...]),
    'flip+crop+jitter': transforms.Compose([...])
}

for name, transform in augmentations.items():
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Train and evaluate
    print(f"{name}: accuracy={accuracy:.4f}")
```

### 11.9.3 Summary: Augmentation as Regularization

| Augmentation | Effect | When to Use |
|--------------|--------|-------------|
| Horizontal flip | Translation invariance | Most tasks |
| Random crop | Translation invariance | Most tasks |
| Color jitter | Lighting invariance | Color-irrelevant tasks |
| Cutout | Robustness | Large objects |
| Mixup | Calibration | Most tasks |

> **Exercise 11.9:** Design and run an ablation study on data augmentation for CIFAR-10. Which augmentations help most? Which hurt?

---

## 11.10 Module Summary

| Concept | Mathematical Form | PyTorch | Research Relevance |
|---------|-------------------|---------|-------------------|
| Image tensor | $ I \in \mathbb{R}^{C \times H \times W} $ | `torch.randn(C, H, W)` | Input representation |
| Batch | $ \mathcal{X} \in \mathbb{R}^{N \times C \times H \times W} $ | `torch.randn(N, C, H, W)` | Mini-batch |
| Normalization | $ (I - \mu) / \sigma $ | `transforms.Normalize` | Standardization |
| Augmentation | $ t \sim \mathcal{T} $ | `transforms` | Regularization |
| Convolution | $ (I * K)_{ij} = \sum I_{i+m,j+n} K_{mn} $ | `nn.Conv2d` | Feature extraction |
| Pooling | $ \max_{m,n} I_{i+m,j+n} $ | `nn.MaxPool2d` | Downsampling |
| CNN | Composition of conv, pool, FC | `nn.Sequential` | Architecture |
| Cross-entropy | $ -\sum y \log \hat{y} $ | `nn.CrossEntropyLoss` | Classification |
| Accuracy | $ \frac{1}{N} \sum \mathbb{1}[\hat{y} = y] $ | `accuracy_score` | Metric |
| Precision | $ TP / (TP + FP) $ | `precision_score` | Metric |
| Recall | $ TP / (TP + FN) $ | `recall_score` | Metric |
| F1 | $ 2PR / (P + R) $ | `f1_score` | Metric |
| Confusion matrix | $ C \times C $ matrix | `confusion_matrix` | Error analysis |

---

## 11.11 Capstone Thread: Image Data for Your Project

By the end of this module, you will be able to load and preprocess image data; implement convolutional neural networks in PyTorch; apply data augmentation; evaluate classification models with appropriate metrics; visualize model predictions; and articulate why image research is a core domain of deep learning.

For your capstone project, you will use these skills to load your image data, preprocess it correctly, implement a CNN, compare against baselines, and report your results.

In Module 12, we will build on this foundation to study **advanced architectures and transfer learning**.

---

## 11.12 Additional Resources

### On CNNs

- LeCun, Y., et al. (1998). "Gradient-Based Learning Applied to Document Recognition." *Proceedings of the IEEE*.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet Classification with Deep Convolutional Neural Networks." *NeurIPS*.
- Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition." *ICLR*.

### On Image Classification

- He, K., et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.
- Huang, G., et al. (2017). "Densely Connected Convolutional Networks." *CVPR*.
- Tan, M., & Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *ICML*.

### On Data Augmentation

- Shorten, C., & Khoshgoftaar, T. M. (2019). "A Survey on Image Data Augmentation for Deep Learning." *Journal of Big Data*.
- Zhang, H., et al. (2018). "mixup: Beyond Empirical Risk Minimization." *ICLR*.
- DeVries, T., & Taylor, G. W. (2017). "Improved Regularization of Convolutional Neural Networks with Cutout." *arXiv*.

### On Evaluation

- Sokolova, M., & Lapalme, G. (2009). "A Systematic Analysis of Performance Measures for Classification Tasks." *Information Processing & Management*.
- Grandini, M., et al. (2020). "Metrics for Multi-Class Classification: An Overview." *arXiv*.

### On Datasets

- CIFAR-10/100: https://www.cs.toronto.edu/~kriz/cifar.html
- ImageNet: https://www.image-net.org/
- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
- COCO: https://cocodataset.org/

---

*End of Module 11*