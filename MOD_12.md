# Module 12: Advanced Architectures and Transfer Learning

## TORA CS 336: PyTorch for Research

**Institution:** The Open Research Academy (TORA)

**Instructor:** Alpha Alimamy Kamara

**Module Length:** 2 weeks (4 lectures + 2 lab sessions)

**Prerequisites:** Modules 1–11 completed; familiarity with CNNs, image data pipelines, and training loops; access to a GPU (recommended) and an image dataset.

---

## 12.0 Module Overview

In Module 11, we built convolutional neural networks from scratch and trained them on image classification tasks. But training a CNN from scratch requires large datasets and significant compute. In practice, most research begins with a **pre-trained model** and adapts it to a new task. This is **transfer learning**, and it is one of the most important techniques in modern deep learning research.

This module is about **advanced architectures and transfer learning**. We will begin with the mathematics of modern architectures — ResNet, EfficientNet, and Vision Transformers — then develop the theory and practice of transfer learning, then work through a complete case study on fine-tuning a pre-trained model. Along the way, we will encounter the design patterns that underlie every modern vision model and the decision frameworks that guide transfer learning strategy.

The pedagogical approach remains **mathematics first, code second**. For every architecture, we will write the mathematical object explicitly — the residual connection, the attention mechanism, the scaling rule — and only then translate it into PyTorch. This mirrors how research is conducted: you have a mathematical model of the architecture in mind, and PyTorch is the instrument that realizes it.

By the end of this module, you will be able to load and use pre-trained models; implement transfer learning strategies; fine-tune models on new datasets; compare architectures; evaluate transfer learning performance; and articulate when transfer learning is preferable to training from scratch.

The module is self-paced. Work through the mathematics carefully before running the code. The code will make much more sense if you understand what it is computing.

---

## 12.1 The Mathematics of Modern Architectures

### 12.1.1 The Depth Problem

Before 2015, the standard approach to improving CNNs was to make them deeper. VGG-16 had 16 layers; VGG-19 had 19. But deeper networks were harder to train. The training error, not just the validation error, increased with depth. This was not overfitting — it was an optimization problem.

The **degradation problem** can be stated mathematically. Consider a shallow network $ f_s $ and a deep network $ f_d $ that contains $ f_s $ as a subnetwork. In principle, $ f_d $ should be at least as good as $ f_s $, because it can learn the identity mapping for the extra layers. But in practice, $ f_d $ performs worse. The optimization landscape of deep networks is harder to navigate.

### 12.1.2 Residual Connections

The **residual connection** (He et al., 2016) 
solves the degradation problem by reformulating 
the layers as learning residual functions. 
Instead of learning $ \mathcal{H}(\mathbf{x}) $, the layer learns:

$$
\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}
$$

and the output is:

$$
\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}
$$

The identity shortcut $ \mathbf{x} $ allows gradients to flow directly through the network, mitigating the vanishing gradient problem. Mathematically, the gradient of the loss with respect to the input of a residual block is:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \left( 1 + \frac{\partial \mathcal{F}}{\partial \mathbf{x}} \right)
$$

The "1" term ensures that gradients do not vanish, even if $ \frac{\partial \mathcal{F}}{\partial \mathbf{x}} $ is small.

### 12.1.3 The ResNet Architecture

The ResNet architecture consists of a stem (initial convolution and pooling), followed by four stages of residual blocks, followed by a classification head. Each stage doubles the number of channels and halves the spatial resolution.

The **basic block** has two $ 3 \times 3 $ convolutions:

$$
\mathbf{y} = \text{ReLU}(\text{BN}(W_2 \cdot \text{ReLU}(\text{BN}(W_1 \mathbf{x}))) + \mathbf{x})
$$

The **bottleneck block** has three convolutions: $ 1 \times 1 $ (reduce), $ 3 \times 3 $ (process), $ 1 \times 1 $ (expand):

$$
\mathbf{y} = \text{ReLU}(\text{BN}(W_3 \cdot \text{ReLU}(\text{BN}(W_2 \cdot \text{ReLU}(\text{BN}(W_1 \mathbf{x}))))) + \mathbf{x})
$$

```python
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
```

### 12.1.4 EfficientNet and Compound Scaling

**EfficientNet** (Tan & Le, 2019) introduced **compound scaling**: instead of scaling depth, width, or resolution independently, scale all three together with a fixed ratio:

$$
\text{depth} = \alpha^\phi, \quad \text{width} = \beta^\phi, \quad \text{resolution} = \gamma^\phi
$$

subject to $ \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2 $. The base model (EfficientNet-B0) was found via neural architecture search; the scaled models (B1–B7) use compound scaling.

### 12.1.5 Vision Transformers (ViTs)

The **Vision Transformer** (Dosovitskiy et al., 2021) applies the Transformer architecture to images. An image is split into patches, each patch is flattened and projected to an embedding, and the sequence of embeddings is processed by a standard Transformer encoder.

Mathematically, for an image $ \mathbf{X} \in \mathbb{R}^{H \times W \times C} $ and patch size $ P $, the number of patches is:

$$
N = \frac{H \cdot W}{P^2}
$$

Each patch $ \mathbf{x}_p \in \mathbb{R}^{P^2 \cdot C} $ is projected to a $ D $-dimensional embedding:

$$
\mathbf{z}_p = W_E \mathbf{x}_p + \mathbf{b}_E
$$

A special [CLS] token is prepended, and positional embeddings are added:

$$
\mathbf{Z} = [\mathbf{z}_{\text{cls}}; \mathbf{z}_1; \ldots; \mathbf{z}_N] + \mathbf{E}_{\text{pos}}
$$

The sequence is processed by $ L $ Transformer encoder layers, each with multi-head self-attention and an MLP. The [CLS] token's output is used for classification.

The **self-attention** mechanism computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

where $ Q = \mathbf{Z}W_Q $, $ K = \mathbf{Z}W_K $, $ V = \mathbf{Z}W_V $.

```python
class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x[:, 0]  # CLS token
        x = self.head(x)
        return x
```

### 12.1.6 Summary: Modern Architectures

| Architecture | Key Innovation | Mathematical Form | When to Use |
|--------------|----------------|-------------------|-------------|
| VGG | Depth with 3×3 convs | Stacked conv layers | Historical baseline |
| ResNet | Residual connections | $ \mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x} $ | Default for vision |
| DenseNet | Dense connections | $ \mathbf{y} = \text{concat}(\mathbf{x}, \mathcal{F}(\mathbf{x})) $ | Parameter-efficient |
| EfficientNet | Compound scaling | $ \alpha^\phi, \beta^\phi, \gamma^\phi $ | Efficient inference |
| ViT | Self-attention on patches | $ \text{softmax}(QK^T/\sqrt{d})V $ | Large datasets |

> **Exercise 12.1:** For each of the following architectures, write the mathematical form of the key innovation and the PyTorch implementation.
>
> 1. Basic residual block
> 2. Bottleneck residual block
> 3. Multi-head self-attention
> 4. Transformer encoder layer

---

## 12.2 Transfer Learning: Theory and Practice

### 12.2.1 The Transfer Learning Problem

**Transfer learning** is the process of using knowledge from a source task to improve performance on a target task. Formally, given a source domain $ \mathcal{D}_S $ and task $ \mathcal{T}_S $, and a target domain $ \mathcal{D}_T $ and task $ \mathcal{T}_T $, transfer learning aims to improve the target predictive function $ f_T $ using knowledge from $ \mathcal{D}_S $ and $ \mathcal{T}_S $.

In computer vision, the source domain is typically ImageNet (1.2M images, 1000 classes), and the target domain is your dataset (e.g., medical images, satellite images, or a custom dataset).

### 12.2.2 Why Transfer Learning Works

Transfer learning works because the early layers of a CNN learn **general features** (edges, textures, colors) that are useful across many tasks. The later layers learn **task-specific features** (object parts, classes) that are specific to the source task.

Mathematically, the feature extractor $ f_{\text{feat}} $ learned on the source task maps inputs to a representation space $ \mathcal{Z} $:

$$
\mathbf{z} = f_{\text{feat}}(\mathbf{x})
$$

If the source and target tasks share similar low-level features, then $ \mathcal{Z} $ is a good representation for the target task, and only the classifier $ f_{\text{cls}}: \mathcal{Z} \to \mathcal{Y} $ needs to be relearned.

### 12.2.3 Transfer Learning Strategies

There are three main strategies:

**Feature extraction:** Freeze the pre-trained feature extractor, train only the new classifier.

$$
\hat{\mathbf{y}} = f_{\text{cls}}(f_{\text{feat}}(\mathbf{x}))
$$

**Fine-tuning:** Unfreeze some or all of the pre-trained layers, train them with a small learning rate.

$$
\hat{\mathbf{y}} = f_{\text{cls}}(f_{\text{feat}}(\mathbf{x}; \theta_{\text{feat}}))
$$

**Top-tuning:** Freeze the feature extractor, train a new classifier on top, then unfreeze and fine-tune the whole model.

**Discriminative fine-tuning:** Use different learning rates for different layers. Early layers get smaller learning rates (they are more general); later layers get larger learning rates (they are more task-specific).

### 12.2.4 When to Use Which Strategy

The choice of strategy depends on:

**Dataset size:** Small datasets favor feature extraction; large datasets favor fine-tuning.

**Domain similarity:** Similar domains favor fine-tuning; dissimilar domains favor feature extraction.

**Compute budget:** Feature extraction is cheaper; fine-tuning is more expensive.

The following table summarizes the standard recommendations:

| Dataset Size | Domain Similarity | Strategy |
|--------------|-------------------|----------|
| Small | High | Feature extraction |
| Small | Low | Feature extraction (with caution) |
| Large | High | Fine-tuning |
| Large | Low | Fine-tuning (from scratch if very different) |

### 12.2.5 The Mathematics of Fine-Tuning

Fine-tuning minimizes the target loss:

$$
\mathcal{L}_T(\theta) = \frac{1}{N_T} \sum_{i=1}^{N_T} \ell(y_i^T, f_\theta(\mathbf{x}_i^T))
$$

starting from the pre-trained parameters $ \theta_0 $. The update is:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}_T(\theta_t)
$$

For discriminative fine-tuning, the learning rate is layer-specific:

$$
\theta_{t+1}^{(l)} = \theta_t^{(l)} - \eta_l \nabla_{\theta^{(l)}} \mathcal{L}_T(\theta_t)
$$

where $ \eta_l $ is the learning rate for layer $ l $.

### 12.2.6 Summary: Transfer Learning Strategies

| Strategy | Frozen Layers | Trainable Layers | When to Use |
|----------|---------------|------------------|-------------|
| Feature extraction | All except classifier | Classifier | Small dataset, similar domain |
| Fine-tuning | None | All | Large dataset, similar domain |
| Partial fine-tuning | Early layers | Later layers + classifier | Medium dataset |
| Discriminative fine-tuning | None (different LRs) | All | Large dataset, mixed similarity |

> **Exercise 12.2:** For each of the following scenarios, recommend a transfer learning strategy and justify your choice.
>
> 1. 500 medical images of a rare disease, similar to ImageNet.
> 2. 50,000 satellite images, very different from ImageNet.
> 3. 10,000 images of everyday objects, similar to ImageNet.
> 4. 1,000 images of handwritten digits, different from ImageNet.

---

## 12.3 Loading Pre-Trained Models

### 12.3.1 `torchvision.models`

PyTorch provides a wide range of pre-trained models in `torchvision.models`:

```python
import torchvision.models as models

# ResNet
resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)

# EfficientNet
efficientnet_b0 = models.efficientnet_b0(pretrained=True)

# Vision Transformer
vit_b_16 = models.vit_b_16(pretrained=True)

# VGG
vgg16 = models.vgg16(pretrained=True)
```

### 12.3.2 Inspecting the Model

```python
print(resnet18)
```

The output shows the architecture: a stem (conv + bn + relu + maxpool), four layers of residual blocks, an adaptive avg pool, and a fully connected classifier.

### 12.3.3 Modifying the Classifier

To adapt a pre-trained model to a new task, replace the final classifier:

```python
import torch.nn as nn

# Replace the classifier for a 10-class task
num_classes = 10
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)

print(resnet18.fc)
```

### 12.3.4 Freezing Layers

To freeze the feature extractor:

```python
# Freeze all layers
for param in resnet18.parameters():
    param.requires_grad = False

# Unfreeze the classifier
for param in resnet18.fc.parameters():
    param.requires_grad = True
```

Or freeze specific layers:

```python
# Freeze the first two layers
for name, param in resnet18.named_parameters():
    if 'layer1' in name or 'layer2' in name:
        param.requires_grad = False
```

### 12.3.5 Summary: Loading Pre-Trained Models

| Model | Parameters | Top-1 Accuracy | When to Use |
|-------|------------|----------------|-------------|
| ResNet-18 | 11.7M | 69.8% | Fast baseline |
| ResNet-50 | 25.6M | 76.1% | Standard |
| EfficientNet-B0 | 5.3M | 77.7% | Efficient |
| ViT-B/16 | 86M | 81.1% | Large datasets |
| VGG-16 | 138M | 71.6% | Historical |

> **Exercise 12.3:** Load ResNet-18, ResNet-50, and EfficientNet-B0. For each, print the number of parameters, the architecture summary, and the shape of the final classifier. Replace the classifier with a 5-class head.

---

## 12.4 Fine-Tuning a Pre-Trained Model

### 12.4.1 The Fine-Tuning Pipeline

The fine-tuning pipeline consists of:

1. **Load** the pre-trained model.
2. **Replace** the classifier.
3. **Freeze** the feature extractor (optional).
4. **Train** the classifier.
5. **Unfreeze** the feature extractor (optional).
6. **Fine-tune** the whole model with a small learning rate.

### 12.4.2 Worked Example 12.1: Fine-Tuning ResNet-18 on CIFAR-10

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)

# Transforms
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# Freeze feature extractor
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

# Phase 1: Train classifier
print("Phase 1: Training classifier")
for epoch in range(5):
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
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            yhat = model(x_batch)
            _, predicted = yhat.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
    print(f"Epoch {epoch}: test_accuracy={100 * correct / total:.2f}%")

# Phase 2: Fine-tune the whole model
print("\nPhase 2: Fine-tuning")
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
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
    print(f"Epoch {epoch}: test_accuracy={100 * correct / total:.2f}%")
```

### 12.4.3 Results

The model achieves approximately 90–95% accuracy on CIFAR-10, significantly better than training from scratch (80–85%). The two-phase approach — first train the classifier, then fine-tune the whole model — is a standard recipe.

### 12.4.4 Reflection

Fine-tuning is one of the most powerful techniques in deep learning research. It allows you to leverage pre-trained models to achieve strong performance on small datasets. The key decisions are: which model to use, which layers to freeze, and what learning rate to use.

**Research relevance:** Transfer learning is the default approach in most applied research. Understanding when and how to fine-tune is essential.

> **Exercise 12.4:** Fine-tune ResNet-18 on a dataset of your choice. Compare:
>
> 1. Feature extraction (frozen feature extractor)
> 2. Fine-tuning (all layers)
> 3. Discriminative fine-tuning (different LRs)
>
> Report the accuracy for each strategy.

---

## 12.5 Case Study: Fine-Tuning a Pre-Trained Model on a Custom Dataset

### 12.5.1 The Problem

We consider a custom dataset of flower images (102 classes, ~8,000 images). The goal is to classify each image into one of the 102 flower species.

This is a fine-grained classification problem: the classes are visually similar, and the dataset is small. Transfer learning is essential.

### 12.5.2 The Data

The Oxford Flowers-102 dataset is available in `torchvision.datasets`:

```python
from torchvision.datasets import Flowers102

train_dataset = Flowers102(root='./data', split='train', download=True, transform=train_transform)
val_dataset = Flowers102(root='./data', split='val', download=True, transform=test_transform)
test_dataset = Flowers102(root='./data', split='test', download=True, transform=test_transform)
```

### 12.5.3 The Model

We use ResNet-50 pre-trained on ImageNet:

```python
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 102)
model = model.to(device)
```

### 12.5.4 Training

```python
# Phase 1: Feature extraction
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
# Train for 5 epochs

# Phase 2: Fine-tuning
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
# Train for 20 epochs
```

### 12.5.5 Results

The model achieves approximately 95% accuracy on the test set, far better than training from scratch (< 50%). This demonstrates the power of transfer learning on small, fine-grained datasets.

### 12.5.6 Reflection

The flower dataset is a classic fine-grained classification problem. Transfer learning is essential because the dataset is small and the classes are visually similar. The pre-trained model provides a strong feature extractor that generalizes across domains.

**Research relevance:** Fine-grained classification is an active area of research. The choice of architecture, augmentation, and fine-tuning strategy significantly affects performance.

> **Exercise 12.5:** Fine-tune a pre-trained model on a fine-grained dataset (e.g., Flowers-102, Stanford Cars, or Food-101). Report the accuracy. Compare with training from scratch.

---

## 12.6 Domain Adaptation

### 12.6.1 The Domain Shift Problem

**Domain shift** occurs when the training distribution $ p_{\text{train}}(\mathbf{x}, \mathbf{y}) $ differs from the test distribution $ p_{\text{test}}(\mathbf{x}, \mathbf{y}) $. This is common in real-world applications: a model trained on images from one hospital may not generalize to another hospital; a model trained on daytime images may not generalize to nighttime images.

**Domain adaptation** is the process of adapting a model trained on a source domain to a target domain.

### 12.6.2 Types of Domain Shift

**Covariate shift:** $ p_{\text{train}}(\mathbf{x}) \neq p_{\text{test}}(\mathbf{x}) $ but $ p(\mathbf{y} \mid \mathbf{x}) $ is the same. Example: different lighting conditions.

**Label shift:** $ p_{\text{train}}(\mathbf{y}) \neq p_{\text{test}}(\mathbf{y}) $ but $ p(\mathbf{x} \mid \mathbf{y}) $ is the same. Example: different class proportions.

**Concept shift:** $ p_{\text{train}}(\mathbf{y} \mid \mathbf{x}) \neq p_{\text{test}}(\mathbf{y} \mid \mathbf{x}) $. Example: different labeling conventions.

### 12.6.3 Domain Adaptation Strategies

**Fine-tuning:** Fine-tune the model on a small labeled target dataset.

**Domain adversarial training:** Train a feature extractor that produces features that are indistinguishable between source and target domains.

**Self-supervised adaptation:** Use unlabeled target data to adapt the model via self-supervised tasks.

**Test-time adaptation:** Adapt the model at test time using the test data.

### 12.6.4 Summary: Domain Adaptation

| Type | Definition | Strategy |
|------|------------|----------|
| Covariate shift | $ p(\mathbf{x}) $ differs | Importance weighting, fine-tuning |
| Label shift | $ p(\mathbf{y}) $ differs | Class reweighting |
| Concept shift | $ p(\mathbf{y} \mid \mathbf{x}) $ differs | Re-labeling, fine-tuning |

> **Exercise 12.6:** Simulate domain shift by training a model on one dataset and testing on another (e.g., train on CIFAR-10, test on CIFAR-10-C with noise). Report the accuracy drop. Propose a domain adaptation strategy.

---

## 12.7 Worked Example: Full Transfer Learning Pipeline

Let us combine everything into a complete pipeline for transfer learning.

### 12.7.1 The Data

We use the Oxford Flowers-102 dataset: 102 classes, ~8,000 images.

### 12.7.2 The Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)

# Transforms
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets
train_dataset = datasets.Flowers102(root='./data', split='train', download=True, transform=train_transform)
val_dataset = datasets.Flowers102(root='./data', split='val', download=True, transform=test_transform)
test_dataset = datasets.Flowers102(root='./data', split='test', download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 102)
model = model.to(device)

# Phase 1: Feature extraction
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

print("Phase 1: Feature extraction")
for epoch in range(5):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        yhat = model(x_batch)
        loss = loss_fn(y_batch, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

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

# Phase 2: Fine-tuning
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

print("\nPhase 2: Fine-tuning")
for epoch in range(20):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        yhat = model(x_batch)
        loss = loss_fn(y_batch, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()

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

# Test
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
print(f"\nTest accuracy: {100 * correct / total:.2f}%")
```

### 12.7.3 Results

The model achieves approximately 95% accuracy on the test set. The two-phase approach — feature extraction, then fine-tuning — is standard and effective.

### 12.7.4 Reflection

This pipeline is the canonical transfer learning pipeline in PyTorch. It includes data loading, augmentation, a pre-trained model, a two-phase training strategy, and evaluation. For your capstone, you will modify this pipeline for your specific dataset and task.

**Research relevance:** Transfer learning is the default approach in most applied research. The choice of pre-trained model, the number of frozen layers, and the learning rate schedule all affect performance.

> **Exercise 12.7:** Extend the pipeline above to include:
>
> 1. A comparison of ResNet-18, ResNet-50, and EfficientNet-B0.
> 2. A comparison of feature extraction and fine-tuning.
> 3. Discriminative fine-tuning (different LRs for different layers).
> 4. Test-time augmentation.
> 5. Model ensembling.

---

## 12.8 Common Errors and Debugging

### 12.8.1 Error: Shape Mismatch

**Symptom:** `RuntimeError: size mismatch for fc.weight`

**Cause:** The pre-trained model's classifier has a different number of classes than your task.

**Fix:** Replace the classifier with a new `nn.Linear` layer matching your number of classes.

### 12.8.2 Error: Slow Training

**Symptom:** Training is very slow.

**Cause:** The model is large, the batch size is small, or the data loading is slow.

**Fix:** Use a smaller model, increase the batch size, or use more workers.

### 12.8.3 Error: Overfitting

**Symptom:** Training accuracy is high but validation accuracy is low.

**Cause:** The model is too large for the dataset, or there is no regularization.

**Fix:** Freeze more layers, add dropout, add weight decay, or use more augmentation.

### 12.8.4 Error: Poor Performance

**Symptom:** The model performs barely above chance.

**Cause:** Learning rate wrong, insufficient training, or bug in the data pipeline.

**Fix:** Check the learning rate, train longer, visualize the data.

### 12.8.5 Summary: Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| Shape mismatch | Wrong number of classes | Replace classifier |
| Slow training | Large model, small batch | Smaller model, larger batch |
| Overfitting | Model too large | Freeze layers, regularize |
| Poor performance | Wrong LR, insufficient training | Tune LR, train longer |

> **Exercise 12.8:** For each of the following scenarios, diagnose the issue and propose a fix:
>
> 1. Fine-tuning a ResNet-50 on a 100-image dataset gives 100% training accuracy but 10% validation accuracy.
> 2. Fine-tuning is very slow despite using a GPU.
> 3. Fine-tuning gives 1% accuracy on a 100-class dataset.
> 4. Fine-tuning gives 99% training accuracy but 50% validation accuracy.

---

## 12.9 Research Application: How Much Does Domain Similarity Matter?

### 12.9.1 The Research Question

Transfer learning assumes that the source and target domains are similar. But how similar is similar enough? And what happens when they are not?

The literature suggests several findings:

**Low-level features transfer well:** Edges, textures, and colors are general across domains.

**High-level features transfer poorly:** Object parts and classes are specific to the source task.

**Fine-tuning helps more when domains are dissimilar:** When domains are similar, feature extraction is sufficient; when they are dissimilar, fine-tuning is necessary.

**More data helps more when domains are dissimilar:** With more target data, the model can learn domain-specific features.

### 12.9.2 Designing a Study

To study domain similarity, select datasets with varying degrees of similarity to ImageNet:

| Dataset | Similarity to ImageNet |
|---------|-----------------------|
| CIFAR-10 | Medium |
| Flowers-102 | Medium |
| Medical images | Low |
| Satellite images | Low |
| Sketches | Low |

Train the same model on each with feature extraction and fine-tuning. Report the accuracy and the gap between the two strategies.

### 12.9.3 Summary: Domain Similarity

| Factor | Effect on Transfer |
|--------|-------------------|
| High similarity | Feature extraction sufficient |
| Low similarity | Fine-tuning necessary |
| More data | Fine-tuning more effective |
| Less data | Feature extraction more effective |

> **Exercise 12.9:** Design a study to investigate the effect of domain similarity on transfer learning. Select 3 datasets, train with feature extraction and fine-tuning, and report your findings.

---

## 12.10 Module Summary

| Concept | Mathematical Form | PyTorch | Research Relevance |
|---------|-------------------|---------|-------------------|
| Residual connection | $ \mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x} $ | `out += shortcut` | Deep networks |
| Self-attention | $ \text{softmax}(QK^T/\sqrt{d})V $ | `nn.MultiheadAttention` | Transformers |
| Compound scaling | $ \alpha^\phi, \beta^\phi, \gamma^\phi $ | `efficientnet_b0` | Efficient models |
| Transfer learning | $ f_{\text{cls}}(f_{\text{feat}}(\mathbf{x})) $ | `pretrained=True` | Applied research |
| Feature extraction | Frozen feature extractor | `requires_grad=False` | Small datasets |
| Fine-tuning | Unfrozen feature extractor | `requires_grad=True` | Large datasets |
| Discriminative fine-tuning | Different LRs per layer | Parameter groups | Mixed similarity |
| Domain adaptation | — | Fine-tuning, adversarial | Domain shift |
| Model comparison | — | — | Research |

---

## 12.11 Capstone Thread: Advanced Architectures for Your Project

By the end of this module, you will be able to load and use pre-trained models; implement transfer learning strategies; fine-tune models on new datasets; compare architectures; evaluate transfer learning performance; and articulate when transfer learning is preferable to training from scratch.

For your capstone project, you will use these skills to select a pre-trained model, adapt it to your task, fine-tune it, and report your results.

In Module 13, we will shift to the **capstone project guidelines**.

---

## 12.12 Additional Resources

### On Modern Architectures

- He, K., et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.
- Tan, M., & Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *ICML*.
- Dosovitskiy, A., et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR*.

### On Transfer Learning

- Pan, S. J., & Yang, Q. (2010). "A Survey on Transfer Learning." *IEEE TKDE*.
- Zhuang, F., et al. (2021). "A Comprehensive Survey on Transfer Learning." *Proceedings of the IEEE*.
- Yosinski, J., et al. (2014). "How Transferable Are Features in Deep Neural Networks?" *NeurIPS*.

### On Domain Adaptation

- Wang, M., & Deng, W. (2018). "Deep Visual Domain Adaptation: A Survey." *Neurocomputing*.
- Ganin, Y., et al. (2016). "Domain-Adversarial Training of Neural Networks." *JMLR*.

### On Fine-Tuning

- Kornblith, S., et al. (2019). "Do Better ImageNet Models Transfer Better?" *CVPR*.
- Zhai, X., et al. (2019). "A Large-Scale Study of Transfer Learning." *arXiv*.

### On Datasets

- Oxford Flowers-102: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
- Stanford Cars: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
- Food-101: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
- DomainNet: http://ai.bu.edu/M3SDA/

---

*End of Module 12*