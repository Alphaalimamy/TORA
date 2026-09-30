# Module 4: Building Models from First Principles

## TORA | CS 336: PyTorch for Research

**Instructor:** Alpha Alimamy Kamara

**Module Length:** 2 weeks (4 lectures + 2 lab sessions)

**Prerequisites:** Modules 1–3 completed; familiarity with autograd, computation graphs, and gradient descent; basic understanding of neural network architecture.

---

## 4.0 Module Overview

In Module 3, we learned how autograd computes gradients. We saw that a training loop consists of a forward pass, a backward pass, and a parameter update. But we did not yet examine *what* is being trained. That is, we did not define a **model**.

This module is about `nn.Module`, the base class for all PyTorch models. We will begin with the mathematics of a linear model, then build it from scratch using `nn.Parameter`, then refactor it using `nn.Module`, and finally compose it using `nn.Linear` and `nn.Sequential`. Along the way, we will encounter the design patterns that underlie every PyTorch model, from the simplest linear regression to the largest transformer.

The pedagogical approach remains **mathematics first, code second**. For every model, we will write the mathematical function explicitly — the linear map, the activation, the composition — and only then translate it into PyTorch. This mirrors how research is conducted: you have a mathematical model in mind, and `nn.Module` is the scaffold that realizes it.

By the end of this module, you will be able to define custom models with `nn.Module`; register learnable parameters with `nn.Parameter`; implement the `forward()` method; compose layers with `nn.Linear` and `nn.Sequential`; inspect model state with `state_dict()`; and articulate why modular model design matters for research reproducibility and extensibility.

The module is self-paced. Work through the mathematics carefully before running the code. The code will make much more sense if you understand what it is computing.

---

## 4.1 The Mathematics of a Linear Model

### 4.1.1 The Simple Linear Regression Model

We return to the linear regression problem from Modules 2 and 3. The model is:

$$
\hat{y}_i = a + b x_i
$$

where $ a \in \mathbb{R} $ is the intercept, $ b \in \mathbb{R} $ is the slope, and $ x_i \in \mathbb{R} $ is the input feature.

In vector form, for a batch of $ N $ inputs $ \mathbf{x} = [x_1, \ldots, x_N]^T \in \mathbb{R}^N $, the model produces:

$$
\hat{\mathbf{y}} = a \mathbf{1} + b \mathbf{x}
$$

where $ \mathbf{1} \in \mathbb{R}^N $ is the vector of all ones.

### 4.1.2 The General Linear Model

For a multi-dimensional input $ \mathbf{x} \in \mathbb{R}^d $, the linear model becomes:

$$
\hat{y} = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^d w_j x_j + b
$$

where $ \mathbf{w} \in \mathbb{R}^d $ is the weight vector and $ b \in \mathbb{R} $ is the bias.

For a batch of $ N $ inputs $ X \in \mathbb{R}^{N \times d} $, the model produces:

$$
\hat{\mathbf{y}} = X \mathbf{w} + b \mathbf{1}
$$

where $ \hat{\mathbf{y}} \in \mathbb{R}^N $.

### 4.1.3 The Linear Layer

A **linear layer** (also called a fully connected layer or dense layer) maps an input $ \mathbf{x} \in \mathbb{R}^{d_{\text{in}}} $ to an output $ \mathbf{y} \in \mathbb{R}^{d_{\text{out}}} $:

$$
\mathbf{y} = W \mathbf{x} + \mathbf{b}
$$

where $ W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}} $ is the weight matrix and $ \mathbf{b} \in \mathbb{R}^{d_{\text{out}}} $ is the bias vector.

For a batch of $ N $ inputs $ X \in \mathbb{R}^{N \times d_{\text{in}}} $, the layer computes:

$$
Y = X W^T + \mathbf{b}
$$

where $ Y \in \mathbb{R}^{N \times d_{\text{out}}} $. The transpose $ W^T $ appears because PyTorch stores weights as $ (d_{\text{out}}, d_{\text{in}}) $ and uses the convention $ Y = X W^T + b $.

**Worked example 4.1:** Consider a linear layer with $ d_{\text{in}} = 3 $ and $ d_{\text{out}} = 2 $. The weight matrix and bias are:

$$
W = \begin{bmatrix} 0.1 & 0.2 & 0.3 \\ 0.4 & 0.5 & 0.6 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 0.7 \\ 0.8 \end{bmatrix}
$$

For the input $ \mathbf{x} = [1, 2, 3]^T $, the output is:

$$
\mathbf{y} = W \mathbf{x} + \mathbf{b} = \begin{bmatrix} 0.1 \cdot 1 + 0.2 \cdot 2 + 0.3 \cdot 3 \\ 0.4 \cdot 1 + 0.5 \cdot 2 + 0.6 \cdot 3 \end{bmatrix} + \begin{bmatrix} 0.7 \\ 0.8 \end{bmatrix} = \begin{bmatrix} 1.4 \\ 3.2 \end{bmatrix} + \begin{bmatrix} 0.7 \\ 0.8 \end{bmatrix} = \begin{bmatrix} 2.1 \\ 4.0 \end{bmatrix}
$$

In PyTorch:

```python
import torch

W = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
b = torch.tensor([0.7, 0.8])
x = torch.tensor([1.0, 2.0, 3.0])

y = W @ x + b
print(f"y: {y}")  # tensor([2.1, 4.0])
```

### 4.1.4 Summary: Linear Models

| Model | Mathematical Form | Parameters | Input Shape | Output Shape |
|-------|-------------------|------------|-------------|--------------|
| Simple linear regression | $ \hat{y} = a + bx $ | $ a, b \in \mathbb{R} $ | $ (N,) $ | $ (N,) $ |
| Multiple linear regression | $ \hat{y} = \mathbf{w}^T \mathbf{x} + b $ | $ \mathbf{w} \in \mathbb{R}^d, b \in \mathbb{R} $ | $ (N, d) $ | $ (N,) $ |
| Linear layer | $ \mathbf{y} = W\mathbf{x} + \mathbf{b} $ | $ W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}, \mathbf{b} \in \mathbb{R}^{d_{\text{out}}} $ | $ (N, d_{\text{in}}) $ | $ (N, d_{\text{out}}) $ |

> **Exercise 4.1:** For each of the following linear models, write the mathematical form, identify the parameters, and compute the output for the given input.
>
> 1. $ \hat{y} = 2 + 3x $ for $ x = 4 $
> 2. $ \hat{y} = 1 \cdot x_1 + 2 \cdot x_2 + 3 $ for $ \mathbf{x} = [1, 1] $
> 3. $ \mathbf{y} = W\mathbf{x} + \mathbf{b} $ with $ W = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} $, $ \mathbf{b} = [0, 0] $, $ \mathbf{x} = [5, 7] $
> 4. $ \mathbf{y} = W\mathbf{x} + \mathbf{b} $ with $ W = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} $, $ \mathbf{b} = [0.1, 0.2] $, $ \mathbf{x} = [1, 0, -1] $

---

## 4.2 From Parameters to `nn.Parameter`

### 4.2.1 The Need for Parameter Registration

In Module 3, we initialized parameters as leaf tensors with `requires_grad=True`:

```python
a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)
```

This works for simple models, but it has limitations. The parameters are not registered with any model, so you cannot easily:
- Move them to a device with `model.to(device)`
- Save and load them with `model.state_dict()`
- Iterate over them with `model.parameters()`
- Share them across modules

`nn.Parameter` solves these problems. It is a subclass of `torch.Tensor` that is automatically registered as a parameter when assigned as an attribute of an `nn.Module`.

### 4.2.2 Creating `nn.Parameter`

The syntax is:

```python
import torch.nn as nn

a = nn.Parameter(torch.randn(1))
b = nn.Parameter(torch.randn(1))
```

By default, `nn.Parameter` sets `requires_grad=True`. You can override this:

```python
a = nn.Parameter(torch.randn(1), requires_grad=False)
```

### 4.2.3 The `ManualLinearRegression` Model

The original notebook introduces a `ManualLinearRegression` model that uses `nn.Parameter`:

```python
class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x):
        return self.a + self.b * x
```

This model has two learnable parameters, $ a $ and $ b $, and a forward pass that computes $ \hat{y} = a + bx $.

### 4.2.4 Inspecting the Model

Once the model is created, you can inspect its parameters:

```python
torch.manual_seed(42)
model = ManualLinearRegression()
print(model.state_dict())
```

The output is an `OrderedDict` mapping parameter names to tensors:

```
OrderedDict([('a', tensor([0.3367])), ('b', tensor([0.1288]))])
```

You can also iterate over the parameters:

```python
for name, param in model.named_parameters():
    print(f"{name}: {param}")
```

### 4.2.5 Training the Model

The training loop is identical to the one in Module 3, except that the parameters come from the model:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ManualLinearRegression().to(device)

lr = 1e-1
n_epochs = 1000
loss_fn = nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(n_epochs):
    model.train()
    yhat = model(x_train_tensor)
    loss = loss_fn(y_train_tensor, yhat)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(model.state_dict())
```

After training, the parameters should be close to $ a = 1 $, $ b = 2 $.

### 4.2.6 Summary: `nn.Parameter`

| Aspect | Plain Tensor | `nn.Parameter` |
|--------|--------------|----------------|
| `requires_grad` | Must set manually | Set automatically |
| Registration | Not registered | Registered with module |
| Device movement | Manual | Automatic with `model.to(device)` |
| State dict | Not included | Included in `state_dict()` |
| Iteration | Manual | `model.parameters()` |
| Sharing | Manual | Automatic |

> **Exercise 4.2:** Implement a `ManualPolynomialRegression` model with parameters $ a, b, c $ such that:
>
> $$ \hat{y} = a + bx + cx^2 $$
>
> 1. Define the model as a subclass of `nn.Module`.
> 2. Register the parameters with `nn.Parameter`.
> 3. Implement the `forward()` method.
> 4. Train the model on synthetic data with $ a = 1, b = 2, c = 3 $.
> 5. Verify that the learned parameters are close to the true values.

---

## 4.3 The `nn.Module` Base Class

### 4.3.1 What Is `nn.Module`?

`nn.Module` is the base class for all PyTorch models. It provides:
- **Parameter registration:** Attributes assigned as `nn.Parameter` are automatically registered.
- **Submodule registration:** Attributes assigned as `nn.Module` are automatically registered.
- **Device movement:** `model.to(device)` moves all parameters and buffers.
- **State dict:** `model.state_dict()` returns all parameters and buffers.
- **Training mode:** `model.train()` and `model.eval()` set the mode.
- **Hooks:** `register_forward_hook`, `register_backward_hook`, etc.

### 4.3.2 The `__init__` Method

The `__init__` method defines the model's architecture. It must call `super().__init__()` first:

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers and parameters here
```

### 4.3.3 The `forward` Method

The `forward` method defines the computation. It takes the input and returns the output:

```python
def forward(self, x):
    # Compute output here
    return output
```

When you call `model(x)`, PyTorch invokes `model.forward(x)`. You should never call `model.forward(x)` directly.

### 4.3.4 Nested Models

Models can contain other models as submodules:

```python
class NestedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x
```

The submodules are automatically registered, so their parameters are included in `model.parameters()`.

### 4.3.5 The `LayerLinearRegression` Model

The original notebook introduces a `LayerLinearRegression` model that uses `nn.Linear`:

```python
class LayerLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
```

This model has the same mathematical form as `ManualLinearRegression`, but the parameters are managed by `nn.Linear`.

### 4.3.6 Inspecting the Model

```python
model = LayerLinearRegression()
print(model.state_dict())
```

The output is:

```
OrderedDict([('linear.weight', tensor([[0.7645]])), ('linear.bias', tensor([0.8300]))])
```

The parameter names are prefixed with the submodule name (`linear`).

### 4.3.7 Summary: `nn.Module`

| Method/Attribute | Description | Example |
|------------------|-------------|---------|
| `__init__` | Define architecture | `self.linear = nn.Linear(1, 1)` |
| `forward` | Define computation | `return self.linear(x)` |
| `model(x)` | Invoke forward pass | `y = model(x)` |
| `model.parameters()` | Iterate over parameters | `for p in model.parameters()` |
| `model.named_parameters()` | Iterate with names | `for n, p in model.named_parameters()` |
| `model.state_dict()` | Get state dict | `model.state_dict()` |
| `model.to(device)` | Move to device | `model.to("cuda")` |
| `model.train()` | Set training mode | `model.train()` |
| `model.eval()` | Set evaluation mode | `model.eval()` |
| `model.zero_grad()` | Zero gradients | `model.zero_grad()` |

> **Exercise 4.3:** Implement the following models as subclasses of `nn.Module`:
>
> 1. A model with two `nn.Linear` layers and a ReLU activation:
>    $$
>    \hat{y} = W_2 \cdot \text{ReLU}(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
>    $$
> 2. A model with three `nn.Linear` layers and ReLU activations.
> 3. A model that takes two inputs, processes them with separate linear layers, concatenates the outputs, and passes them through a final linear layer.

---

## 4.4 The `nn.Linear` Layer

### 4.4.1 Definition

`nn.Linear(in_features, out_features)` implements the linear map:

$$
\mathbf{y} = W \mathbf{x} + \mathbf{b}
$$

where $ W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}} $ and $ \mathbf{b} \in \mathbb{R}^{d_{\text{out}}} $.

### 4.4.2 Parameter Initialization

By default, PyTorch initializes the weights with a uniform distribution:

$$
W_{ij} \sim \mathcal{U}\left(-\frac{1}{\sqrt{d_{\text{in}}}}, \frac{1}{\sqrt{d_{\text{in}}}}\right)
$$

and the bias with the same distribution. This is known as **Kaiming uniform initialization**.

```python
linear = nn.Linear(4, 3)
print(f"Weight shape: {linear.weight.shape}")
print(f"Bias shape: {linear.bias.shape}")
print(f"Weight: {linear.weight}")
print(f"Bias: {linear.bias}")
```

### 4.4.3 The Forward Pass

```python
x = torch.randn(8, 4)  # Batch of 8, 4 features each
y = linear(x)
print(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")
```

The output has shape $ (8, 3) $, as expected.

### 4.4.4 Manual Computation

We can reproduce the linear layer manually:

```python
W = linear.weight  # Shape (3, 4)
b = linear.bias    # Shape (3,)
y_manual = x @ W.T + b

print(f"Manual y shape: {y_manual.shape}")
print(f"Matches: {torch.allclose(y, y_manual)}")
```

### 4.4.5 Research Relevance

`nn.Linear` is the most common layer in neural networks. It appears in:
- **MLPs:** Stacked linear layers with nonlinearities.
- **Transformers:** Query, key, value projections; feed-forward networks; output projections.
- **CNNs:** The final classification head.
- **Autoencoders:** Encoder and decoder.

Understanding `nn.Linear` is essential for understanding any neural network architecture.

### 4.4.6 Summary: `nn.Linear`

| Aspect | Value |
|--------|-------|
| Mathematical form | $ \mathbf{y} = W\mathbf{x} + \mathbf{b} $ |
| Weight shape | $ (d_{\text{out}}, d_{\text{in}}) $ |
| Bias shape | $ (d_{\text{out}},) $ |
| Input shape | $ (N, d_{\text{in}}) $ |
| Output shape | $ (N, d_{\text{out}}) $ |
| Initialization | Kaiming uniform |
| Parameters | 2 (weight, bias) |

> **Exercise 4.4:** For each of the following, create an `nn.Linear` layer, inspect its parameters, and compute the output for a random input.
>
> 1. `nn.Linear(10, 5)`
> 2. `nn.Linear(5, 1)`
> 3. `nn.Linear(784, 256)` (typical for MNIST)
> 4. `nn.Linear(768, 3072)` (typical for transformer feed-forward)

---

## 4.5 Sequential Models

### 4.5.1 Definition

`nn.Sequential` is a container that chains modules together. The output of each module is the input to the next:

$$
\mathbf{y} = f_n(f_{n-1}(\cdots f_1(\mathbf{x}) \cdots))
$$

### 4.5.2 Creating a Sequential Model

```python
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)
```

This model is equivalent to the nested model from Section 4.3.4.

### 4.5.3 The Forward Pass

```python
x = torch.randn(8, 10)
y = model(x)
print(f"y shape: {y.shape}")  # (8, 1)
```

### 4.5.4 Inspecting the Model

```python
print(model)
```

The output is:

```
Sequential(
  (0): Linear(in_features=10, out_features=20, bias=True)
  (1): ReLU()
  (2): Linear(in_features=20, out_features=1, bias=True)
)
```

The layers are indexed by their position in the sequence.

### 4.5.5 When to Use `nn.Sequential`

`nn.Sequential` is convenient for simple feed-forward models. However, it has limitations:
- It cannot handle branching (e.g., residual connections).
- It cannot handle multiple inputs or outputs.
- It cannot handle control flow (e.g., conditional computation).

For complex architectures, you should use a custom `nn.Module`.

### 4.5.6 The `Sequential` Model in the Notebook

The original notebook uses `nn.Sequential` for the linear regression model:

```python
model = nn.Sequential(nn.Linear(1, 1)).to(device)
```

This is a single linear layer, equivalent to `LayerLinearRegression`.

### 4.5.7 Summary: `nn.Sequential`

| Aspect | Value |
|--------|-------|
| Purpose | Chain modules together |
| Forward pass | Sequential composition |
| Indexing | By position |
| Limitations | No branching, no multiple I/O |
| Use case | Simple feed-forward models |

> **Exercise 4.5:** Implement the following models using `nn.Sequential`:
>
> 1. A 3-layer MLP with ReLU activations: `Linear(10, 20) → ReLU → Linear(20, 20) → ReLU → Linear(20, 1)`
> 2. A convolutional model: `Conv2d(3, 16, 3) → ReLU → MaxPool2d(2) → Conv2d(16, 32, 3) → ReLU → MaxPool2d(2) → Flatten → Linear(32*6*6, 10)`
> 3. A transformer encoder block (simplified): `Linear(768, 768) → ReLU → Linear(768, 768)`

---

## 4.6 Model Initialization

### 4.6.1 Why Initialization Matters

The initial values of the parameters affect both the convergence speed and the final performance of the model. Poor initialization can lead to:
- **Vanishing gradients:** Gradients become exponentially small as they propagate backward.
- **Exploding gradients:** Gradients become exponentially large.
- **Symmetry breaking failure:** All neurons learn the same features.

### 4.6.2 Common Initialization Schemes

**Xavier/Glorot initialization** (for tanh/sigmoid activations):

$$
W_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{d_{\text{in}} + d_{\text{out}}}}, \sqrt{\frac{6}{d_{\text{in}} + d_{\text{out}}}}\right)
$$

**Kaiming/He initialization** (for ReLU activations):

$$
W_{ij} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{d_{\text{in}}}}\right)
$$

**Orthogonal initialization** (for RNNs):

$$
W = U \Sigma V^T \quad \text{(orthogonal)}
$$

### 4.6.3 Applying Initialization in PyTorch

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)
model.apply(init_weights)
```

The `apply` method applies a function to all submodules recursively.

### 4.6.4 Research Relevance

Initialization is a hyperparameter that can significantly affect results. When reproducing a paper, always check whether the authors specify their initialization scheme. If they do not, this is a reproducibility concern.

### 4.6.5 Summary: Initialization

| Scheme | Formula | Use Case |
|--------|---------|----------|
| Xavier/Glorot | $ \mathcal{U}\left(-\sqrt{\frac{6}{d_{\text{in}} + d_{\text{out}}}}, \sqrt{\frac{6}{d_{\text{in}} + d_{\text{out}}}}\right) $ | Tanh, sigmoid |
| Kaiming/He | $ \mathcal{N}\left(0, \sqrt{\frac{2}{d_{\text{in}}}}\right) $ | ReLU |
| Orthogonal | $ W = U\Sigma V^T $ | RNNs |
| Zero | $ W = 0 $ | Bias |
| Constant | $ W = c $ | Special cases |

> **Exercise 4.6:** For each of the following models, apply the appropriate initialization scheme and verify that the initial outputs have reasonable statistics (e.g., mean ≈ 0, std ≈ 1).
>
> 1. A 3-layer MLP with ReLU activations.
> 2. A 3-layer MLP with tanh activations.
> 3. A single `nn.Linear` layer.

---

## 4.7 Worked Example: Full Model Pipeline

Let us combine everything we have learned into a complete model pipeline.

### 4.7.1 Data Generation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(42)

# Generate data
N = 100
x = torch.rand(N, 1)
y = 1 + 2 * x + 0.1 * torch.randn(N, 1)

# Train/validation split
idx = torch.randperm(N)
train_idx = idx[:80]
val_idx = idx[80:]
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
x_train, y_train = x_train.to(device), y_train.to(device)
x_val, y_val = x_val.to(device), y_val.to(device)
```

### 4.7.2 Model Definition

```python
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression().to(device)
print(model.state_dict())
```

### 4.7.3 Loss and Optimizer

```python
loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=1e-1)
```

### 4.7.4 Training Loop

```python
n_epochs = 1000
losses = []
val_losses = []

for epoch in range(n_epochs):
    # Training
    model.train()
    yhat = model(x_train)
    loss = loss_fn(y_train, yhat)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    losses.append(loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        yhat_val = model(x_val)
        val_loss = loss_fn(y_val, yhat_val)
        val_losses.append(val_loss.item())

print(model.state_dict())
```

The output is:

```
OrderedDict([('linear.weight', tensor([[1.9690]])), ('linear.bias', tensor([1.0235]))])
```

The learned weight $ \approx 1.97 $ and bias $ \approx 1.02 $ are close to the true values $ b = 2 $, $ a = 1 $.

### 4.7.5 Reflection

This pipeline illustrates the full model workflow. The model is defined as a subclass of `nn.Module`, with a linear layer registered as a submodule. The forward pass is defined in the `forward` method. The training loop performs the forward pass, computes the loss, calls `.backward()` to compute gradients, and calls `optimizer.step()` to update parameters. The validation loop uses `torch.no_grad()` to disable gradient tracking.

**Research relevance:** This is the canonical model pipeline in PyTorch. Every research project you run will follow this structure, with modifications for the specific model, data, and task.

> **Exercise 4.7:** Extend the pipeline above to include:
>
> 1. A two-layer MLP with ReLU activation.
> 2. Early stopping (stop training when validation loss stops improving).
> 3. Model checkpointing (save the best model).
> 4. Logging (save the loss curves to a file).

---

## 4.8 Model State: `state_dict()` and Checkpointing

### 4.8.1 What Is `state_dict()`?

`state_dict()` returns a dictionary mapping parameter and buffer names to tensors:

```python
model = LinearRegression()
print(model.state_dict())
```

The output is:

```
OrderedDict([('linear.weight', tensor([[...]])), ('linear.bias', tensor([...]))])
```

### 4.8.2 Saving and Loading Models

**Saving:**

```python
torch.save(model.state_dict(), 'model.pth')
```

**Loading:**

```python
model = LinearRegression()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### 4.8.3 Saving the Optimizer State

For checkpointing (resuming training), you also need to save the optimizer state:

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item()
}
torch.save(checkpoint, 'checkpoint.pth')
```

**Loading:**

```python
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

### 4.8.4 Research Relevance

Checkpointing is essential for long-running experiments. It allows you to resume training after interruptions and to save the best model for evaluation.

### 4.8.5 Summary: State Dict

| Method | Description |
|--------|-------------|
| `model.state_dict()` | Get parameter and buffer dict |
| `model.load_state_dict(dict)` | Load parameters and buffers |
| `torch.save(obj, path)` | Save object to file |
| `torch.load(path)` | Load object from file |
| `optimizer.state_dict()` | Get optimizer state |
| `optimizer.load_state_dict(dict)` | Load optimizer state |

> **Exercise 4.8:** Implement a checkpointing system for the training pipeline that:
>
> 1. Saves the model and optimizer state every 100 epochs.
> 2. Saves the best model based on validation loss.
> 3. Allows resuming training from a checkpoint.
> 4. Logs the training and validation losses to a CSV file.

---

## 4.9 Common Errors and Debugging

### 4.9.1 Error: `AttributeError: 'MyModel' object has no attribute 'forward'`

This error occurs when you call `model.forward(x)` directly instead of `model(x)`. The fix is to call `model(x)`.

### 4.9.2 Error: `RuntimeError: Expected all tensors to be on the same device`

This error occurs when the model and input are on different devices. The fix is to move both to the same device.

```python
model = model.to(device)
x = x.to(device)
y = model(x)
```

### 4.9.3 Error: `RuntimeError: size mismatch`

This error occurs when the input shape does not match the model's expected input shape. The fix is to check the shapes.

```python
model = nn.Linear(10, 5)
x = torch.randn(8, 20)  # Wrong shape
# y = model(x)  # Error
x = torch.randn(8, 10)  # Correct shape
y = model(x)
```

### 4.9.4 Error: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

This error occurs when the dimensions of the input and the weight matrix do not match for matrix multiplication. The fix is to check the shapes.

### 4.9.5 Summary: Common Model Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `no attribute 'forward'` | Calling `model.forward(x)` | Call `model(x)` |
| `same device` | Model and input on different devices | Move both to same device |
| `size mismatch` | Input shape wrong | Check shapes |
| `mat1 and mat2` | Matrix multiplication mismatch | Check weight and input shapes |

> **Exercise 4.9:** For each of the following errors, identify the cause and fix it:
>
> ```python
> # Error 1
> model = nn.Linear(10, 5)
> x = torch.randn(8, 20)
> y = model(x)
> ```
>
> ```python
> # Error 2
> model = nn.Linear(10, 5).to("cuda")
> x = torch.randn(8, 10)
> y = model(x)
> ```
>
> ```python
> # Error 3
> class MyModel(nn.Module):
>     def __init__(self):
>         self.linear = nn.Linear(10, 5)
>     def forward(self, x):
>         return self.linear(x)
> model = MyModel()
> ```
>
> ```python
> # Error 4
> model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
> x = torch.randn(8, 10)
> y = model.forward(x)
> ```

---

## 4.10 Research Application: Model Ablation

### 4.10.1 What Is Ablation?

**Ablation** is the process of removing or modifying components of a model to understand their contribution. It is a standard technique in research for attributing performance to specific design choices.

For example, to ablate the effect of a ReLU activation, you would train the model with and without the activation and compare the results.

### 4.10.2 Implementing Ablation

```python
def create_model(use_relu=True):
    layers = [nn.Linear(10, 20)]
    if use_relu:
        layers.append(nn.ReLU())
    layers.append(nn.Linear(20, 1))
    return nn.Sequential(*layers)

# Train with ReLU
model_with_relu = create_model(use_relu=True)
# Train without ReLU
model_without_relu = create_model(use_relu=False)
```

### 4.10.3 Research Relevance

Ablation studies are essential for understanding which components of a model are responsible for its performance. A well-designed ablation study isolates the effect of each component by holding everything else constant.

**Research relevance:** When reproducing a paper, always check whether the authors performed ablation studies. If they did not, this is a reproducibility concern.

### 4.10.4 Summary: Ablation

| Aspect | Description |
|--------|-------------|
| Purpose | Attribute performance to components |
| Method | Remove or modify one component at a time |
| Control | Keep everything else constant |
| Reporting | Report results for each variant |

> **Exercise 4.10:** Design and run an ablation study for the following model:
>
> $$
> \hat{y} = W_3 \cdot \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2) + \mathbf{b}_3
> $$
>
> Vary the following:
>
> 1. Number of layers (1, 2, 3)
> 2. Activation function (ReLU, tanh, sigmoid)
> 3. Width of hidden layers (10, 20, 50)
>
> Report the validation loss for each variant.

---

## 4.11 Module Summary

| Concept | Mathematical Form | PyTorch | Research Relevance |
|---------|-------------------|---------|-------------------|
| Linear model | $ \hat{y} = \mathbf{w}^T \mathbf{x} + b $ | `nn.Linear` | Baseline |
| Linear layer | $ \mathbf{y} = W\mathbf{x} + \mathbf{b} $ | `nn.Linear(d_in, d_out)` | Building block |
| Parameter | $ \theta $ | `nn.Parameter` | Learnable weights |
| Module | $ f_\theta $ | `nn.Module` | Model abstraction |
| Forward pass | $ \hat{y} = f_\theta(\mathbf{x}) $ | `model(x)` | Computation |
| Sequential | $ f_n \circ \cdots \circ f_1 $ | `nn.Sequential` | Simple models |
| Initialization | $ W \sim \mathcal{D} $ | `nn.init` | Convergence |
| State dict | $ \{\theta_i\} $ | `model.state_dict()` | Checkpointing |
| Training mode | — | `model.train()` | Dropout, BatchNorm |
| Evaluation mode | — | `model.eval()` | Inference |
| Ablation | — | Vary components | Attribution |

---

## 4.12 Capstone Thread: Model Skills for Your Project

By the end of this module, you should be able to define custom models with `nn.Module`; register learnable parameters with `nn.Parameter`; implement the `forward()` method; compose layers with `nn.Linear` and `nn.Sequential`; inspect model state with `state_dict()`; and articulate why modular model design matters for research reproducibility and extensibility.

For your capstone project, you will need these skills to implement your model, inspect its parameters, save and load checkpoints, and perform ablation studies.

In Module 5, we will build on this foundation to study **training loops and optimization** — the machinery that trains your model.

---

## 4.13 Additional Resources

### On Model Design

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 6: Deep Feedforward Networks.
- PyTorch `nn.Module` documentation: [Link](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)

### On Initialization

- Glorot, X., & Bengio, Y. (2010). "Understanding the Difficulty of Training Deep Feedforward Neural Networks." *AISTATS*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." *ICCV*.

### On Ablation

- Meyes, R., et al. (2019). "Ablation Studies in Artificial Neural Networks." *arXiv:1901.08644*.
- Ren, M., et al. (2021). "Towards Interpretable Ablation Studies." *ICLR*.

### On Checkpointing

- PyTorch saving and loading documentation: [Link](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

---

*End of Module 4*