# Module 3: Autograd and Computation Graphs

## Stanford University | CS 336: PyTorch for Research

**Instructor:** Alpha Alimamy Kamara

**Module Length:** 2 weeks (4 lectures + 2 lab sessions)

**Prerequisites:** Module 2 completed; familiarity with tensors, matrix multiplication, and basic calculus (partial derivatives, chain rule).

---

## 3.0 Module Overview

In Module 2, we learned that tensors are the 
fundamental data structure of PyTorch. 
But tensors alone do not make a deep learning 
framework. What distinguishes PyTorch from NumPy 
is **autograd**, the automatic differentiation engine that computes gradients of arbitrary scalar functions with respect to arbitrary tensor inputs.

This module is about autograd and the **dynamic computation graph** that PyTorch builds on the fly. We will begin with the mathematics of differentiation, then develop the computational graph abstraction, then examine how PyTorch implements both. Along the way, we will encounter the three most common mistakes researchers make with autograd: forgetting to zero gradients, accidentally accumulating gradients, and breaking the graph with in-place operations.

The pedagogical approach remains 
**mathematics first, code second**. For 
every concept, we will write the mathematical
object explicitly the derivative, the Jacobian,
the chain rule, and only then translate it into
PyTorch. This mirrors how research is conducted:
you derive the gradient on paper, and PyTorch 
computes it for you.

By the end of this module, you will be able to 
compute gradients of scalar functions with
respect to tensor inputs; understand and 
visualize computation graphs; diagnose and fix 
common autograd errors; implement manual 
gradient descent and compare it against PyTorch's
autograd; and articulate why autograd is 
essential for research at scale.

The module is self-paced. Work through the mathematics carefully before running the code. The code will make much more sense if you understand what it is computing.

---

## 3.1 The Mathematics of Differentiation

### 3.1.1 The Derivative of a Scalar Function

Let $f: \mathbb{R} \to \mathbb{R}$ be a 
scalar function. The **derivative** of $f$ at a point $ x $ is defined as the limit:

$$
f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

Geometrically, $f'(x)$ is the slope of 
the tangent line to the graph of 
$f$ at $x$. It measures the instantaneous
rate of change of $f$ with respect to 
$x$.

For example, if $f(x) = x^2$, then:
$$
f'(x) = \lim_{h \to 0} \frac{(x + h)^2 - x^2}{h} = \lim_{h \to 0} \frac{x^2 + 2xh + h^2 - x^2}{h} = \lim_{h \to 0} (2x + h) = 2x
$$

In PyTorch, we can approximate this derivative numerically:

```python
import torch

def f(x):
    return x ** 2

x = torch.tensor(3.0)
h = 1e-5
derivative_approx = (f(x + h) - f(x)) / h
print(f"Numerical derivative of x^2 at x=3: {derivative_approx:.6f}")
print(f"Analytical derivative: {2 * x:.6f}")
```

The numerical approximation is close to 
the analytical value $2 \cdot 3 = 6$.

### 3.1.2 The Partial Derivative

For a function of several variables
$f: \mathbb{R}^n \to \mathbb{R} $, 
the **partial derivative** with respect to 
$x_i$ is:

$$
\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}
$$

It measures the rate of change of $f$ with 
respect to $ x_i$, holding all other variables fixed.

For example, if $ f(x, y) = x^2 + 3xy + y^2 $, then:

$$
\frac{\partial f}{\partial x} = 2x + 3y, \quad \frac{\partial f}{\partial y} = 3x + 2y
$$

In PyTorch:

```python
def f(x, y):
    return x ** 2 + 3 * x * y + y ** 2

x = torch.tensor(2.0)
y = torch.tensor(3.0)
h = 1e-5

df_dx_approx = (f(x + h, y) - f(x, y)) / h
df_dy_approx = (f(x, y + h) - f(x, y)) / h

print(f"df/dx numerical: {df_dx_approx:.6f}, analytical: {2*x + 3*y:.6f}")
print(f"df/dy numerical: {df_dy_approx:.6f}, analytical: {3*x + 2*y:.6f}")
```

### 3.1.3 The Gradient

For a function $f: \mathbb{R}^n \to \mathbb{R}$, 
the **gradient** is the vector of all 
partial derivatives:

$$
\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix} \in \mathbb{R}^n
$$

The gradient points in the direction of the 
steepest ascent of $f$. To minimize $f$, we move in the direction of the negative gradient:

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)
$$

where $\eta > 0$ is the **learning rate**. 
This is the **gradient descent** algorithm, 
the foundation of nearly all deep learning 
optimization.

For example, if $f(\mathbf{x}) = 
\|\mathbf{x}\|^2 = x_1^2 + x_2^2 +
\cdots + x_n^2$, then:

$$
\nabla f(\mathbf{x}) = \begin{bmatrix} 2x_1 \\ 2x_2 \\ \vdots \\ 2x_n \end{bmatrix} = 2\mathbf{x}
$$

In PyTorch:

```python
def f(x):
    return (x ** 2).sum()

x = torch.tensor([1.0, 2.0, 3.0])
h = 1e-5

grad_approx = torch.zeros_like(x)
for i in range(len(x)):
    x_plus = x.clone()
    x_plus[i] += h
    grad_approx[i] = (f(x_plus) - f(x)) / h

print(f"Numerical gradient: {grad_approx}")
print(f"Analytical gradient: {2 * x}")
```

### 3.1.4 The Chain Rule

The **chain rule** is the fundamental tool for 
computing derivatives of composite functions.
If $f(g(x))$ is a composition of functions, 
then:

$$
\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}
$$

For multivariate functions, if 
$\mathbf{y} = g(\mathbf{x})$ and 
$z = f(\mathbf{y})$, then:

$$
\frac{\partial z}{\partial x_i} = \sum_j \frac{\partial z}{\partial y_j} \frac{\partial y_j}{\partial x_i}
$$

In matrix form, if $\mathbf{y} = g(\mathbf{x})$ 
and $z = f(\mathbf{y})$, then:

$$
\nabla_{\mathbf{x}} z = J_g(\mathbf{x})^T \nabla_{\mathbf{y}} z
$$

where $J_g(\mathbf{x})$ is the **Jacobian** of 
$ g$ at $\mathbf{x}$:

$$
J_g(\mathbf{x}) = \begin{bmatrix} \frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n} \end{bmatrix} \in \mathbb{R}^{m \times n}
$$

The chain rule is what makes deep learning possible. A neural network is a composition of many functions (layers), and the gradient of the loss with respect to the parameters is computed by applying the chain rule recursively. This is **backpropagation**.

**Worked example 3.1:** Consider the 
composition $z = f(g(x))$ where 
$g(x) = x^2 + 1$ and $f(y) = y^3$. Compute 
$\frac{dz}{dx}$.

By the chain rule:
$$
\frac{dz}{dx} = \frac{df}{dy} \cdot \frac{dg}{dx} = 3y^2 \cdot 2x = 3(x^2 + 1)^2 \cdot 2x = 6x(x^2 + 1)^2
$$

At $x = 1$:

$$
\frac{dz}{dx}\bigg|_{x=1} = 6 \cdot 1 \cdot (1 + 1)^2 = 6 \cdot 4 = 24
$$

In PyTorch:

```python
x = torch.tensor(1.0, requires_grad=True)
y = x ** 2 + 1
z = y ** 3

z.backward()
print(f"dz/dx at x=1: {x.grad}")  # Should be 24
```

### 3.1.5 Summary: Differentiation

| Concept | Mathematical Form | PyTorch |
|---------|-------------------|---------|
| Derivative | $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$ | `torch.autograd.grad` |
| Partial derivative | $\frac{\partial f}{\partial x_i} $ | `x.grad[i]` |
| Gradient | $\nabla f = [\partial f / \partial x_i] $ | `x.grad` |
| Chain rule | $\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$ | `loss.backward()` |
| Jacobian | $J_{ij} = \frac{\partial y_i}{\partial x_j} $ | `torch.autograd.functional.jacobian` |
| Gradient descent | $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)$ | `optimizer.step()` |

> **Exercise 3.1:** For each of the following functions, compute the gradient analytically, then verify numerically and with PyTorch's autograd.
>
> 1. $f(x) = 3x^2 + 2x + 1 $
> 2. $ f(x, y) = x^2 y + xy^2$
> 3. $ f(\mathbf{x}) = \sum_{i=1}^n x_i^3 $
> 4. $f(\mathbf{x}) = \log\left(\sum_{i=1}^n e^{x_i}\right) $ (log-sum-exp)

## 3.2 The Autograd Engine

### 3.2.1 What Is Autograd?

**Autograd** is PyTorch's automatic differentiation engine. It computes gradients of scalar functions with respect to tensor inputs by building a **computational graph** during the forward pass and traversing it backward during the backward pass.

The key idea is that every operation on a tensor that has `requires_grad=True` is recorded. When you call `.backward()` on a scalar output, PyTorch traverses the graph in reverse, applying the chain rule at each node to compute gradients.

Mathematically, autograd computes:

$$
\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}_n} \cdot \frac{\partial \mathbf{y}_n}{\partial \mathbf{y}_{n-1}} \cdots \frac{\partial \mathbf{y}_1}{\partial \mathbf{x}}
$$

where $L$ is the scalar loss, $\mathbf{y}_i$ 
are intermediate tensors, and $ \mathbf{x}$ is the input.

### 3.2.2 The `requires_grad` Attribute

The `requires_grad` attribute tells PyTorch to track operations on a tensor. By default, it is `False` for tensors created from data and `True` for parameters of `nn.Module`.

```python
import torch

# Tensor without gradient tracking
x = torch.tensor([1.0, 2.0, 3.0])
print(f"x.requires_grad: {x.requires_grad}")

# Tensor with gradient tracking
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"x.requires_grad: {x.requires_grad}")

# Operations on x are tracked
y = x ** 2
print(f"y.requires_grad: {y.requires_grad}")
print(f"y.grad_fn: {y.grad_fn}")
```

The `grad_fn` attribute tells you which
operation produced the tensor. For $`y = x^2`$, the `grad_fn` is `PowBackward0`.

### 3.2.3 Computing Gradients with `.backward()`

To compute gradients, you call `.backward()` on a scalar tensor. This populates the `.grad` attribute of all tensors that have `requires_grad=True` and contributed to the scalar.

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()  # Scalar
y.backward()

print(f"x.grad: {x.grad}")  # Should be 2 * x = [2, 4, 6]
```

**Important:** `.backward()` can only be 
called on a scalar (0-dimensional) tensor. To compute gradients of a non-scalar, you must first reduce it to a scalar (e.g., with `.sum()` or `.mean()`).

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # Non-scalar

# y.backward()  # Error: grad can be implicitly created only for scalar outputs

# Correct: reduce to scalar
y.sum().backward()
print(f"x.grad: {x.grad}")
```

### 3.2.4 The `.grad` Attribute

After calling `.backward()`, the gradients are stored in the `.grad` attribute of the input tensors. By default, gradients accumulate across calls to `.backward()`. This is useful for some applications (e.g., gradient accumulation for large batches) but is a common source of bugs.

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# First backward
y1 = (x ** 2).sum()
y1.backward()
print(f"After first backward: x.grad = {x.grad}")

# Second backward (accumulates!)
y2 = (x ** 3).sum()
y2.backward()
print(f"After second backward: x.grad = {x.grad}")  # Accumulated!
```

To prevent accumulation, you must zero the gradients before each backward pass:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

y1 = (x ** 2).sum()
y1.backward()
print(f"After first backward: x.grad = {x.grad}")

x.grad.zero_()  # Zero the gradients

y2 = (x ** 3).sum()
y2.backward()
print(f"After second backward: x.grad = {x.grad}")  # Not accumulated
```

### 3.2.5 The `torch.no_grad()` Context Manager

During inference (evaluation), we do not need gradients. Computing them wastes memory and time. The `torch.no_grad()` context manager disables gradient tracking:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

with torch.no_grad():
    y = x ** 2
    print(f"y.requires_grad: {y.requires_grad}")  # False
```

**Research relevance:** Always use `torch.no_grad()` during validation and inference. This can reduce memory usage by 2–3× and speed up computation.

### 3.2.6 Summary: Autograd Basics

| Concept | Mathematical Form | PyTorch | Research Relevance |
|---------|-------------------|---------|-------------------|
| Gradient tracking | $\frac{\partial L}{\partial \mathbf{x}}$ | `requires_grad=True` | Enable for parameters |
| Backward pass | Chain rule | `.backward()` | Compute gradients |
| Gradient storage | $\nabla_{\mathbf{x}} L$ | `.grad` | Access gradients |
| Gradient accumulation | $\sum_i \nabla_{\mathbf{x}} L_i$ | Default behavior | Useful for large batches |
| Gradient zeroing | $\nabla_{\mathbf{x}} L \leftarrow 0 $ | `.grad.zero_()` | Prevent accumulation |
| No-grad mode | — | `torch.no_grad()` | Inference, memory savings |

> **Exercise 3.2:** For each of the following scenarios, write the PyTorch code to compute the gradient. Be careful about gradient accumulation.
>
> 1. $ f(x) = x^3 $ at $ x = 2 $
> 2. $ f(x, y) = x^2 + y^2 $ at $ (x, y) = (1, 2)$
> 3. $f(\mathbf{x}) = \sum_i x_i^2 $ at $\mathbf{x} = [1, 2, 3] $
> 4. $ f(\mathbf{x}) = \log\left(\sum_i e^{x_i}\right)$ at $ \mathbf{x} = [1, 2, 3]$

---

## 3.3 The Dynamic Computation Graph

### 3.3.1 What Is a Computation Graph?

A **computation graph** is a directed acyclic graph (DAG) where:
- **Nodes** represent tensors (variables) or operations.
- **Edges** represent the flow of data from inputs to outputs.

For a function $L = f(g(h(\mathbf{x})))$, the computation graph is:
$$
\mathbf{x} \xrightarrow{h} \mathbf{y}_1 \xrightarrow{g} \mathbf{y}_2 \xrightarrow{f} L
$$

The forward pass computes $L$ from $\mathbf{x}$. The backward pass computes $\frac{\partial L}{\partial \mathbf{x}}$ by traversing the graph in reverse.

### 3.3.2 Dynamic vs. Static Graphs

PyTorch uses **dynamic** computation graphs, meaning the graph is built on the fly during the forward pass. This is in contrast to **static** graphs (e.g., TensorFlow 1.x), where the graph is defined once and then executed.

The dynamic approach has several advantages for research:
- **Flexibility:** The graph can change between iterations (e.g., for variable-length sequences).
- **Debuggability:** You can inspect intermediate values with standard Python debugging tools.
- **Pythonic:** Control flow (if/else, loops) works naturally.

The trade-off is that dynamic graphs are slightly slower than static graphs for production deployment. For research, the flexibility is almost always worth it.

### 3.3.3 Visualizing the Computation Graph

PyTorch provides tools to visualize computation graphs. The `torchviz` library generates a graph from a tensor:

```python
# Install torchviz
!pip install torchviz

import torch
from torchviz import make_dot

# Build a computation graph
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
z = y.sum()

# Visualize
make_dot(z, params={'x': x})
```

The output is a graph showing the operations that produced `z`. For a simple example like this, the graph is small. For a deep neural network, the graph can have thousands of nodes.

**Worked example 3.2:** Visualize the computation graph for the linear regression model from Module 2.

Recall the model:

$$
\hat{y} = a + b \cdot x
$$

$$
\text{error} = y - \hat{y}
$$

$$
\text{loss} = \text{mean}(\text{error}^2)
$$

In PyTorch:

```python
import torch
from torchviz import make_dot

# Parameters
a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)

# Data
x = torch.randn(80, 1)
y = 1 + 2 * x + 0.1 * torch.randn(80, 1)

# Forward pass
yhat = a + b * x
error = y - yhat
loss = (error ** 2).mean()

# Visualize
make_dot(loss, params={'a': a, 'b': b, 'yhat': yhat, 'error': error, 'loss': loss})
```

The graph shows the flow from `a` and `b` through `yhat`, `error`, and `loss`. This is exactly the graph that PyTorch traverses backward when you call `loss.backward()`.

### 3.3.4 Breaking the Graph

The computation graph is built only for operations on tensors with `requires_grad=True`. If you detach a tensor from the graph, operations on it are no longer tracked.

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
z = y.detach()  # Detach from graph
w = z ** 3

print(f"y.requires_grad: {y.requires_grad}")  # True
print(f"z.requires_grad: {z.requires_grad}")  # False
print(f"w.requires_grad: {w.requires_grad}")  # False
```

**Research relevance:** Detaching is used in:
- **Target networks** in reinforcement learning (to prevent gradients from flowing into the target).
- **Adversarial training** (to prevent gradients from flowing into the adversary).
- **Stop-gradient** operations in self-supervised learning (e.g., BYOL, SimSiam).

### 3.3.5 In-place Operations and the Graph

In-place operations (e.g., `x += 1`, `x.relu_()`) modify a tensor without creating a new one. This can break the computation graph because PyTorch needs the original value to compute gradients.

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

# In-place operation on x
# x += 1  # Error: a leaf Variable that requires grad has been used in an in-place operation

# In-place operation on y
# y += 1  # Error: a view of a leaf Variable that requires grad is being used in an in-place operation

# Correct: use out-of-place operations
z = y + 1
print(f"z: {z}")
```

**Research relevance:** In-place operations are sometimes used for memory efficiency (e.g., `nn.ReLU(inplace=True)`). However, they can cause subtle bugs. Use them with caution.

### 3.3.6 Summary: Computation Graphs

| Concept | Description | PyTorch | Research Relevance |
|---------|-------------|---------|-------------------|
| Computation graph | DAG of tensors and operations | Built automatically | Foundation of backprop |
| Dynamic graph | Built on the fly | Default in PyTorch | Flexibility, debuggability |
| Visualization | Graph of operations | `torchviz.make_dot` | Debugging, understanding |
| Detach | Break the graph | `.detach()` | Target networks, stop-gradient |
| In-place operations | Modify tensor without new allocation | `x += 1`, `x.relu_()` | Memory efficiency, but risky |

> **Exercise 3.3:** For each of the following, draw the computation graph by hand, then verify with `torchviz`.
>
> 1. $z = (x + y) \cdot (x - y) $
> 2. $z = \sigma(w \cdot x + b)$ where $\sigma$ is the sigmoid function
> 3. $z = \text{softmax}(\mathbf{x}) \cdot \mathbf{y}$
> 4. $z = \text{mean}(\text{ReLU}(W\mathbf{x} + \mathbf{b}))$

## 3.4 Manual Gradient Descent vs. Autograd

### 3.4.1 The Linear Regression Problem

We return to the linear regression problem from Module 2 and the original notebook. The model is:

$$
\hat{y}_i = a + b x_i
$$

The loss is the mean squared error:

$$
L(a, b) = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 = \frac{1}{N} \sum_{i=1}^N (y_i - a - b x_i)^2
$$

The gradients are:

$$
\frac{\partial L}{\partial a} = -\frac{2}{N} \sum_{i=1}^N (y_i - a - b x_i)
$$

$$
\frac{\partial L}{\partial b} = -\frac{2}{N} \sum_{i=1}^N x_i (y_i - a - b x_i)
$$

### 3.4.2 Manual Gradient Descent in NumPy

The original notebook implements manual gradient descent in NumPy:

```python
import numpy as np

# Data generation
np.random.seed(42)
x = np.random.rand(100, 1)
y = 1 + 2 * x + 0.1 * np.random.randn(100, 1)

# Train/validation split
idx = np.arange(100)
np.random.shuffle(idx)
train_idx = idx[:80]
val_idx = idx[80:]
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# Initialize parameters
np.random.seed(42)
a = np.random.randn(1)
b = np.random.randn(1)

# Hyperparameters
lr = 1e-1
n_epochs = 1000

for epoch in range(n_epochs):
    # Forward pass
    yhat = a + b * x_train

    # Compute loss
    error = y_train - yhat
    loss = (error ** 2).mean()

    # Compute gradients manually
    a_grad = -2 * error.mean()
    b_grad = -2 * (x_train * error).mean()

    # Update parameters
    a = a - lr * a_grad
    b = b - lr * b_grad

print(f"a = {a}, b = {b}")
```

The output is approximately $a = 1.0235 $, 
$b = 1.9690$, close to the true values 
$a = 1$, $b = 2$.

### 3.4.3 Autograd in PyTorch

The same problem in PyTorch with autograd:

```python
import torch

# Data generation
torch.manual_seed(42)
x = torch.rand(100, 1)
y = 1 + 2 * x + 0.1 * torch.randn(100, 1)

# Train/validation split
idx = torch.randperm(100)
train_idx = idx[:80]
val_idx = idx[80:]
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# Initialize parameters
torch.manual_seed(42)
a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)

# Hyperparameters
lr = 1e-1
n_epochs = 1000

for epoch in range(n_epochs):
    # Forward pass
    yhat = a + b * x_train

    # Compute loss
    error = y_train - yhat
    loss = (error ** 2).mean()

    # Compute gradients with autograd
    loss.backward()

    # Update parameters (with no_grad to avoid tracking)
    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad

    # Zero gradients
    a.grad.zero_()
    b.grad.zero_()

print(f"a = {a.item()}, b = {b.item()}")
```

The output is the same: $a = 1.0235$, 
$b = 1.9690$.

### 3.4.4 The Three Attempts at Parameter Update

The original notebook documents three attempts at updating parameters with autograd:

**First attempt:**

```python
a = a - lr * a.grad
b = b - lr * b.grad
```

This fails with `AttributeError: 'NoneType' object has no attribute 'zero_'` because reassigning `a` creates a new tensor that is not a leaf, and its `.grad` is `None`.

**Second attempt:**

```python
a -= lr * a.grad
b -= lr * b.grad
```

This fails with `RuntimeError: a leaf Variable that requires grad has been used in an in-place operation.` because in-place operations on leaf tensors that require gradients are not allowed.

**Third attempt (correct):**

```python
with torch.no_grad():
    a -= lr * a.grad
    b -= lr * b.grad
```

This works because `torch.no_grad()` disables gradient tracking, allowing in-place updates.

### 3.4.5 Why Zero the Gradients?

After each backward pass, gradients accumulate in `.grad`. If you do not zero them, the next backward pass will add to the existing gradients, leading to incorrect updates.

```python
# Without zeroing
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

for epoch in range(3):
    yhat = a + b * x_train
    loss = ((y_train - yhat) ** 2).mean()
    loss.backward()
    print(f"Epoch {epoch}: a.grad = {a.grad.item():.4f}")

# With zeroing
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

for epoch in range(3):
    yhat = a + b * x_train
    loss = ((y_train - yhat) ** 2).mean()
    loss.backward()
    print(f"Epoch {epoch}: a.grad = {a.grad.item():.4f}")
    a.grad.zero_()
    b.grad.zero_()
```

In the first case, the gradients grow with each epoch. In the second case, they are computed fresh each time.

### 3.4.6 Summary: Manual vs. Autograd

| Aspect | Manual (NumPy) | Autograd (PyTorch) |
|--------|----------------|-------------------|
| Gradient computation | Hand-derived | Automatic |
| Error-prone | Yes | No |
| Scalable | No | Yes |
| Flexibility | Limited | High |
| Research relevance | Historical | Standard |

> **Exercise 3.4:** Implement manual gradient descent for the following models, then compare with autograd.
>
> 1. Polynomial regression: $\hat{y} = a + bx + cx^2 $
> 2. Logistic regression:$ \hat{y} = \sigma(a + bx)$ with binary cross-entropy loss
> 3. Two-layer MLP: $\hat{y} = W_2 \cdot \text{ReLU}(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$


## 3.5 The Optimizer

### 3.5.1 From Manual Updates to Optimizers

The manual update $ a \leftarrow a - \eta \nabla_a L $ is the simplest form of gradient descent. PyTorch provides the `torch.optim` module with a variety of optimizers that implement more sophisticated update rules.

The most basic optimizer is **stochastic gradient descent (SGD)**:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L
$$

In PyTorch:

```python
import torch.optim as optim

# Parameters
a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)

# Optimizer
optimizer = optim.SGD([a, b], lr=1e-1)

# Training loop
for epoch in range(1000):
    yhat = a + b * x_train
    error = y_train - yhat
    loss = (error ** 2).mean()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(f"a = {a.item()}, b = {b.item()}")
```

The optimizer handles the update (`step()`) and the gradient zeroing (`zero_grad()`).

### 3.5.2 The Optimizer API

All PyTorch optimizers share the same API:

| Method | Description |
|--------|-------------|
| `optimizer.step()` | Update parameters using gradients |
| `optimizer.zero_grad()` | Zero the gradients of all parameters |
| `optimizer.param_groups` | Access parameter groups |
| `optimizer.state_dict()` | Get optimizer state (for checkpointing) |
| `optimizer.load_state_dict()` | Load optimizer state |

### 3.5.3 Why Optimizers Matter for Research

Different optimizers have different convergence properties. SGD with momentum is often preferred for computer vision, while Adam is often preferred for NLP and transformers. The choice can significantly affect both convergence speed and final performance.

**Research relevance:** The choice of optimizer is a hyperparameter that must be tuned. Reporting which optimizer was used, along with its hyperparameters, is essential for reproducibility.

### 3.5.4 Summary: Optimizers

| Optimizer | Update Rule | Use Case |
|-----------|-------------|----------|
| SGD | $ \theta \leftarrow \theta - \eta \nabla L $ | Simple, interpretable |
| SGD + Momentum | $ v \leftarrow \beta v + \nabla L; \theta \leftarrow \theta - \eta v $ | Computer vision |
| Adam | Adaptive per-parameter learning rates | NLP, transformers |
| RMSprop | Adaptive learning rates | RNNs, reinforcement learning |

> **Exercise 3.5:** Compare the convergence of SGD, SGD with momentum, and Adam on the linear regression problem. Plot the loss curves. Which converges fastest? Which is most stable?

---

## 3.6 Loss Functions

### 3.6.1 From Manual Loss to `nn.MSELoss`

In the manual implementation, we computed the loss as:

$$
L = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
$$

PyTorch provides `nn.MSELoss` which does the same:

```python
import torch.nn as nn

loss_fn = nn.MSELoss(reduction='mean')
loss = loss_fn(y_train, yhat)
```

### 3.6.2 Reduction Modes

The `reduction` parameter controls how the loss is aggregated:

| Reduction | Description | Formula |
|-----------|-------------|---------|
| `'mean'` | Average over all elements | $ \frac{1}{N} \sum_i \ell_i $ |
| `'sum'` | Sum over all elements | $ \sum_i \ell_i $ |
| `'none'` | No reduction | $ \ell_i $ |

```python
loss_fn_mean = nn.MSELoss(reduction='mean')
loss_fn_sum = nn.MSELoss(reduction='sum')
loss_fn_none = nn.MSELoss(reduction='none')

# Example
y_true = torch.tensor([1.0, 2.0, 3.0])
y_pred = torch.tensor([1.1, 2.1, 2.9])

print(f"Mean: {loss_fn_mean(y_true, y_pred)}")
print(f"Sum: {loss_fn_sum(y_true, y_pred)}")
print(f"None: {loss_fn_none(y_true, y_pred)}")
```

### 3.6.3 Loss Functions in Research

The choice of loss function encodes assumptions about the data. MSE assumes Gaussian noise; L1 assumes Laplace noise; cross-entropy assumes categorical distributions. Choosing the right loss is a research decision, not a default.

**Research relevance:** The loss function should be reported in any paper. Ablations over loss functions can reveal the sensitivity of the model to this choice.

### 3.6.4 Summary: Loss Functions

| Loss | Formula | Use Case |
|------|---------|----------|
| MSE | $ \frac{1}{N} \sum (y - \hat{y})^2 $ | Regression, Gaussian noise |
| MAE (L1) | $ \frac{1}{N} \sum |y - \hat{y}| $ | Regression, outliers |
| Cross-entropy | $ -\sum y \log \hat{y} $ | Classification |
| Binary cross-entropy | $ -\sum [y \log \hat{y} + (1-y) \log(1-\hat{y})] $ | Binary classification |
| NLL | $ -\sum \log p(y) $ | Probabilistic models |

> **Exercise 3.6:** Implement the following loss functions manually, then compare with PyTorch's built-in versions.
>
> 1. MSE
> 2. MAE
> 3. Binary cross-entropy
> 4. Cross-entropy for multi-class classification

---

## 3.7 From Manual to Modular: The Training Step

### 3.7.1 The `make_train_step` Function

The original notebook introduces a `make_train_step` function that encapsulates the training step:

```python
def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(y, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step
```

This is a **higher-order function** — a function that returns a function. It encapsulates the training step, making the training loop clean and reusable.

### 3.7.2 Why This Matters for Research

In research, you will run many experiments with different models, losses, and optimizers. A reusable training step reduces code duplication and the risk of bugs. It also makes your code more readable and easier to share.

**Research relevance:** The `make_train_step` pattern is a precursor to more sophisticated abstractions like PyTorch Lightning's `LightningModule` and Hugging Face's `Trainer`. Understanding the pattern helps you understand these frameworks.

### 3.7.3 The Training Loop

With `make_train_step`, the training loop becomes:

```python
train_step = make_train_step(model, loss_fn, optimizer)
losses = []

for epoch in range(n_epochs):
    loss = train_step(x_train_tensor, y_train_tensor)
    losses.append(loss)

print(model.state_dict())
```

This is much cleaner than the manual loop. The training step is now a single function call.

### 3.7.4 Summary: Training Step

| Component | Manual | `make_train_step` |
|-----------|--------|-------------------|
| Forward pass | Inline | Inside function |
| Loss computation | Inline | Inside function |
| Backward pass | Inline | Inside function |
| Parameter update | Inline | Inside function |
| Gradient zeroing | Inline | Inside function |
| Reusability | Low | High |
| Readability | Low | High |

> **Exercise 3.7:** Extend `make_train_step` to include:
>
> 1. Gradient clipping (to prevent exploding gradients)
> 2. Learning rate scheduling (to decay the learning rate)
> 3. Logging (to record the loss and other metrics)
> 4. Mixed precision training (to use float16 for speed)

---

## 3.8 Research Application: Gradient Checking

### 3.8.1 What Is Gradient Checking?

**Gradient checking** is the process of verifying that your analytical gradients match numerical gradients. This is essential when implementing custom autograd functions or debugging training issues.

The numerical gradient is computed using finite differences:

$$
\frac{\partial f}{\partial x_i} \approx \frac{f(x_i + \epsilon) - f(x_i - \epsilon)}{2\epsilon}
$$

The analytical gradient is computed by autograd. The two should match to within a small tolerance.

### 3.8.2 Implementing Gradient Checking

```python
def gradient_check(f, x, epsilon=1e-5):
    """Check analytical gradient against numerical gradient."""
    # Analytical gradient
    x = x.clone().detach().requires_grad_(True)
    y = f(x)
    y.backward()
    analytical = x.grad.clone()

    # Numerical gradient
    numerical = torch.zeros_like(x)
    for i in range(x.numel()):
        x_plus = x.clone().detach()
        x_minus = x.clone().detach()
        x_plus.view(-1)[i] += epsilon
        x_minus.view(-1)[i] -= epsilon
        numerical.view(-1)[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)

    # Compare
    diff = (analytical - numerical).abs().max()
    print(f"Max difference: {diff:.2e}")
    return diff < 1e-4

# Test
def f(x):
    return (x ** 3).sum()

x = torch.tensor([1.0, 2.0, 3.0])
gradient_check(f, x)
```

### 3.8.3 Research Relevance

Gradient checking is used when:
- Implementing custom autograd functions (`torch.autograd.Function`)
- Debugging training issues (e.g., loss not decreasing)
- Verifying that a new architecture is correctly implemented
- Reproducing a paper's results

**Research relevance:** A paper that reports gradient checking is more trustworthy than one that does not. Include gradient checking in your capstone project.

> **Exercise 3.8:** Implement gradient checking for the following functions:
>
> 1. $ f(x) = \log(x) $
> 2. $ f(x) = \sigma(x) = \frac{1}{1 + e^{-x}} $
> 3. $ f(x) = \tanh(x) $
> 4. $ f(x) = \text{softmax}(x) $

---

## 3.9 Common Autograd Errors and Debugging

### 3.9.1 Error: `RuntimeError: element 0 of tensors does not require grad`

This error occurs when you call `.backward()` on a tensor that does not require grad. The fix is to ensure that the input tensors have `requires_grad=True`.

```python
# Error
x = torch.tensor([1.0, 2.0, 3.0])
y = (x ** 2).sum()
# y.backward()  # Error

# Fix
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(f"x.grad: {x.grad}")
```

### 3.9.2 Error: `RuntimeError: Trying to backward through the graph a second time`

This error occurs when you call `.backward()` twice on the same graph without retaining it. The fix is to use `retain_graph=True` or to recompute the graph.

```python
# Error
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
# y.backward()  # Error

# Fix
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward(retain_graph=True)
y.backward()
print(f"x.grad: {x.grad}")  # Accumulated
```

### 3.9.3 Error: `RuntimeError: a leaf Variable that requires grad has been used in an in-place operation`

This error occurs when you modify a leaf tensor in-place. The fix is to use `torch.no_grad()` or avoid in-place operations.

```python
# Error
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# x += 1  # Error

# Fix
with torch.no_grad():
    x += 1
print(f"x: {x}")
```

### 3.9.4 Error: `RuntimeError: grad can be implicitly created only for scalar outputs`

This error occurs when you call `.backward()` on a non-scalar tensor. The fix is to reduce to a scalar first.

```python
# Error
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
# y.backward()  # Error

# Fix
y.sum().backward()
print(f"x.grad: {x.grad}")
```

### 3.9.5 Summary: Common Autograd Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `does not require grad` | Tensor does not have `requires_grad=True` | Set `requires_grad=True` |
| `backward through the graph a second time` | Calling `.backward()` twice | Use `retain_graph=True` |
| `in-place operation` | Modifying leaf tensor in-place | Use `torch.no_grad()` |
| `scalar outputs` | `.backward()` on non-scalar | Reduce to scalar first |
| `NoneType` | Reassigning a leaf tensor | Use `torch.no_grad()` |

> **Exercise 3.9:** For each of the following errors, identify the cause and fix it:
>
> ```python
> # Error 1
> x = torch.tensor([1.0, 2.0, 3.0])
> y = (x ** 2).sum()
> y.backward()
> ```
>
> ```python
> # Error 2
> x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
> y = x ** 2
> y.backward()
> ```
>
> ```python
> # Error 3
> x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
> x += 1
> ```
>
> ```python
> # Error 4
> x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
> y = (x ** 2).sum()
> y.backward()
> y.backward()
> ```

---

## 3.10 Worked Example: Full Training Pipeline with Autograd

Let us combine everything we have learned into a complete training pipeline.

### 3.10.1 Data Generation

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

### 3.10.2 Model Definition

```python
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.a + self.b * x

model = LinearRegression().to(device)
print(model.state_dict())
```

### 3.10.3 Loss and Optimizer

```python
loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=1e-1)
```

### 3.10.4 Training Loop

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

print(f"Final model: a = {model.a.item():.4f}, b = {model.b.item():.4f}")
```

### 3.10.5 Reflection

This pipeline illustrates the full autograd workflow. The model is defined as a subclass of `nn.Module`, with parameters registered as `nn.Parameter`. The loss and optimizer are created. The training loop performs the forward pass, computes the loss, calls `.backward()` to compute gradients, and calls `optimizer.step()` to update parameters. The validation loop uses `torch.no_grad()` to disable gradient tracking.

**Research relevance:** This is the canonical training pipeline in PyTorch. Every research project you run will follow this structure, with modifications for the specific model, data, and task.

> **Exercise 3.10:** Extend the pipeline above to include:
>
> 1. Early stopping (stop training when validation loss stops improving)
> 2. Learning rate scheduling (reduce learning rate on plateau)
> 3. Model checkpointing (save the best model)
> 4. Logging (save the loss curves to a file)

---

## 3.11 Module Summary

| Concept | Mathematical Form | PyTorch | Research Relevance |
|---------|-------------------|---------|-------------------|
| Derivative | $ f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} $ | Numerical approximation | Gradient checking |
| Partial derivative | $ \frac{\partial f}{\partial x_i} $ | `.grad` | Parameter sensitivity |
| Gradient | $ \nabla f = [\partial f / \partial x_i] $ | `.grad` | Optimization |
| Chain rule | $ \frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx} $ | `.backward()` | Backpropagation |
| Jacobian | $ J_{ij} = \frac{\partial y_i}{\partial x_j} $ | `torch.autograd.functional.jacobian` | Sensitivity analysis |
| Autograd | Automatic differentiation | `requires_grad=True` | All deep learning |
| Computation graph | DAG of operations | Built automatically | Debugging, understanding |
| Dynamic graph | Built on the fly | Default in PyTorch | Flexibility |
| Detach | Break the graph | `.detach()` | Target networks, stop-gradient |
| In-place operations | Modify without new allocation | `x += 1` | Memory efficiency |
| Gradient accumulation | $ \sum_i \nabla L_i $ | Default | Large batches |
| Gradient zeroing | $ \nabla L \leftarrow 0 $ | `.zero_grad()` | Prevent accumulation |
| No-grad mode | — | `torch.no_grad()` | Inference |
| Optimizer | $ \theta \leftarrow \theta - \eta \nabla L $ | `optim.SGD`, `optim.Adam` | Training |
| Loss function | $ L(y, \hat{y}) $ | `nn.MSELoss`, `nn.CrossEntropyLoss` | Objective |
| Training step | Forward, backward, update | `make_train_step` | Reusability |
| Gradient checking | Numerical vs. analytical | Finite differences | Debugging |

---

## 3.12 Capstone Thread: Autograd Skills for Your Project

By the end of this module, you should be able to compute gradients of scalar functions with respect to tensor inputs; understand and visualize computation graphs; diagnose and fix common autograd errors; implement manual gradient descent and compare it against PyTorch's autograd; and implement a full training pipeline with autograd.

For your capstone project, you will need these skills to implement your model's training loop, debug gradient issues, verify your gradients with gradient checking, and ensure reproducibility.

In Module 4, we will build on this foundation to study **model abstraction** with `nn.Module` — the base class for all PyTorch models.

---

## 3.13 Additional Resources

### On Automatic Differentiation

- Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). "Automatic Differentiation in Machine Learning: A Survey." *JMLR*, 18(153), 1–43.
- Griewank, A., & Walther, A. (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation* (2nd ed.). SIAM.

### On Backpropagation

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning Representations by Back-Propagating Errors." *Nature*, 323(6088), 533–536.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 6: Deep Feedforward Networks.

### On Optimization

- Ruder, S. (2016). "An Overview of Gradient Descent Optimization Algorithms." *arXiv:1609.04747*.
- Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization." *ICLR*.

### On Gradient Checking

- CS231n notes on gradient checking: [Link](https://cs231n.github.io/neural-networks-3/#gradcheck)
- PyTorch `torch.autograd.gradcheck`: [Link](https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html)

### On Computation Graphs

- PyTorch autograd documentation: [Link](https://pytorch.org/docs/stable/notes/autograd.html)
- `torchviz` GitHub repository: [Link](https://github.com/szagoruyko/pytorchviz)

---

*End of Module 3*