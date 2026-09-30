# Module 2: Tensor Computing with Research Intent

## TORA | CS 336: PyTorch for Research

**Instructor:** Alpha Alimamy Kamara

**Module Length:** 2 weeks (4 lectures + 2 lab sessions)

**Prerequisites:** Module 1 completed; familiarity with linear algebra (vectors, matrices, matrix multiplication); basic Python and NumPy.

## 2.0 Module Overview

In Module 1, we established that research is distinguished from engineering by its commitment to knowledge generation, reproducibility, and falsifiability. Now we begin the technical work.

This module is about **tensors**, the fundamental data structure of PyTorch and, by extension, of modern deep learning research. We will not treat tensors as mere containers for numbers. We will treat them as mathematical objects with algebraic structure, geometric interpretation, and computational properties that have profound implications for research.

The pedagogical approach in this module 
is **mathematics first, code second**.
For every tensor construct, we will write 
the mathematical object, explicitly the 
scalar, the vector, the matrix, the tensor, and only then translate it into PyTorch. This mirrors how research is actually conducted: you have a mathematical model in mind, and PyTorch is the instrument that realizes it.

By the end of this module, you will be able to create tensors of arbitrary shape, dtype, and device; reason about tensor operations algebraically and geometrically; diagnose and fix the three most common tensor errors in research code; write reproducible tensor code with controlled randomness; and articulate *why* each tensor operation matters for your research question.

The module is self-paced. Each section contains mathematical foundations, practical implementations, worked examples, and exercises. Do not skip the exercises — they are where the learning happens.



## 2.1 Tensors as Numerical Abstractions

### 2.1.1 The Mathematical Definition

A **tensor** is a multilinear map from a product of 
vector spaces to the real numbers. More concretely, 
a tensor of **order** $n$ (also called **rank** or **ndim**) is an element of the tensor product space:

$$T \in \underbrace{V \otimes V \otimes \cdots \otimes V}_{n \text{ times}}$$

where $V$ is a vector space (typically $\mathbb{R}^d$). In coordinates, a tensor is represented by a multidimensional array of numbers:

$$T_{i_1 i_2 \cdots i_n} \in \mathbb{R}$$

where each index $i_k$ ranges over $1, \ldots, d_k$.

This definition is abstract. For our purposes, 
the following operational definition suffices: 
**a tensor is a multidimensional array of numbers, 
equipped with algebraic operations 
(addition, multiplication, contraction) that 
generalize matrix algebra.** The **order** of a
tensor is the number of indices (dimensions) it has. 
The **shape** is the tuple $(d_1, d_2, \ldots, d_n)$ 
giving the size of each dimension.

### 2.1.2 The Hierarchy of Tensors

The following table summarizes the special cases of tensors that arise most frequently in research.

| Name | Order | Shape                | Mathematical Object                                     | Example |
|------|-----|----------------------|---------------------------------------------------------|---------|
| Scalar | 0 | $()$                 | $a \in \mathbb{R}$                                      | Loss value |
| Vector | 1 | $(d,)$               | $ \mathbf{x} \in \mathbb{R}^d$                          | Feature vector |
| Matrix | 2 |$ (m, n)$              | $A \in \mathbb{R}^{m \times n}$                         | Weight matrix |
| 3-tensor | 3 | $(d_1, d_2, d_3) $   | $T \in \mathbb{R}^{d_1 \times d_2 \times d_3}$          | Image $(C, H, W)$ |
| 4-tensor | 4 | $(d_1, d_2, d_3, d_4)$ | $T \in \mathbb{R}^{d_1 \times d_2 \times d_3 \times d_4}$ | Batch of images $(N, C, H, W)$ |
| $ n $-tensor |$ n $| $ (d_1, \ldots, d_n) $ | $ T \in \mathbb{R}^{d_1 \times \cdots \times d_n} $      | Attention tensors |

> **Research note:** In the deep learning literature, the term "tensor" is often used loosely to mean "multidimensional array." The strict mathematical definition (multilinear map) is rarely invoked. However, understanding the mathematical definition helps clarify why certain operations (like contraction) are natural and others (like reshaping) are not.

### 2.1.3 Why Tensors Matter for Research

Tensors are not merely a convenient data structure. They are the **mathematical objects** that neural networks manipulate. A neural network is, at its core, a composition of tensor operations:

$$f(\mathbf{x}) = \sigma_n(W_n \sigma_{n-1}(W_{n-1} \cdots \sigma_1(W_1 \mathbf{x} + \mathbf{b}_1) \cdots + \mathbf{b}_{n-1}) + \mathbf{b}_n)$$

Each $W_i$ is a matrix (2-tensor), each $\mathbf{b}_i$ is a vector (1-tensor), and the intermediate activations are vectors or higher-order tensors (for convolutional and attention layers).

When you debug a neural network, you are debugging tensor operations. When you optimize a neural network, you are optimizing tensor operations. When you interpret a neural network, you are interpreting tensor operations. The tensor is the atom of deep learning.

### 2.1.4 The Three Questions Applied to Tensors

Recall from Module 1 the three questions of research. For tensors, the question is: *What is the mathematical structure of the data, and what operations preserve or exploit that structure?* The evidence is: *Does the tensor operation produce the expected mathematical result, and does it improve the model's performance?* The meaning is: *What does the tensor operation reveal about the data or the model?* We will return to these questions throughout the module.


## 2.2 Creating Tensors: The Practical Foundations

### 2.2.1 Installation and Import

Before we proceed, ensure PyTorch is installed. In a research environment, we recommend using a conda environment or a Docker container for reproducibility.

```bash
# Create a conda environment
conda create -n pytorch-research python=3.10
conda activate pytorch-research

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### 2.2.2 Scalars: The Simplest Tensors

A **scalar** is a 0-dimensional tensor. Mathematically, it is an element of $ \mathbb{R} $:
$$a \in \mathbb{R}, \quad \text{ndim}(a) = 0, \quad \text{shape}(a) = ()$$

The scalar $ a = 7 $ is written mathematically as:

$$a = 7$$

In PyTorch, this becomes:

```python
# Creating a scalar
scalar = torch.tensor(7)
print(f"Scalar: {scalar}")
print(f"ndim: {scalar.ndim}")
print(f"shape: {scalar.shape}")
print(f"dtype: {scalar.dtype}")

# Extracting the Python number
print(f"Item: {scalar.item()}")
print(f"Type of item: {type(scalar.item())}")
```

The output is `tensor(7)`, indicating a 0-dimensional tensor containing the value 7. The `.item()` method extracts the Python number.

**Research relevance:** Scalars are ubiquitous in research. The loss value is a scalar. The learning rate is a scalar. The accuracy is a scalar. Understanding that these are 0-dimensional tensors (not Python floats) matters when you want to backpropagate through them or log them.

**Worked example 2.1:** Consider the scalar $ a = 89$. 
Write it mathematically, then translate to PyTorch.

Mathematically:

$$a = 89$$

In PyTorch:

```python
scalar_0 = torch.tensor(89)
print(scalar_0)  # tensor(89)
print(scalar_0.ndim)  # 0
```

### 2.2.3 Vectors: 1-Dimensional Tensors

A **vector** is a 1-dimensional tensor. Mathematically, it is an element of $\mathbb{R}^d $:

$$\mathbf{x} \in \mathbb{R}^d, \quad \text{ndim}(\mathbf{x}) = 1, \quad \text{shape}(\mathbf{x}) = (d,)$$

The vector $\mathbf{x} = [7, 7]$ is written mathematically as:

$$\mathbf{x} = \begin{bmatrix} 7 \\ 7 \end{bmatrix} \in \mathbb{R}^2$$

In PyTorch, this becomes:

```python
# Creating a vector
vector = torch.tensor([7, 7])
print(f"Vector: {vector}")
print(f"ndim: {vector.ndim}")
print(f"shape: {vector.shape}")
```

The output is `tensor([7, 7])`, indicating a 1-dimensional tensor with two elements. 
The number of dimensions is 1, and the shape 
is `torch.Size([2])`.

**Worked example 2.2:** Consider the feature vector
$\mathbf{x} = [1.0, 2.0, 3.0, 4.0, 5.0] $. 
Write it mathematically, then translate to PyTorch.

Mathematically:

$$
\mathbf{x} = \begin{bmatrix} 1.0 \\ 2.0 \\ 3.0 \\ 4.0 \\ 5.0 \end{bmatrix} \in \mathbb{R}^5
$$
In PyTorch:

```python
features = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print(features)  # tensor([1., 2., 3., 4., 5.])
print(features.shape)  # torch.Size([5])
```

**Research relevance:** Feature vectors, word embeddings, hidden states, and gradient vectors are all 1-dimensional tensors. When you compute the gradient of a loss with respect to a parameter vector, you get a vector of the same shape.

### 2.2.4 Matrices: 2-Dimensional Tensors

A **matrix** is a 2-dimensional tensor. Mathematically, it is an element of $\mathbb{R}^{m \times n}$:

$$A \in \mathbb{R}^{m \times n}, \quad \text{ndim}(A) = 2, \quad \text{shape}(A) = (m, n)$$

The matrix

$$A = \begin{bmatrix} 7 & 8 \\ 9 & 10 \end{bmatrix} \in \mathbb{R}^{2 \times 2}$$

is written in PyTorch as:

```python
# Creating a matrix
MATRIX = torch.tensor([[7, 8],
                       [9, 10]])
print(f"Matrix:\n{MATRIX}")
print(f"ndim: {MATRIX.ndim}")
print(f"shape: {MATRIX.shape}")
```

The output is:

```
tensor([[ 7,  8],
        [ 9, 10]])
```

with `ndim = 2` and `shape = torch.Size([2, 2])`.

**Worked example 2.3:** Consider the weight matrix

$$W = \begin{bmatrix} 0.1 & 0.2 & 0.3 \\ 0.4 & 0.5 & 0.6 \\ 0.7 & 0.8 & 0.9 \\ 1.0 & 1.1 & 1.2 \end{bmatrix} \in \mathbb{R}^{4 \times 3}$$

Write it in PyTorch:

```python
W = torch.tensor([[0.1, 0.2, 0.3],
                  [0.4, 0.5, 0.6],
                  [0.7, 0.8, 0.9],
                  [1.0, 1.1, 1.2]])
print(W.shape)  # torch.Size([4, 3])
```

**Research relevance:** Weight matrices are
the learnable parameters of linear layers.
The matrix multiplication $\mathbf{x} + 
\mathbf{b}$ is the fundamental 
operation of a neural network layer. 
Understanding the shape constraints of matrix multiplication is essential for debugging shape errors.

### 2.2.5 Higher-Order Tensors

Tensors of order 3 and above are common in research. The most important is the **4-tensor** used for batches of images:

$$X \in \mathbb{R}^{N \times C \times H \times W}$$

where $N$ is the batch size, $C$ is the number of channels, $H$ is the height, and $W$ is the width.

Consider the 3-tensor

$$T = \begin{bmatrix} \begin{bmatrix} 1 & 2 & 3 \\ 3 & 6 & 9 \\ 2 & 4 & 5 \end{bmatrix} \end{bmatrix} \in \mathbb{R}^{1 \times 3 \times 3}$$

In PyTorch:

```python
# Creating a 3-tensor
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print(f"3-tensor:\n{TENSOR}")
print(f"ndim: {TENSOR.ndim}")
print(f"shape: {TENSOR.shape}")
```

The output is:

```
tensor([[[1, 2, 3],
         [3, 6, 9],
         [2, 4, 5]]])
```

with `ndim = 3` and `shape = torch.Size([1, 3, 3])`. 
The dimensions go outer to inner: there is 1 dimension of 3 by 3.

**Worked example 2.4:** Consider a batch of 32 RGB
images, each of size $224 \times 224$. 
Write the mathematical object, then translate
to PyTorch.

Mathematically:

$$X \in \mathbb{R}^{32 \times 3 \times 224 \times 224}$$

In PyTorch:

```python
batch_images = torch.randn(32, 3, 224, 224)
print(f"Batch of images shape: {batch_images.shape}")
print(f"  N (batch size): {batch_images.shape[0]}")
print(f"  C (channels): {batch_images.shape[1]}")
print(f"  H (height): {batch_images.shape[2]}")
print(f"  W (width): {batch_images.shape[3]}")
```

**Research relevance:** The 4-tensor $(N, C, H, W)$ is the standard input format for convolutional neural networks.
The 3-tensor $(L, N, d)$ is the standard format for 
transformer inputs (sequence length, batch size, 
embedding dimension). Understanding the shape
conventions of your architecture is the first step
in debugging.

### 2.2.6 Summary: Creating Tensors

| Creation Method | Description                     | Mathematical Object | Example | Use Case |
|-----------------|---------------------------------|------------------|---------|----------|
| `torch.tensor(data)` | From Python list or NumPy array | Any              | `torch.tensor([1, 2, 3])` | General-purpose |
| `torch.rand(size)` | Uniform random in [0, 1]        | $X \sim \mathcal{U}(0,1)$ | `torch.rand(3, 4)` | Weight initialization |
| `torch.randn(size)` | Standard normal                 | $ X \sim \mathcal{N}(0,1)$ | `torch.randn(3, 4)` | Weight initialization |
| `torch.zeros(size)` | All zeros                       | $\mathbf{0}$      | `torch.zeros(3, 4)` | Bias initialization, masking |
| `torch.ones(size)` | All ones                        | $\mathbf{1}$     | `torch.ones(3, 4)` | Scaling, masking |
| `torch.arange(start, end, step)` | Range of values                 | $\{a, a+s, \ldots, b-s\} $ | `torch.arange(0, 10, 1)` | Indexing, positional encodings |
| `torch.linspace(start, end, steps)` | Linearly spaced values          | $\{a + i\frac{b-a}{n-1}\} $ | `torch.linspace(0, 1, 5)` | Interpolation, schedules |
| `torch.eye(n)` | Identity matrix                 | $ I_n $          | `torch.eye(3)` | Linear algebra |
| `torch.empty(size)` | Uninitialized memory            | —                | `torch.empty(3, 4)` | Performance-critical code |
| `torch.full(size, fill_value)` | Constant value                  | $ c \cdot \mathbf{1}$ | `torch.full((3, 4), 7.0)` | Constants |

> **Exercise 2.1:** Create a tensor for each of the following mathematical objects. For each, print the shape, ndim, and dtype.
>
> 1. A scalar $ a = 3.14 \)
> 2. A vector $ \mathbf{x} \in \mathbb{R}^5 $ with values $[1, 2, 3, 4, 5]$
> 3. A matrix $ A \in \mathbb{R}^{3 \times 3} $ with all entries equal to 1
> 4. A 4-tensor $ X \in \mathbb{R}^{16 \times 3 \times 32 \times 32} $ with random normal entries
> 5. A tensor $ T \in \mathbb{R}^{2 \times 3 \times 4} $ with all entries equal to 0


## 2.3 Tensor Attributes: Shape, Dtype, Device

### 2.3.1 The Three Questions Every Researcher Must Ask

Recall the "what, what, where" framework 
from the original notebook: *What shape 
are my tensors? What datatype are they? 
Where are they stored?* These three 
attributes: **shape**, **dtype**, 
and **device** are the source of the vast majority of errors in PyTorch research code.

### 2.3.2 Shape

The **shape** of a tensor is a tuple giving 
the size of each dimension:

$$\text{shape}(T) = (d_1, d_2, \ldots, d_n)$$


The **number of elements** is the product of the dimensions:

$$|T| = \prod_{i=1}^{n} d_i$$

For a tensor $ X \in \mathbb{R}^{3 \times 4 \times 5}$, the shape is $(3, 4, 5) $ and the number of elements
is $3 \times 4 \times 5 = 60 $. In PyTorch:

```python
x = torch.randn(3, 4, 5)
print(f"Shape: {x.shape}")
print(f"Number of elements: {x.numel()}")
print(f"Product of dimensions: {3 * 4 * 5}")
```

**Shape constraints in research:**

| Operation | Shape Constraint | Mathematical Rule                          | Example |
|-----------|------------------|--------------------------------------------|---------|
| Element-wise addition | Shapes must be broadcastable | $ (m, n) + (m, n) $                        | $(3, 4) + (3, 4) $✓ |
| Matrix multiplication | Inner dimensions must match | $(m, n) @ (n, p) \to (m, p) $              | $ (3, 4) @ (4, 5) \to (3, 5) $ |
| Concatenation | All dimensions except the concatenation axis must match | \( [(m, n_1), (m, n_2)] \to (m, n_1 + n_2) $ | `torch.cat([(3, 4), (3, 5)], dim=1)` |
| Reshaping | Total number of elements must be preserved | $\prod d_i = \prod e_j $                    | $(3, 4) \to (2, 6) $ ✓ |
| Broadcasting | Dimensions must be equal or one of them must be 1 | $(m, 1) + (1, n) \to (m, n) $              | $(3, 1) + (1, 4) \to (3, 4)$ |

### 2.3.3 Dtype

The **dtype** (data type) specifies how each element is stored in memory. The most common dtypes in research are:

| Dtype | Bits | Range                   | Precision | Use Case |
|-------|------|-------------------------|-----------|----------|
| `torch.float32` | 32 | $ \pm 3.4 \times 10^{38} $ | ~7 decimal digits | Default for most research |
| `torch.float16` | 16 | $\pm 6.5 \times 10^{4} $ | ~3 decimal digits | Mixed precision training |
| `torch.bfloat16` | 16 | $ \pm 3.4 \times 10^{38} $ | ~3 decimal digits | Mixed precision (better range) |
| `torch.float64` | 64 | $\pm 1.8 \times 10^{308} $ | ~15 decimal digits | Scientific computing |
| `torch.int64` | 64 | $ \pm 9.2 \times 10^{18}$ | Exact | Indexing, token IDs |
| `torch.int32` | 32 | $\pm 2.1 \times 10^{9} $ | Exact | General integers |
| `torch.bool` | 8 | True/False              | Exact | Masks, conditions |

**Precision in computing:** The precision of a floating-point number is the number of significant digits it can represent. Higher precision means more accurate computations but more memory and slower operations:

$$\text{float32} \approx 7 \text{ decimal digits}, \quad \text{float16} \approx 3 \text{ decimal digits}$$

In PyTorch:

```python
# Dtype examples
float_32 = torch.tensor([3.0, 6.0, 9.0])
print(f"Default dtype: {float_32.dtype}")

float_16 = torch.tensor([3.0, 6.0, 9.0], dtype=torch.float16)
print(f"Float16 dtype: {float_16.dtype}")

int_64 = torch.tensor([3, 6, 9])
print(f"Int64 dtype: {int_64.dtype}")

# Dtype conversion
converted = float_32.type(torch.float16)
print(f"Converted: {converted.dtype}")
```

**Research relevance:** Mixed precision training (using float16 for forward/backward passes and float32 for parameter updates) can speed up training by 2–3× with minimal accuracy loss. This is a standard technique in large-scale research. We will cover it in Module 16.

### 2.3.4 Device

The **device** specifies where the tensor is stored: CPU or GPU. In PyTorch:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Create tensor on device
x = torch.randn(3, 4, device=device)
print(f"Tensor device: {x.device}")

# Move tensor to device
x_cpu = torch.randn(3, 4)
x_gpu = x_cpu.to(device)
print(f"CPU tensor device: {x_cpu.device}")
print(f"GPU tensor device: {x_gpu.device}")
```

**Device rules in research:** Operations between tensors on different devices will error. Models and their inputs must be on the same device. `.to(device)` returns a copy, not a view. NumPy arrays cannot be on GPU.

```python
# Device mismatch error (will error)
x_cpu = torch.randn(3, 4)
x_gpu = torch.randn(3, 4, device=device)
# x_cpu + x_gpu  # RuntimeError: Expected all tensors to be on the same device

# Correct approach
x_cpu = x_cpu.to(device)
result = x_cpu + x_gpu
print(f"Result device: {result.device}")
```

### 2.3.5 The Three Attributes

| Attribute | Question | Common Errors | Debugging Strategy |
|-----------|----------|---------------|-------------------|
| Shape | What are the dimensions? | Matrix multiplication mismatch; reshape failure | Print shapes before every operation |
| Dtype | How are elements stored? | Float32 vs. Float64; Int vs. Float | Use `tensor.dtype`; cast explicitly |
| Device | Where is it stored? | CPU vs. GPU mismatch | Use `tensor.device`; move with `.to(device)` |

> **Exercise 2.2:** For each of the following code snippets, identify the error (if any) and fix it.
>
> ```python
> # Snippet 1
> a = torch.randn(3, 4)
> b = torch.randn(4, 5)
> c = a + b
> ```
>
> ```python
> # Snippet 2
> a = torch.tensor([1.0, 2.0, 3.0])
> b = torch.tensor([1, 2, 3])
> c = a + b
> ```
>
> ```python
> # Snippet 3
> device = "cuda"
> a = torch.randn(3, 4).to(device)
> b = torch.randn(3, 4)
> c = a + b
> ```
>
> ```python
> # Snippet 4
> a = torch.randn(3, 4)
> b = a.reshape(2, 6)
> print(b.shape)
> ```
>
> ```python
> # Snippet 5
> a = torch.randn(3, 4).to("cuda")
> b = a.numpy()
> ```

---

## 2.4 Randomness and Reproducibility

### 2.4.1 The Role of Randomness in Deep Learning

Randomness is ubiquitous in deep learning. Weights are initialized randomly (e.g., Xavier, Kaiming, or simple Gaussian). Training data is shuffled each epoch. Dropout randomly drops neurons during training. Data augmentation applies random transformations to inputs. Stochastic gradient descent samples mini-batches randomly.

Without randomness, neural networks would not train effectively. But uncontrolled randomness makes experiments irreproducible.

### 2.4.2 Pseudorandomness and Seeds

Computers generate **pseudorandom** numbers using deterministic algorithms. A **seed** is an integer that initializes the pseudorandom number generator (PRNG). Given the same seed, the PRNG produces the same sequence of numbers:

$$\text{PRNG}(s) = (r_1, r_2, r_3, \ldots)$$

where $s$ is the seed and $r_i$ are the pseudorandom numbers.

```python
import torch
import numpy as np
import random

# Without seed: different results each time
print("Without seed:")
print(torch.rand(3))
print(torch.rand(3))

# With seed: same results each time
print("\nWith seed:")
torch.manual_seed(42)
print(torch.rand(3))
torch.manual_seed(42)
print(torch.rand(3))
```

### 2.4.3 Setting Seeds in PyTorch

For full reproducibility, you must set seeds for all sources of randomness:

```python
def set_seed(seed=42):
    """Set seeds for reproducibility across all libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

| Seed Function | What It Controls |
|---------------|------------------|
| `torch.manual_seed(seed)` | PyTorch CPU random operations |
| `torch.cuda.manual_seed(seed)` | PyTorch GPU random operations (current GPU) |
| `torch.cuda.manual_seed_all(seed)` | PyTorch GPU random operations (all GPUs) |
| `np.random.seed(seed)` | NumPy random operations |
| `random.seed(seed)` | Python random operations |
| `torch.backends.cudnn.deterministic = True` | Makes cuDNN operations deterministic |
| `torch.backends.cudnn.benchmark = False` | Disables cuDNN auto-tuning (which is non-deterministic) |

### 2.4.4 The Determinism Trade-off

Setting `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` makes your code reproducible but may slow it down. The `benchmark` mode auto-tunes convolution algorithms for your hardware, which is faster but non-deterministic. For research, the trade-off is usually worth it: reproducibility is more important than a 10–20% speedup.

> **Exercise 2.3:** Write a function `set_seed(seed)` that sets all seeds. Then run the following experiment:
>
> 1. Without setting seeds, create 3 random tensors of shape (3, 4) and compute their pairwise differences.
> 2. Set seed to 42, then create 3 random tensors and compute their pairwise differences.
> 3. Set seed to 42 again, then create 3 random tensors and compute their pairwise differences.
> 4. Compare the results of steps 2 and 3. Are they identical?
> 5. Compare the results of steps 1 and 2. Are they different?

## 2.5 Mathematical Operations on Tensors

### 2.5.1 Element-wise Operations

Given two tensors $ A, B \in \mathbb{R}^{d_1 \times \cdots \times d_n}$ of the same shape, the element-wise sum is:

$$(A + B)_{i_1 \cdots i_n} = A_{i_1 \cdots i_n} + B_{i_1 \cdots i_n}
$$

Similarly for element-wise subtraction, multiplication, and division. For example, with

$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
$$

the element-wise sum is:

$$
A + B = \begin{bmatrix} 1+5 & 2+6 \\ 3+7 & 4+8 \end{bmatrix} = \begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}
$$

In PyTorch:

```python
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

print(f"a + b:\n{a + b}")
print(f"a - b:\n{a - b}")
print(f"a * b:\n{a * b}")
print(f"a / b:\n{a / b}")
```

**Scalar operations:** A scalar \( c \in \mathbb{R} \) is broadcast across the tensor:

$$
(c \cdot A)_{i_1 \cdots i_n} = c \cdot A_{i_1 \cdots i_n}
$$

For example, with $c = 10$:
$$
10 \cdot A = \begin{bmatrix} 10 & 20 \\ 30 & 40 \end{bmatrix}
$$

In PyTorch:

```python
a = torch.tensor([1.0, 2.0, 3.0])
print(f"a + 10 = {a + 10}")
print(f"a * 10 = {a * 10}")
```

### 2.5.2 Broadcasting

Broadcasting allows operations between tensors of different shapes by virtually expanding the smaller tensor. The broadcasting rules are: if the tensors have different numbers of dimensions, prepend 1s to the smaller tensor's shape; for each dimension, the sizes must be equal, or one of them must be 1; the result has the size of the larger tensor in each dimension.

Mathematically, if $A \in \mathbb{R}^{m \times 1}$ and \( B \in \mathbb{R}^{1 \times n} \), then:

$$
(A + B)_{ij} = A_{i1} + B_{1j}
$$

resulting in a tensor of shape $(m, n) $. For example, with

$$A = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} 
\in \mathbb{R}^{3 \times 1}, \quad B = \begin{bmatrix} 
10 & 20 & 30 & 40 \end{bmatrix} \in
\mathbb{R}^{1 \times 4}$$

the broadcast sum is:

$$A + B = \begin{bmatrix} 1+10 & 1+20 & 1+30 & 1+40 
\\ 2+10 & 2+20 & 2+30 & 2+40 
\\ 3+10 & 3+20 & 3+30 & 3+40 \end{bmatrix} = 
\begin{bmatrix} 11 & 21 & 31 & 41 \\ 12
& 22 & 32 & 42 \\ 13 & 23 & 33 & 43
\end{bmatrix} \in \mathbb{R}^{3 \times 4}$$

In PyTorch:

```python
a = torch.tensor([[1.0], [2.0], [3.0]])  # Shape (3, 1)
b = torch.tensor([[10.0, 20.0, 30.0, 40.0]])  # Shape (1, 4)

c = a + b  # Shape (3, 4)
print(f"a shape: {a.shape}")
print(f"b shape: {b.shape}")
print(f"c shape: {c.shape}")
print(f"c:\n{c}")
```

**Research relevance:** Broadcasting is used extensively in adding a bias vector to a batch of activations `(N, d) + (d,)`, computing pairwise distances `(N, 1, d) - (1, M, d)`, and normalizing by a per-channel mean `(N, C, H, W) - (1, C, 1, 1)`.

### 2.5.3 Matrix Multiplication

Given $ A \in \mathbb{R}^{m \times n} $ and $B \in \mathbb{R}^{n \times p}$, 
the matrix product is:

$$
(AB)_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}
$$

The result is $C \in \mathbb{R}^{m \times p}$. 
The shape rule is: the inner dimensions must match, 
$(m, n) @ (n, p) \to (m, p)$.

For example, with

$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} \in \mathbb{R}^{3 \times 2}, \quad B = \begin{bmatrix} 7 & 8 & 9 \\ 10 & 11 & 12 \end{bmatrix} \in \mathbb{R}^{2 \times 3}$$

the product is:

$$
C = AB = \begin{bmatrix} 1 \cdot 7 + 2 \cdot 10 & 1 \cdot 8 + 2 \cdot 11 & 1 \cdot 9 + 2 \cdot 12 \\ 3 \cdot 7 + 4 \cdot 10 & 3 \cdot 8 + 4 \cdot 11 & 3 \cdot 9 + 4 \cdot 12 \\ 5 \cdot 7 + 6 \cdot 10 & 5 \cdot 8 + 6 \cdot 11 & 5 \cdot 9 + 6 \cdot 12 \end{bmatrix} = \begin{bmatrix} 27 & 30 & 33 \\ 61 & 68 & 75 \\ 95 & 106 & 117 \end{bmatrix} \in \mathbb{R}^{3 \times 3}
$$

In PyTorch:

```python
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])  # Shape (3, 2)

B = torch.tensor([[7.0, 8.0, 9.0],
                  [10.0, 11.0, 12.0]])  # Shape (2, 3)

C = torch.matmul(A, B)  # Shape (3, 3)
print(f"C:\n{C}")
```

**Worked example 2.5:** Compute the matrix product $AB$ where

$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \in \mathbb{R}^{2 \times 3}, \quad B = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \in \mathbb{R}^{3 \times 2}
$$

Mathematically:
$$
AB = \begin{bmatrix} 1 \cdot 1 + 2 \cdot 0 + 3 \cdot 1 & 1 \cdot 0 + 2 \cdot 1 + 3 \cdot 1 \\ 4 \cdot 1 + 5 \cdot 0 + 6 \cdot 1 & 4 \cdot 0 + 5 \cdot 1 + 6 \cdot 1 \end{bmatrix} = \begin{bmatrix} 4 & 5 \\ 10 & 11 \end{bmatrix}
$$

In PyTorch:

```python
A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
B = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
C = A @ B
print(C)  # tensor([[ 4.,  5.], [10., 11.]])
```

**Research relevance:** Matrix multiplication is 
the dominant operation in neural networks. A linear
layer computes $Y = XW^T + b$, where 
$ X \in \mathbb{R}^{N \times d_{\text{in}}}$, 
$W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, and $ Y \in \mathbb{R}^{N \times d_{\text{out}}}$.

### 2.5.4 Transpose

The transpose of $A \in \mathbb{R}^{m \times n} $ is 
$A^T \in \mathbb{R}^{n \times m}$, defined by:

$$A^T_{ij} = A_{ji}$$

For example, with
$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \in \mathbb{R}^{2 \times 3}
$$

the transpose is:
$$
A^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix} \in \mathbb{R}^{3 \times 2}
$$

In PyTorch:

```python
A = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])  # Shape (2, 3)

A_T = A.T  # Shape (3, 2)
print(f"A shape: {A.shape}")
print(f"A.T shape: {A_T.shape}")
print(f"A.T:\n{A_T}")
```

**Research relevance:** Transposes appear in the
linear layer $ Y = XW^T + b$, in attention 
$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $, and in gradient computation 
$\frac{\partial L}{\partial W}$ = $X^T \frac{\partial L}{\partial Y}$.

### 2.5.5 Reshaping

Reshaping changes the shape of a tensor without 
changing its data. If $A \in \mathbb{R}^{d_1 \times \cdots \times d_n}$ and 
$B \in \mathbb{R}^{e_1 \times \cdots \times e_m}$, 
then reshaping is valid if:
$$\prod_{i=1}^{n} d_i = \prod_{j=1}^{m} e_j$$

For example, a vector of 12 elements can be reshaped to
$3 \times 4$, $2 \times 6 $, $2 \times 2 \times 3$, 
etc., because 
$12 = 3 \times 4 = 2 \times 6 = 2 \times 2 \times 3 $.

```python
A = torch.arange(12)  # Shape (12,)
print(f"A: {A}")

B = A.reshape(3, 4)  # Shape (3, 4)
print(f"B:\n{B}")

C = A.reshape(2, 2, 3)  # Shape (2, 2, 3)
print(f"C shape: {C.shape}")
```

The difference between `reshape` and `view`: `reshape` returns a tensor with the same data but a different shape, and may return a copy if the data is not contiguous. `view` returns a view of the original tensor (shares the same memory) and requires the data to be contiguous.

```python
A = torch.arange(12)
B = A.view(3, 4)  # View (shares memory)
C = A.reshape(3, 4)  # View or copy

# Modifying B modifies A
B[0, 0] = 100
print(f"A after modifying B: {A}")
```

### 2.5.6 Squeeze and Unsqueeze

**Squeeze** removes dimensions of size 1:
$$
\text{squeeze}: \mathbb{R}^{1 \times d \times 1} \to \mathbb{R}^{d}
$$

**Unsqueeze** adds a dimension of size 1 at a specified position:

$$
\text{unsqueeze}_0: \mathbb{R}^{d} \to \mathbb{R}^{1 \times d}
$$

```python
A = torch.randn(1, 3, 1, 4)
print(f"A shape: {A.shape}")

B = A.squeeze()
print(f"B shape: {B.shape}")

C = B.unsqueeze(0)
print(f"C shape: {C.shape}")

D = B.unsqueeze(1)
print(f"D shape: {D.shape}")
```

**Research relevance:** Squeeze and unsqueeze are 
used to match shapes for broadcasting. For example, 
to add a bias vector $\mathbf{b} \in \mathbb{R}^d$
to a batch $ X \in \mathbb{R}^{N \times d}$, 
you can use `X + b` (broadcasting) 
or `X + b.unsqueeze(0)`.

### 2.5.7 Concatenation and Stacking
**Concatenation** joins tensors along an 
existing dimension:

$$
\text{cat}: (\mathbb{R}^{m \times n_1}, \mathbb{R}^{m \times n_2}) \to \mathbb{R}^{m \times (n_1 + n_2)}
$$

**Stacking** joins tensors along a new dimension:

$$
\text{stack}: (\mathbb{R}^{m \times n}, \mathbb{R}^{m \times n}) \to \mathbb{R}^{2 \times m \times n}
$$

For example, with

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
$$
the concatenation along dim=0 is:

$$
\text{cat}(A, B, \text{dim}=0) = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \\ 7 & 8 \end{bmatrix} \in \mathbb{R}^{4 \times 2}
$$

and the stacking along dim=0 is:

$$
\text{stack}(A, B, \text{dim}=0) = \begin{bmatrix} \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \\ \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} \end{bmatrix} \in \mathbb{R}^{2 \times 2 \times 2}
$$

In PyTorch:

```python
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

C = torch.cat([A, B], dim=0)  # Shape (4, 2)
print(f"Concat dim=0:\n{C}")

D = torch.cat([A, B], dim=1)  # Shape (2, 4)
print(f"Concat dim=1:\n{D}")

E = torch.stack([A, B], dim=0)  # Shape (2, 2, 2)
print(f"Stack dim=0 shape: {E.shape}")
```

### 2.5.8 Summary: Tensor Operations

| Operation                   | Mathematical Notation | PyTorch | Shape Rule |
|-----------------------------|----------------------|---------|------------|
| Element-wise addition       | $ A + B$ | `a + b` | Same shape or broadcastable |
| Element-wise multiplication | $ A \odot B$ | `a * b` | Same shape or broadcastable |
| Matrix multiplication       | $AB$ | `a @ b` or `torch.matmul(a, b)` | $ (m, n) @ (n, p) \to (m, p) $ |
| Transpose                   |$ A^T $ | `a.T` | $ (m, n) \to (n, m)$ |
| Reshape                     | — | `a.reshape(...)` | Total elements preserved |
| Squeeze                     | — | `a.squeeze()` | Removes size-1 dimensions |
| Unsqueeze                   | — | `a.unsqueeze(dim)` | Adds size-1 dimension |
| Concatenation               | — | `torch.cat([a, b], dim)` | All dims except `dim` must match |
| Stacking                    | — | `torch.stack([a, b], dim)` | All shapes must match |

> **Exercise 2.4:** Implement the following operations both manually (using loops) and with PyTorch built-in functions. Compare the results and the execution time.
>
> 1. Element-wise sum of two vectors of size 1000
> 2. Matrix multiplication of two matrices of size \( 100 \times 100 \)
> 3. Transpose of a \( 500 \times 300 \) matrix
> 4. Reshape of a vector of size 12000 to \( 100 \times 120 \)
>
> Use `%%time` or `time.time()` to measure execution time. Which is faster? Why?

---

## 2.6 Aggregation Operations

### 2.6.1 Reduction Operations

A reduction operation maps a tensor to a lower-order tensor by aggregating values along one or more dimensions. The sum is:

$$
\text{sum}(A) = \sum_{i_1, \ldots, i_n} A_{i_1 \cdots i_n}
$$

The mean is:
$$
\text{mean}(A) = \frac{1}{|A|} \text{sum}(A)
$$

The max and min are:

$$
\text{max}(A) = \max_{i_1, \ldots, i_n} A_{i_1 \cdots i_n}, \quad \text{min}(A) = \min_{i_1, \ldots, i_n} A_{i_1 \cdots i_n}
$$

For example, with
$$
\mathbf{x} = \begin{bmatrix} 0 \\ 10 \\ 20 \\ 30 \\ 40 \\ 50 \\ 60 \\ 70 \\ 80 \\ 90 \end{bmatrix}
$$

the sum is \( 450 \), the mean is \( 45 \), the max is \( 90 \), and the min is \( 0 \).

In PyTorch:

```python
x = torch.arange(0, 100, 10, dtype=torch.float32)
print(f"x: {x}")
print(f"Sum: {x.sum()}")
print(f"Mean: {x.mean()}")
print(f"Max: {x.max()}")
print(f"Min: {x.min()}")
print(f"Argmax: {x.argmax()}")
print(f"Argmin: {x.argmin()}")
```

### 2.6.2 Reduction Along a Dimension

You can reduce along a specific dimension by specifying `dim`:
$$
\text{sum}(A, \text{dim}=0)_{j} = \sum_{i} A_{ij}
$$

For example, with
$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}
$$

the sum over dim=0 is:
$$
\text{sum}(A, \text{dim}=0) = \begin{bmatrix} 1+4 & 2+5 & 3+6 \end{bmatrix} = \begin{bmatrix} 5 & 7 & 9 \end{bmatrix}
$$

and the sum over dim=1 is:
$$
\text{sum}(A, \text{dim}=1) = \begin{bmatrix} 1+2+3 \\ 4+5+6 \end{bmatrix} = \begin{bmatrix} 6 \\ 15 \end{bmatrix}
$$
In PyTorch:

```python
A = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

print(f"A:\n{A}")
print(f"Sum over dim=0: {A.sum(dim=0)}")  # Shape (3,)
print(f"Sum over dim=1: {A.sum(dim=1)}")  # Shape (2,)
print(f"Mean over dim=0: {A.mean(dim=0)}")
print(f"Max over dim=1: {A.max(dim=1)}")
```

**Research relevance:** Reduction operations are used in computing the loss (mean over batch), computing accuracy (mean of correct predictions), normalizing activations (mean and std over batch), and attention (sum over keys).

### 2.6.3 The `keepdim` Parameter

By default, reduction operations remove the reduced dimension. The `keepdim=True` parameter preserves it as size 1. For example, with \( A \in \mathbb{R}^{3 \times 4} \):
$$
\text{mean}(A, \text{dim}=0) \in \mathbb{R}^{4}, \quad \text{mean}(A, \text{dim}=0, \text{keepdim=True}) \in \mathbb{R}^{1 \times 4}
$$
In PyTorch:

```python
A = torch.randn(3, 4)
print(f"A shape: {A.shape}")

mean_dim0 = A.mean(dim=0)
print(f"Mean dim=0 shape: {mean_dim0.shape}")

mean_dim0_keep = A.mean(dim=0, keepdim=True)
print(f"Mean dim=0 keepdim shape: {mean_dim0_keep.shape}")
```

**Research relevance:** `keepdim=True` is essential for broadcasting.

To normalize a batch $X \in \mathbb{R}^{N \times d}$ by its mean $\mu \in \mathbb{R}^{d}$:

$$X_{\text{norm}} = X - \mu$$

If $\mu$ has shape $(d,)$, broadcasting works because NumPy/PyTorch aligns
trailing dimensions: $(N, d)$ vs $(d,)$ → $(N, d) - (d,)$ broadcasts correctly.

But if you compute the mean along a **different 
dimension**, for example,
reducing over the batch dimension $N$, you get a result of shape $(d,)$ only
if you use `keepdim=True`. Without `keepdim=True`, reducing over `dim=0` gives
shape $(d,)$, which still broadcasts against $(N, d)$ fine in this case.
However, if you reduce over `dim=1` (the feature dimension), you get shape
$(N,)$ **without** `keepdim=True`, which does 
**not** broadcast against
$(N, d)$ as intended, you wouldd need shape $(N, 1)$.

So the key point:

- **`keepdim=True`** preserves the reduced dimension as size 1, e.g. reducing
  $(N, d)$ over `dim=1` gives $(N, 1)$ instead of $(N,)$.
- This ensures the reduced tensor broadcasts correctly against the original
  along the dimension you want to normalize.

Example:

```python
# Normalize along features (per-sample), reduce over dim=1
mu = X.mean(dim=1, keepdim=True)   # shape (N, 1)
X_norm = X - mu                     # broadcasts to (N, d) ✓

# Without keepdim:
mu = X.mean(dim=1)                  # shape (N,)
X_norm = X - mu                     # tries (N, d) - (N,) → aligns to (d,) vs (N,) → error or wrong result
```
> **Exercise 2.5:** 
> Given a batch of images $X \in \mathbb{R}^{32 \times 3 \times 224 \times 224}$, compute:
> 1. The mean and standard deviation of each channel across the batch, height, and width. The result should have shape \( (3,) \).
> 2. Normalize the batch using these statistics: 
  $X_{\text{norm}} = \frac{(X - \mu)}{\sigma}$.
> 3. Verify that the normalized batch has mean ≈ 0 and std ≈ 1 for each channel.


## 2.7 Indexing and Slicing

### 2.7.1 Basic Indexing

Indexing extracts a sub-tensor. For a tensor $A \in \mathbb{R}^{d_1 \times \cdots \times d_n}$, 
indexing with integers $i_1, \ldots, i_n $ returns a scalar:
$$A_{i_1 \cdots i_n} \in \mathbb{R}$$
Indexing with a colon `:` selects all values along that dimension. For example, with

$$
x = \begin{bmatrix} \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} \end{bmatrix} \in \mathbb{R}^{1 \times 3 \times 3}
$$
the indexing operations are:

$$
x[0] = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}, \quad x[0][0] = \begin{bmatrix} 1 & 2 & 3 \end{bmatrix}, \quad x[0][0][0] = 1
$$
In PyTorch:

```python
x = torch.arange(1, 10).reshape(1, 3, 3)
print(f"x:\n{x}")
print(f"x[0]:\n{x[0]}")
print(f"x[0][0]: {x[0][0]}")
print(f"x[0][0][0]: {x[0][0][0]}")

# Using colons
print(f"x[:, 0]:\n{x[:, 0]}")
print(f"x[:, :, 1]:\n{x[:, :, 1]}")
print(f"x[:, 1, 1]: {x[:, 1, 1]}")
print(f"x[0, 0, :]: {x[0, 0, :]}")
```

### 2.7.2 Advanced Indexing

Boolean indexing selects elements based on a condition:
$$
A[A > 0] = \{A_{i_1 \cdots i_n} : A_{i_1 \cdots i_n} > 0\}
$$

For example, with
$$
A = \begin{bmatrix} -1 & 2 \\ -3 & 4 \end{bmatrix}
$$
the boolean mask is:
$$
A > 0 = \begin{bmatrix} \text{False} & \text{True} \\ \text{False} & \text{True} \end{bmatrix}
$$

and the selected elements are \( \{2, 4\} \).

In PyTorch:

```python
x = torch.randn(3, 4)
print(f"x:\n{x}")
print(f"x > 0:\n{x > 0}")
print(f"x[x > 0]: {x[x > 0]}")
```

**Research relevance:** Boolean indexing is used in masking padding tokens in NLP, selecting valid pixels in segmentation, and filtering outliers in data preprocessing.

> **Exercise 2.6:**
> Given a batch of images $X \in \mathbb{R}^{8 \times 3 \times 32 \times 32}$:
>
> 1. Extract the first image: $X[0]$
> 2. Extract the red channel of all images: $ X[:, 0, :, :]$
> 3. Extract the top-left $16 \times 16$ patch of each image: $ X[:, :, :16, :16]$
> 4. Create a mask for pixels with value > 0.5 and count how many such pixels exist.
> 5. Set all pixels with value < 0 to 0 (ReLU operation).


## 2.8 NumPy Interoperability

### 2.8.1 Converting Between NumPy and PyTorch
NumPy to PyTorch uses `torch.from_numpy(ndarray)`. PyTorch to NumPy uses `tensor.numpy()`.

```python
import numpy as np
import torch

# NumPy to PyTorch
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(f"Array: {array}")
print(f"Tensor: {tensor}")
print(f"Array dtype: {array.dtype}")
print(f"Tensor dtype: {tensor.dtype}")

# PyTorch to NumPy
tensor = torch.ones(7)
array = tensor.numpy()
print(f"Tensor: {tensor}")
print(f"Array: {array}")
```

By default, NumPy arrays are created with the datatype `float64` and if you convert it to a PyTorch tensor, it will keep the same datatype. However, many PyTorch calculations default to using `float32`. So if you want to convert your NumPy array (float64) to PyTorch tensor (float64) to PyTorch tensor (float32), you can use `tensor = torch.from_numpy(array).type(torch.float32)`.

### 2.8.2 Shared Memory

When you convert a NumPy array to a PyTorch tensor (or vice versa), the two share memory by default. Modifying one modifies the other.

```python
array = np.array([1.0, 2.0, 3.0])
tensor = torch.from_numpy(array)

# Modify the tensor
tensor[0] = 100.0
print(f"Tensor: {tensor}")
print(f"Array: {array}")  # Also modified!
```

To avoid this, use `.clone()` or `.copy()`:

```python
array = np.array([1.0, 2.0, 3.0])
tensor = torch.from_numpy(array).clone()

tensor[0] = 100.0
print(f"Tensor: {tensor}")
print(f"Array: {array}")  # Not modified
```

### 2.8.3 GPU Tensors and NumPy

NumPy cannot operate on GPU tensors. To convert a GPU tensor to NumPy, first move it to CPU:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
tensor_gpu = torch.ones(7, device=device)

# This will error:
# array = tensor_gpu.numpy()

# Correct:
array = tensor_gpu.cpu().numpy()
print(f"Array: {array}")
```

> **Exercise 2.7:** Write a function that takes a NumPy array, converts it to a PyTorch tensor, moves it to the GPU (if available), performs a matrix multiplication with its transpose, moves the result back to CPU, and returns it as a NumPy array. Test it with a random \( 100 \times 50 \) array.



## 2.9 Worked Example: Linear Regression Data Pipeline

Let us apply everything we have learned to a complete, reproducible data pipeline for a linear regression problem.

### 2.9.1 The Mathematical Model

We consider the linear model:
$$
y = a + bx + \epsilon
$$

where $a$ is the intercept, $b$ is the slope, and
$\epsilon \sim \mathcal{N}(0, \sigma^2)$ is Gaussian noise. This is the same problem studied in the original notebook. We know that \( a = 1 \) and \( b = 2 \), and we want to recover these values from noisy data using gradient descent.

### 2.9.2 Data Generation

Following the original notebook, we generate 100 data points:
$$
x_i \sim \mathcal{U}(0, 1), \quad y_i = 1 + 2x_i + 0.1 \cdot \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, 1)
$$

In PyTorch:

```python
import torch
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

print(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")
print(f"x mean: {x.mean():.4f}, x std: {x.std():.4f}")
print(f"y mean: {y.mean():.4f}, y std: {y.std():.4f}")
```

### 2.9.3 Train/Validation Split

Following the original notebook, we shuffle the indices and use the first 80 for training and the remaining 20 for validation:

```python
# Shuffle indices
idx = torch.randperm(N)
train_idx = idx[:80]
val_idx = idx[80:]

x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

print(f"Train size: {x_train.shape[0]}")
print(f"Validation size: {x_val.shape[0]}")
```

### 2.9.4 Device Management

Following the original notebook, we move all tensors to the chosen device:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

x_train = x_train.to(device)
y_train = y_train.to(device)
x_val = x_val.to(device)
y_val = y_val.to(device)

print(f"x_train device: {x_train.device}")
```

### 2.9.5 Summary Statistics

```python
print(f"Train x mean: {x_train.mean().item():.4f}")
print(f"Train y mean: {y_train.mean().item():.4f}")
print(f"Train x std: {x_train.std().item():.4f}")
print(f"Train y std: {y_train.std().item():.4f}")
```

### 2.9.6 Reflection

This pipeline illustrates several important principles. Reproducibility: all seeds are set before any random operation. Shape awareness: we print shapes at each step to catch errors early. Device management: we move all tensors to the same device. Train/validation split: we hold out data for evaluation. These principles will carry through every experiment you run in this course.

> **Exercise 2.8:** Extend the pipeline above to include the following:
>
> 1. Standardize the features: $x_{\text{norm}} =\frac{ (x - \mu_x)}{\sigma_x}$.
> 2. Standardize the targets: $ y_{\text{norm}} = \frac{(y - \mu_y)}{\sigma_y}$.
> 3. Verify that the normalized features and targets have mean ≈ 0 and std ≈ 1.
> 4. Explain why standardization might be important for gradient descent.


## 2.10 Common Errors and Debugging

### 2.10.1 Shape Mismatch

The error `RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x2 and 3x2)` occurs when matrix multiplication requires inner dimensions to match. The fix is to transpose one of the tensors or reshape.

```python
A = torch.randn(3, 2)
B = torch.randn(3, 2)
# C = A @ B  # Error
C = A @ B.T  # Correct
print(f"C shape: {C.shape}")
```

### 2.10.2 Dtype Mismatch

The error `RuntimeError: expected scalar type Float but found Double` occurs when operations between tensors of different dtypes. The fix is to cast to the same dtype.

```python
a = torch.tensor([1.0, 2.0, 3.0])  # Float32
b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)  # Float64
# c = a + b  # Error
c = a + b.float()  # Correct
print(f"c dtype: {c.dtype}")
```

### 2.10.3 Device Mismatch

The error `RuntimeError: Expected all tensors to be on the same device` occurs when operations between CPU and GPU tensors. The fix is to move all tensors to the same device.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
a = torch.randn(3, 4).to(device)
b = torch.randn(3, 4).to(device)
c = a + b
print(f"c device: {c.device}")
```

### 2.10.4 In-place Operation Error

The error `RuntimeError: a leaf Variable that requires grad has been used in an in-place operation` occurs when modifying a tensor that requires gradients in-place. The fix is to use `torch.no_grad()` or avoid in-place operations.

```python
a = torch.randn(1, requires_grad=True)
# a -= 1  # Error
with torch.no_grad():
    a -= 1  # Correct
print(f"a: {a}")
```

### 2.10.5 Common Errors

| Error               | Cause                                        | Fix                                |
|---------------------|----------------------------------------------|------------------------------------|
| Shape mismatch      | Matrix multiplication inner dims don't match | Transpose or reshape               |
| Dtype mismatch      | Different dtypes in operation                | Cast to same dtype                 |
| Device mismatch     | CPU and GPU tensors mixed                    | Move to same device                |
| In-place operation  | Modifying leaf tensor with `requires_grad`   | Use `torch.no_grad()`              |
| Non-contiguous view | `view()` on non-contiguous tensor            | Use `reshape()` or `.contiguous()` |

> **Exercise 2.9:** For each of the following errors, identify the cause and fix it:
>
> ```python
> # Error 1
> a = torch.randn(3, 4)
> b = torch.randn(5, 6)
> c = a @ b
> ```
>
> ```python
> # Error 2
> a = torch.tensor([1, 2, 3])
> b = torch.tensor([1.0, 2.0, 3.0])
> c = a + b
> ```
>
> ```python
> # Error 3
> a = torch.randn(3, 4).to("cuda")
> b = torch.randn(3, 4)
> c = a + b
> ```
>
> ```python
> # Error 4
> a = torch.randn(3, 4)
> b = a.T
> c = b.view(12)
> ```
>
> ```python
> # Error 5
> a = torch.randn(1, requires_grad=True)
> a = a - 1
> b = a * 2
> b.backward()
> ```

## 2.11 Research Application: Tensor Operations in a Neural Network

Let us see how the tensor operations we have learned combine to form a neural network layer.

### 2.11.1 The Linear Layer

A linear layer computes:
$$Y = XW^T + b$$

where  $ X \in \mathbb{R}^{N \times d_{\text{in}}} $ is the input batch, 
$W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is the weight matrix, 
$b \in \mathbb{R}^{d_{\text{out}}}$ is the bias vector,
and $Y \in \mathbb{R}^{N \times d_{\text{out}}}$ is the output.

For example, with  $ N = 8  $,  $d_{\text{in}} = 4  $,
and  $d_{\text{out}} = 3 $, the shapes are:
$$
X \in \mathbb{R}^{8 \times 4}, \quad W \in \mathbb{R}^{3 \times 4}, \quad b \in \mathbb{R}^{3}, \quad Y \in \mathbb{R}^{8 \times 3}
$$

In PyTorch:

```python
import torch
import torch.nn as nn

# Create a linear layer
linear = nn.Linear(in_features=4, out_features=3)

# Input batch
X = torch.randn(8, 4)  # Batch of 8, 4 features each

# Forward pass
Y = linear(X)

print(f"X shape: {X.shape}")
print(f"Weight shape: {linear.weight.shape}")
print(f"Bias shape: {linear.bias.shape}")
print(f"Y shape: {Y.shape}")
```

### 2.11.2 Manual Computation

We can reproduce the linear layer manually:
$$Y_{\text{manual}} = XW^T + b$$

In PyTorch:

```python
W = linear.weight  # Shape (3, 4)
b = linear.bias    # Shape (3,)

Y_manual = X @ W.T + b

print(f"Manual Y shape: {Y_manual.shape}")
print(f"Matches: {torch.allclose(Y, Y_manual)}")
```

### 2.11.3 Shape Analysis

The shapes in the linear layer follow the rules of matrix multiplication:
$$
\underbrace{X}_{(N, d_{\text{in}})} \cdot \underbrace{W^T}_{(d_{\text{in}}, d_{\text{out}})} + \underbrace{b}_{(d_{\text{out}},)} = \underbrace{Y}_{(N, d_{\text{out}})}
$$

The bias  $ b $ is broadcast across the batch dimension.

### 2.11.4 Research Relevance

Understanding the shape rules of the linear layer is
essential for debugging (most shape errors occur in 
linear layers), architecture design (the choice of 
$d_{\text{in}}$ and $ d_{\text{out}}$ determines the model's capacity), and transfer learning (when fine-tuning, you replace the final linear layer to match the number of classes).

> **Exercise 2.10:** Implement a two-layer MLP manually using tensor operations. The architecture is:
>
> 
> $$H = \text{ReLU}(XW_1^T + b_1)$$
> 
> $$Y = HW_2^T + b_2$$
> where $ X \in \mathbb{R}^{16 \times 10} $, $ W_1 \in \mathbb{R}^{20 \times 10}$, $ b_1 \in \mathbb{R}^{20} $, 
> $W_2 \in \mathbb{R}^{5 \times 20}$, 
> $b_2 \in \mathbb{R}^{5}$.
>
> 1. Initialize the weights and biases randomly.
> 2. Compute the forward pass.
> 3. Verify that the output has shape $(16, 5)$.
> 4. Compare with `nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))`.


## 2.12 Module Summary

| Concept      | Mathematical Form                                    | PyTorch                          | Research Relevance             |
|--------------|------------------------------------------------------|----------------------------------|--------------------------------|
| Scalar       | $ a \in \mathbb{R}  $                                | `torch.tensor(7)`                | Loss values, metrics           |
| Vector       | $ \mathbf{x} \in \mathbb{R}^d  $                     | `torch.randn(d)`                 | Features, embeddings           |
| Matrix       | $ A \in \mathbb{R}^{m \times n}  $                   | `torch.randn(m, n)`              | Weight matrices                |
| 4-tensor     | $ X \in \mathbb{R}^{N \times C \times H \times W}  $ | `torch.randn(N, C, H, W)`        | Image batches                  |
| Shape        | $ (d_1, \ldots, d_n)  $                              | `.shape`                         | Debugging, architecture design |
| Dtype        | $\mathbb{R}, \mathbb{Z}, \mathbb{B}  $               | `.dtype`                         | Memory, precision              |
| Device       | CPU/GPU                                              | `.device`                        | Performance                    |
| Broadcasting | $ (m, 1) + (1, n) \to (m, n)  $                      | `a + b`                          | Bias addition, normalization   |
| Matmul       | $ AB  $                                              | `a @ b`                          | Linear layers, attention       |
| Transpose    | $A^T  $                                              | `a.T`                            | Gradient computation           |
| Reshape      | —                                                    | `a.reshape(...)`                 | Shape compatibility            |
| Reduction    | $\sum, \frac{1}{n}\sum, \max, \min $                 | `.sum()`, `.mean()`, `.max()`    | Loss, accuracy                 |
| Indexing     | $ A_{i_1 \cdots i_n}  $                              | `a[i, j, k]`                     | Data selection                 |
| NumPy bridge | —                                                    | `torch.from_numpy()`, `.numpy()` | Data loading                   |


## 2.13 Capstone Thread: Tensor Skills for Your Project

By the end of this module, you should be able to create tensors of any shape, dtype, and device; diagnose and fix shape, dtype, and device errors; write reproducible tensor code with controlled randomness; perform matrix multiplication, broadcasting, and reduction operations; convert between NumPy and PyTorch; and implement a linear layer from scratch.

For your capstone project, you will need these skills to load and preprocess your data, implement your model's forward pass, debug shape errors, and ensure reproducibility.

In Module 3, we will build on this foundation to study **autograd** and **computation graphs** — the machinery that makes PyTorch a deep learning framework rather than just a tensor library.

## 2.14 Additional Resources

### On Tensors and Linear Algebra

- Strang, G. (2016). *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 2: Linear Algebra.
- PyTorch documentation: [torch.Tensor](https://pytorch.org/docs/stable/tensors.html)

### On Reproducibility

- PyTorch reproducibility guide: [Link](https://pytorch.org/docs/stable/notes/randomness.html)
- Pineau, J., et al. (2021). "Improving Reproducibility in Machine Learning Research." *JMLR*.

### On Debugging

- PyTorch debugging guide: [Link](https://pytorch.org/docs/stable/notes/autograd.html)
- "What, What, Where" framework from the original notebook.

### On Numerical Computing

- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.
- Goldberg, D. (1991). "What Every Computer Scientist Should Know About Floating-Point Arithmetic." *ACM Computing Surveys*, 23(1), 5–48.

