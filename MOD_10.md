# Module 10: Tabular Data and Regression Research

## TORA CS 336: PyTorch for Research

**Institution:** The Open Research Academy (TORA)

**Instructor:** Alpha Alimamy Kamara

**Module Length:** 2 weeks (4 lectures + 2 lab sessions)

**Prerequisites:** Modules 1–9 completed; familiarity with `nn.Module`, training loops, data pipelines, and basic statistics; access to a tabular dataset (Excel, CSV, or public repository).

---

## 10.0 Module Overview

In Modules 1–9, we built the foundations of research practice: the research mindset, tensor computing, autograd, model building, training loops, data pipelines, literature management, critical reading, and reproduction. Now we turn to **applied research** — the process of conducting original research on a specific data modality and task.

This module is about **tabular data and regression research**. Tabular data is the most common data format in industry, medicine, finance, and the physical sciences. It consists of structured rows and columns: patient records, financial transactions, sensor readings, experimental measurements. Regression is the task of predicting a continuous target from these features: predicting house prices, patient outcomes, material properties, or process parameters.

The choice of tabular data as the first applied module is deliberate. Unlike images or text, tabular data does not have a canonical architecture. Neural networks often underperform gradient boosting on tabular data, and understanding *when* and *why* is an open research question. This makes tabular data an excellent domain for original research: the baselines are strong, the datasets are accessible, and the questions are unresolved.

The pedagogical approach remains **mathematics first, code second**. For every method, we will write the mathematical object explicitly — the regression model, the loss function, the evaluation metric — and only then translate it into PyTorch. This mirrors how research is conducted: you have a mathematical model in mind, and PyTorch is the instrument that realizes it.

By the end of this module, you will be able to load and preprocess tabular data from Excel/CSV; implement regression models in PyTorch; evaluate them with appropriate metrics; compare against gradient boosting baselines; perform statistical significance testing; and articulate when neural networks are preferable to tree-based methods.

The module is self-paced. Work through the mathematics carefully before running the code. The code will make much more sense if you understand what it is computing.

---

## 10.1 The Tabular Data Problem

### 10.1.1 The Mathematical Formulation

A tabular dataset is a matrix $ X \in \mathbb{R}^{N \times d} $, where $ N $ is the number of samples and $ d $ is the number of features. Each row $ \mathbf{x}_i \in \mathbb{R}^d $ is a sample, and each column $ \mathbf{x}_{:,j} \in \mathbb{R}^N $ is a feature.

For regression, the target is a vector $ \mathbf{y} \in \mathbb{R}^N $. The goal is to learn a function $ f_\theta: \mathbb{R}^d \to \mathbb{R} $ that maps features to targets:

$$
\hat{y}_i = f_\theta(\mathbf{x}_i)
$$

The learning objective is to minimize the empirical risk:

$$
\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(y_i, \hat{y}_i)
$$

where $ \ell $ is a loss function, typically the mean squared error (MSE):

$$
\ell_{\text{MSE}}(y, \hat{y}) = (y - \hat{y})^2
$$

### 10.1.2 The Tabular Data Landscape

Tabular data differs from images and text in several important ways:

**Heterogeneous features.** Tabular features can be numerical (continuous or discrete), categorical (nominal or ordinal), or temporal. Images have homogeneous pixel values; text has homogeneous token indices.

**No spatial or sequential structure.** The order of columns in a table is arbitrary. In contrast, the order of pixels in an image and tokens in a sentence is meaningful.

**Mixed scales.** Features can have vastly different scales (e.g., age in years vs. income in dollars). This requires normalization.

**Missing values.** Tabular data often has missing values, which must be imputed.

**Small sample sizes.** Many tabular datasets have hundreds or thousands of samples, not millions. This limits the capacity of neural networks.

**Strong baselines.** Gradient boosting methods (XGBoost, LightGBM, CatBoost) are strong baselines on tabular data. Neural networks do not always outperform them.

### 10.1.3 Why Neural Networks Struggle on Tabular Data

The 2022 paper "Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data?" by Grinsztajn et al. identified several reasons:

**Rotation invariance.** Neural networks are rotation-invariant: they treat all directions in feature space equally. But tabular data often has axis-aligned structure: the target depends on individual features, not on linear combinations. Tree-based models exploit this structure.

**Smoothness.** Neural networks assume smoothness: nearby inputs produce nearby outputs. But tabular data often has sharp, discontinuous decision boundaries. Tree-based models can represent these boundaries naturally.

**Feature selection.** Neural networks use all features in every layer. Tree-based models select the most informative features at each split. On datasets with many irrelevant features, tree-based models are more efficient.

**Hyperparameter sensitivity.** Neural networks are sensitive to architecture, learning rate, and initialization. Tree-based models are more robust.

This does not mean neural networks are useless on tabular data. On large datasets (millions of samples), neural networks can outperform tree-based models. On datasets with complex interactions, neural networks can learn representations that tree-based models cannot. The question of *when* is an open research problem.

### 10.1.4 Summary: The Tabular Data Problem

| Aspect | Tabular Data | Images | Text |
|--------|--------------|--------|------|
| Structure | Heterogeneous features | Homogeneous pixels | Homogeneous tokens |
| Order | Arbitrary | Spatial | Sequential |
| Scale | Mixed | Fixed | Fixed |
| Missing values | Common | Rare | Rare |
| Sample size | Small–medium | Large | Large |
| Baseline | Gradient boosting | CNN | Transformer |

> **Exercise 10.1:** For each of the following datasets, identify the features (numerical, categorical, temporal), the target, and the likely challenges.
>
> 1. A medical dataset with patient age, sex, blood pressure, cholesterol, and heart disease risk.
> 2. A financial dataset with transaction amount, merchant category, time of day, and fraud label.
> 3. An industrial dataset with temperature, pressure, flow rate, and product quality.

---

## 10.2 Loading Excel/CSV Data into PyTorch

### 10.2.1 The Data Pipeline

Loading tabular data into PyTorch involves several steps:

1. **Load** the data from Excel or CSV into a pandas DataFrame.
2. **Explore** the data: shape, dtypes, missing values, summary statistics.
3. **Preprocess** the data: handle missing values, encode categorical features, normalize numerical features.
4. **Split** the data into train/validation/test sets.
5. **Convert** to PyTorch tensors.
6. **Wrap** in a `Dataset` and `DataLoader`.

### 10.2.2 Loading Excel Data

Excel files are loaded with `pandas.read_excel`:

```python
import pandas as pd

# Load Excel file
df = pd.read_excel('data.xlsx')
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.describe())
```

**Worked example 10.1:** Consider a welding process dataset with columns `Current`, `Angle`, `Speed`, `Time`, and `Height` (target). The goal is to predict `Height` from the other features.

```python
# Load data
df = pd.read_excel('welding_data.xlsx')

# Explore
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Dtypes:\n{df.dtypes}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Summary statistics:\n{df.describe()}")
```

The output reveals the structure of the data: how many samples, what types of features, whether there are missing values, and the range of each feature.

### 10.2.3 Loading CSV Data

CSV files are loaded with `pandas.read_csv`:

```python
df = pd.read_csv('data.csv')
```

The same exploration steps apply.

### 10.2.4 Exploring the Data

Before preprocessing, explore the data:

**Shape:** How many samples and features?

**Dtypes:** Which features are numerical? Which are categorical?

**Missing values:** Which features have missing values? How many?

**Summary statistics:** What are the mean, std, min, max of each numerical feature?

**Distributions:** Are the features normally distributed? Skewed? Bimodal?

**Correlations:** Which features are correlated with the target? With each other?

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of target
sns.histplot(df['Height'], kde=True)
plt.show()

# Correlation matrix
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
```

**Research relevance:** Exploratory data analysis (EDA) is a critical step in research. It reveals data quality issues, outliers, and relationships that inform preprocessing and modeling.

### 10.2.5 Handling Missing Values

Missing values are common in tabular data. Strategies:

**Drop rows:** If missing values are rare, drop the affected rows.

**Drop columns:** If a column has many missing values, drop it.

**Impute with mean/median:** For numerical features, impute with the mean or median.

**Impute with mode:** For categorical features, impute with the mode.

**Model-based imputation:** Use a model (e.g., k-NN, iterative imputer) to predict missing values.

```python
from sklearn.impute import SimpleImputer

# Impute numerical features with mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
```

**Research relevance:** Imputation must be fit on the training set only, then applied to the validation and test sets. Fitting on the full dataset leaks information.

### 10.2.6 Encoding Categorical Features

Categorical features must be encoded as numbers. Strategies:

**One-hot encoding:** Create a binary column for each category.

**Ordinal encoding:** Assign an integer to each category (for ordinal features).

**Target encoding:** Replace each category with the mean target value.

**Embedding:** Learn a dense vector for each category (for neural networks).

```python
from sklearn.preprocessing import OneHotEncoder

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X_categorical)
```

### 10.2.7 Normalizing Numerical Features

Numerical features should be normalized to have mean 0 and std 1:

$$
x_{\text{norm}} = \frac{x - \mu}{\sigma}
$$

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numerical)
```

**Research relevance:** Normalization is essential for gradient descent. Without it, features with large scales dominate the gradient, leading to slow convergence.

### 10.2.8 Summary: Loading Tabular Data

| Step | Tool | Description |
|------|------|-------------|
| Load | `pd.read_excel`, `pd.read_csv` | Read data |
| Explore | `df.head`, `df.describe`, `df.isnull` | Understand data |
| Impute | `SimpleImputer` | Handle missing values |
| Encode | `OneHotEncoder`, `OrdinalEncoder` | Encode categorical |
| Normalize | `StandardScaler`, `MinMaxScaler` | Scale numerical |
| Split | `train_test_split` | Train/val/test |
| Convert | `torch.tensor` | PyTorch tensors |

> **Exercise 10.2:** Load a tabular dataset from Excel or CSV. Explore it: shape, dtypes, missing values, summary statistics. Identify which features are numerical and which are categorical. Propose a preprocessing strategy.

---

## 10.3 Preprocessing Tabular Data

### 10.3.1 The Preprocessing Pipeline

A complete preprocessing pipeline for tabular data:

1. **Split** the data into train/validation/test sets.
2. **Impute** missing values (fit on train, apply to val/test).
3. **Encode** categorical features (fit on train, apply to val/test).
4. **Normalize** numerical features (fit on train, apply to val/test).
5. **Convert** to PyTorch tensors.

The key principle: **fit on train, apply to val/test**. This prevents data leakage.

### 10.3.2 Worked Example 10.2: Preprocessing the Welding Dataset

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import torch

# Load data
df = pd.read_excel('welding_data.xlsx')

# Separate features and target
X = df.drop('Height', axis=1).values
y = df['Height'].values

# Split first
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Impute (fit on train)
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

# Normalize (fit on train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
```

### 10.3.3 Common Pitfalls

**Fitting preprocessing on the full dataset.** This leaks information from val/test into train. Always split first.

**Forgetting to apply preprocessing to val/test.** The model expects inputs with the same distribution as training. If you normalize train but not val/test, performance will degrade.

**Using different preprocessing for train and val/test.** The scaler and encoder must be the same for all sets.

**Ignoring missing values.** Many models cannot handle NaN. Impute before training.

### 10.3.4 Summary: Preprocessing

| Step | Fit On | Apply To |
|------|--------|----------|
| Imputation | Train | Train, val, test |
| Encoding | Train | Train, val, test |
| Normalization | Train | Train, val, test |

> **Exercise 10.3:** Implement a complete preprocessing pipeline for a tabular dataset of your choice. Ensure that all preprocessing steps are fit on the training set only. Verify that the training, validation, and test sets have consistent shapes and distributions.

---

## 10.4 A Custom Dataset for Tabular Data

### 10.4.1 The `TabularDataset` Class

A custom `Dataset` for tabular data:

```python
from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

# Create datasets
train_dataset = TabularDataset(X_train, y_train)
val_dataset = TabularDataset(X_val, y_val)
test_dataset = TabularDataset(X_test, y_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

### 10.4.2 Why a Custom Dataset?

For simple tabular data, `TensorDataset` is sufficient:

```python
from torch.utils.data import TensorDataset
train_dataset = TensorDataset(X_train, y_train)
```

A custom `Dataset` is useful when:
- You need to apply transformations on the fly.
- You have multiple inputs or outputs.
- You need to handle variable-length sequences.
- You want to implement custom sampling.

### 10.4.3 Summary: Tabular Dataset

| Aspect | `TensorDataset` | Custom `Dataset` |
|--------|-----------------|------------------|
| Simplicity | High | Medium |
| Flexibility | Low | High |
| On-the-fly transforms | No | Yes |
| Multiple inputs | No | Yes |
| Use case | Simple tabular | Complex tabular |

> **Exercise 10.4:** Implement a `TabularDataset` that applies on-the-fly normalization using statistics computed from the training set. Verify that the normalized samples have mean ≈ 0 and std ≈ 1.

---

## 10.5 Case Study: Welding Process Parameter Prediction

### 10.5.1 The Problem

Welding is a manufacturing process that joins metals by applying heat and pressure. The quality of the weld depends on several process parameters: current, angle, speed, and time. The goal is to predict the weld height from these parameters.

This is a regression problem: given a set of process parameters, predict the weld height. Accurate predictions allow manufacturers to optimize the process and reduce defects.

### 10.5.2 The Data

The dataset has $ N = 480 $ samples with $ d = 4 $ features:

| Feature | Description | Range |
|---------|-------------|-------|
| Current | Welding current (A) | 50–200 |
| Angle | Torch angle (degrees) | 0–45 |
| Speed | Welding speed (mm/s) | 1–10 |
| Time | Welding time (s) | 1–20 |

The target is `Height` (mm), ranging from 0.5 to 5.

### 10.5.3 The Mathematical Model

We model the relationship between features and target as a neural network:

$$
\hat{y} = f_\theta(\mathbf{x}) = W_3 \cdot \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2) + \mathbf{b}_3
$$

where $ \mathbf{x} \in \mathbb{R}^4 $, $ W_1 \in \mathbb{R}^{64 \times 4} $, $ W_2 \in \mathbb{R}^{64 \times 64} $, $ W_3 \in \mathbb{R}^{1 \times 64} $.

### 10.5.4 The Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)

# Load data
df = pd.read_excel('welding_data.xlsx')
X = df[['Current', 'Angle', 'Speed', 'Time']].values
y = df['Height'].values

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP(input_dim=4).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training
n_epochs = 200
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    model.train()
    epoch_train_losses = []
    for i in range(0, len(X_train), 32):
        x_batch = X_train[i:i+32].to(device)
        y_batch = y_train[i:i+32].to(device)
        yhat = model(x_batch)
        loss = loss_fn(y_batch, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_train_losses.append(loss.item())
    train_losses.append(np.mean(epoch_train_losses))

    model.eval()
    with torch.no_grad():
        yhat_val = model(X_val.to(device))
        val_loss = loss_fn(y_val.to(device), yhat_val)
        val_losses.append(val_loss.item())

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: train_loss={train_losses[-1]:.4f}, val_loss={val_losses[-1]:.4f}")

# Test
model.eval()
with torch.no_grad():
    yhat_test = model(X_test.to(device))
    test_loss = loss_fn(y_test.to(device), yhat_test)
    print(f"Test MSE: {test_loss.item():.4f}")
```

### 10.5.5 Evaluation

Beyond MSE, we compute:

**RMSE:** $ \sqrt{\text{MSE}} $ — interpretable in the units of the target.

**MAE:** $ \frac{1}{N} \sum |y - \hat{y}| $ — robust to outliers.

**R²:** $ 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2} $ — proportion of variance explained.

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = yhat_test.cpu().numpy()
y_true = y_test.cpu().numpy()

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
```

### 10.5.6 Reflection

The welding dataset is a classic tabular regression problem. The MLP achieves reasonable performance, but a gradient boosting model might do better. The choice between them depends on the dataset size, the complexity of the relationships, and the availability of compute.

**Research relevance:** This case study illustrates the full tabular regression pipeline. For your capstone, you will apply the same pipeline to your own dataset, with modifications for the specific features and target.

> **Exercise 10.5:** Apply the pipeline above to a tabular dataset of your choice. Report RMSE, MAE, and R² on the test set. Compare with a linear regression baseline.

---

## 10.6 Baselines: Gradient Boosting vs. Neural Networks

### 10.6.1 Why Baselines Matter

In research, a model is only as good as its baseline. If you propose a neural network and it performs worse than a simple linear regression, your contribution is not meaningful. Baselines provide context for your results and ensure that your method is actually an improvement.

For tabular regression, the standard baselines are:

**Linear regression:** $ \hat{y} = \mathbf{w}^T \mathbf{x} + b $. Simple, interpretable, fast.

**Ridge regression:** Linear regression with L2 regularization. Handles multicollinearity.

**Lasso regression:** Linear regression with L1 regularization. Performs feature selection.

**Random forest:** Ensemble of decision trees. Handles nonlinearity and interactions.

**Gradient boosting (XGBoost, LightGBM, CatBoost):** State-of-the-art for tabular data. Handles nonlinearity, interactions, and missing values.

### 10.6.2 Gradient Boosting

**Gradient boosting** builds an ensemble of weak learners (typically decision trees) sequentially, with each tree correcting the errors of the previous ones.

Mathematically, the model is:

$$
f_M(\mathbf{x}) = \sum_{m=1}^M \gamma_m h_m(\mathbf{x})
$$

where $ h_m $ are weak learners and $ \gamma_m $ are their weights. The weak learners are fit to the negative gradient of the loss:

$$
h_m = \arg\min_h \sum_{i=1}^N \left( -\frac{\partial \ell(y_i, f_{m-1}(\mathbf{x}_i))}{\partial f_{m-1}(\mathbf{x}_i)} - h(\mathbf{x}_i) \right)^2
$$

In practice, gradient boosting is implemented with libraries like XGBoost, LightGBM, and CatBoost, which add regularization, efficient tree construction, and handling of missing values.

### 10.6.3 Implementation

```python
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Train XGBoost
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
xgb.fit(X_train.numpy(), y_train.numpy().ravel())

# Predict
y_pred_xgb = xgb.predict(X_test.numpy())

# Evaluate
rmse_xgb = np.sqrt(mean_squared_error(y_test.numpy(), y_pred_xgb))
print(f"XGBoost RMSE: {rmse_xgb:.4f}")
```

### 10.6.4 Comparing Neural Networks and Gradient Boosting

The comparison should include:

**Predictive performance:** RMSE, MAE, R² on the test set.

**Training time:** How long does each model take to train?

**Inference time:** How long does each model take to predict?

**Robustness:** How sensitive is each model to hyperparameters?

**Interpretability:** Can you explain the model's predictions?

**Scalability:** How does each model scale with dataset size?

**Research relevance:** The comparison is the core of your research. Your contribution is not just "a neural network" but "a neural network that outperforms gradient boosting under conditions X, Y, Z."

### 10.6.5 Summary: Baselines

| Baseline | Type | Strengths | Weaknesses |
|----------|------|-----------|------------|
| Linear regression | Linear | Simple, interpretable | Cannot capture nonlinearity |
| Ridge | Linear + L2 | Handles multicollinearity | Still linear |
| Lasso | Linear + L1 | Feature selection | Still linear |
| Random forest | Ensemble | Handles nonlinearity | Can overfit |
| XGBoost | Boosting | State-of-the-art | Requires tuning |
| LightGBM | Boosting | Fast | Requires tuning |
| CatBoost | Boosting | Handles categorical | Slower |

> **Exercise 10.6:** Train XGBoost, LightGBM, and CatBoost on the welding dataset. Compare their RMSE, MAE, and R² with the MLP. Which performs best? Why?

---

## 10.7 Evaluation: RMSE, MAE, R², and Statistical Tests

### 10.7.1 Regression Metrics

**Mean Squared Error (MSE):**

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
$$

**Root Mean Squared Error (RMSE):**

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

**Mean Absolute Error (MAE):**

$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|
$$

**Coefficient of Determination (R²):**

$$
R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}
$$

where $ \bar{y} $ is the mean of the true targets.

**Research relevance:** RMSE penalizes large errors more than MAE. R² measures the proportion of variance explained. Report all three for a complete picture.

### 10.7.2 Statistical Significance

A single number is not enough. You must report the variance across runs and test whether the difference between models is statistically significant.

**Confidence intervals:** Report the mean and standard deviation over $ k $ runs (typically $ k \geq 5 \)).

**Paired t-test:** Test whether the difference between two models is significant.

**Wilcoxon signed-rank test:** Non-parametric alternative to the t-test.

```python
from scipy import stats

# Assume we have RMSE values for two models over 5 runs
rmse_model_a = [0.45, 0.47, 0.44, 0.46, 0.45]
rmse_model_b = [0.50, 0.52, 0.49, 0.51, 0.50]

t_stat, p_value = stats.ttest_rel(rmse_model_a, rmse_model_b)
print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")

if p_value < 0.05:
    print("The difference is statistically significant.")
else:
    print("The difference is not statistically significant.")
```

### 10.7.3 Multiple Runs

To estimate the variance of your results, run each model with different random seeds:

```python
def run_experiment(seed, model_fn, X_train, y_train, X_test, y_test):
    set_seed(seed)
    model = model_fn()
    # Train and evaluate
    return rmse

rmse_values = [run_experiment(seed, model_fn, ...) for seed in range(5)]
print(f"RMSE: {np.mean(rmse_values):.4f} ± {np.std(rmse_values):.4f}")
```

**Research relevance:** Reporting a single number is not enough. Report mean ± std over multiple runs, and test for statistical significance.

### 10.7.4 Summary: Evaluation

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MSE | $ \frac{1}{N} \sum (y - \hat{y})^2 $ | Squared error |
| RMSE | $ \sqrt{\text{MSE}} $ | Error in target units |
| MAE | $ \frac{1}{N} \sum \|y - \hat{y}\| $ | Robust error |
| R² | $ 1 - \frac{\text{SS}_\text{res}}{\text{SS}_\text{tot}} $ | Variance explained |
| Paired t-test | — | Significance |
| Wilcoxon | — | Non-parametric significance |

> **Exercise 10.7:** Run the MLP and XGBoost on the welding dataset with 5 different random seeds. Report the mean and std of RMSE, MAE, and R². Perform a paired t-test to compare the models.

---

## 10.8 Worked Example: Full Tabular Regression Pipeline

Let us combine everything into a complete pipeline for a tabular regression task.

### 10.8.1 The Data

We use the welding dataset: 480 samples, 4 features, 1 target.

### 10.8.2 The Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_and_preprocess(path):
    df = pd.read_excel(path)
    X = df[['Current', 'Angle', 'Speed', 'Time']].values
    y = df['Height'].values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_mlp(X_train, y_train, X_val, y_val, input_dim, n_epochs=200, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(input_dim).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    for epoch in range(n_epochs):
        model.train()
        yhat = model(X_train_t)
        loss = loss_fn(y_train_t, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return model

def evaluate(model, X_test, y_test):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        yhat = model(X_test_t).cpu().numpy().ravel()
    rmse = np.sqrt(mean_squared_error(y_test, yhat))
    mae = mean_absolute_error(y_test, yhat)
    r2 = r2_score(y_test, yhat)
    return rmse, mae, r2

# Main
X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess('welding_data.xlsx')

# MLP
mlp = train_mlp(X_train, y_train, X_val, y_val, input_dim=4)
rmse_mlp, mae_mlp, r2_mlp = evaluate(mlp, X_test, y_test)
print(f"MLP: RMSE={rmse_mlp:.4f}, MAE={mae_mlp:.4f}, R²={r2_mlp:.4f}")

# XGBoost
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"XGBoost: RMSE={rmse_xgb:.4f}, MAE={mae_xgb:.4f}, R²={r2_xgb:.4f}")
```

### 10.8.3 Results

The output might be:

```
MLP: RMSE=0.4523, MAE=0.3412, R²=0.8712
XGBoost: RMSE=0.4215, MAE=0.3189, R²=0.8891
```

XGBoost outperforms the MLP on all three metrics. This is consistent with the literature: tree-based models are strong baselines for tabular data.

### 10.8.4 Reflection

The comparison between MLP and XGBoost is the core of the research. The next step is to investigate *why* XGBoost performs better:

- Is the dataset small? (Yes, 480 samples.)
- Are the relationships axis-aligned? (Likely, given the physics of welding.)
- Are there interactions? (Possible, but tree-based models capture them.)

The research question becomes: *Under what conditions does an MLP outperform XGBoost on this dataset?* Possible directions: larger dataset, more complex features, different architecture, different loss function.

**Research relevance:** This is the essence of tabular research: comparing models, understanding when each is preferable, and proposing conditions for improvement.

> **Exercise 10.8:** Extend the pipeline above to include:
>
> 1. A deeper MLP with 3 hidden layers.
> 2. A wider MLP with 256 hidden units.
> 3. Dropout regularization.
> 4. A comparison with Ridge and Lasso.
> 5. A learning rate schedule.

---

## 10.9 Common Errors and Debugging

### 10.9.1 Error: Data Leakage

**Symptom:** Validation performance is suspiciously high.

**Cause:** Preprocessing was fit on the full dataset.

**Fix:** Split first, then fit preprocessing on the training set only.

### 10.9.2 Error: NaN Loss

**Symptom:** Loss becomes NaN during training.

**Cause:** Learning rate too high, numerical instability, or missing values.

**Fix:** Reduce learning rate, check for missing values, use gradient clipping.

### 10.9.3 Error: Poor Generalization

**Symptom:** Training loss is low but validation loss is high.

**Cause:** Overfitting, insufficient data, or too large a model.

**Fix:** Add regularization (dropout, weight decay), reduce model size, or collect more data.

### 10.9.4 Error: Slow Convergence

**Symptom:** Loss decreases slowly.

**Cause:** Learning rate too low, features not normalized, or poor initialization.

**Fix:** Increase learning rate, normalize features, use better initialization.

### 10.9.5 Summary: Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| Data leakage | Preprocessing on full dataset | Split first |
| NaN loss | High LR, missing values | Reduce LR, impute |
| Poor generalization | Overfitting | Regularize |
| Slow convergence | Low LR, unnormalized features | Increase LR, normalize |

> **Exercise 10.9:** For each of the following scenarios, diagnose the issue and propose a fix:
>
> 1. Training loss decreases but validation loss increases.
> 2. Loss is NaN after 5 epochs.
> 3. Loss decreases very slowly.
> 4. Validation performance is 99% but test performance is 50%.

---

## 10.10 Research Application: When Do Neural Networks Beat GBMs?

### 10.10.1 The Research Question

The central research question in tabular data is: *When do neural networks outperform gradient boosting?*

The literature suggests several factors:

**Dataset size:** Neural networks improve with more data. On small datasets (< 10,000 samples), GBMs often win. On large datasets (> 100,000 samples), neural networks can win.

**Feature interactions:** Neural networks can learn complex interactions. GBMs can too, but may require more trees.

**Categorical features:** GBMs (especially CatBoost) handle categorical features well. Neural networks require embedding layers.

**Missing values:** GBMs handle missing values natively. Neural networks require imputation.

**Interpretability:** GBMs provide feature importance. Neural networks are black boxes.

**Compute:** Neural networks require GPUs. GBMs run on CPUs.

### 10.10.2 Designing a Study

To investigate this question, design a study:

**Datasets:** Select 5–10 tabular datasets with varying sizes and feature types.

**Models:** Train MLP, XGBoost, LightGBM, and CatBoost on each.

**Metrics:** Report RMSE, MAE, R², training time, and inference time.

**Analysis:** Identify the conditions under which neural networks outperform GBMs.

### 10.10.3 Summary: When NN Beats GBM

| Factor | NN Wins | GBM Wins |
|--------|---------|----------|
| Dataset size | Large | Small |
| Feature interactions | Complex | Simple |
| Categorical features | With embeddings | Native |
| Missing values | After imputation | Native |
| Compute | GPU available | CPU only |

> **Exercise 10.10:** Design a study to investigate when neural networks outperform gradient boosting on tabular data. Select 3 datasets, train 2 neural networks and 2 GBMs, and report your findings.

---

## 10.11 Module Summary

| Concept | Mathematical Form | PyTorch | Research Relevance |
|---------|-------------------|---------|-------------------|
| Tabular data | $ X \in \mathbb{R}^{N \times d} $ | `pd.read_excel` | Industry, science |
| Regression | $ \hat{y} = f_\theta(\mathbf{x}) $ | `nn.Linear` | Continuous targets |
| MSE | $ \frac{1}{N} \sum (y - \hat{y})^2 $ | `nn.MSELoss` | Loss function |
| RMSE | $ \sqrt{\text{MSE}} $ | `np.sqrt` | Interpretable metric |
| MAE | $ \frac{1}{N} \sum \|y - \hat{y}\| $ | `nn.L1Loss` | Robust metric |
| R² | $ 1 - \frac{\text{SS}_\text{res}}{\text{SS}_\text{tot}} $ | `r2_score` | Variance explained |
| Preprocessing | Impute, encode, normalize | `sklearn` | Data quality |
| Data leakage | Fit on train only | — | Reproducibility |
| Baselines | Linear, Ridge, Lasso, GBM | `sklearn`, `xgboost` | Context |
| Statistical tests | Paired t-test, Wilcoxon | `scipy.stats` | Significance |
| Multiple runs | Mean ± std | — | Variance |

---

## 10.12 Capstone Thread: Tabular Data for Your Project

By the end of this module, you will be able to load and preprocess tabular data from Excel/CSV; implement regression models in PyTorch; evaluate them with appropriate metrics; compare against gradient boosting baselines; perform statistical significance testing; and articulate when neural networks are preferable to tree-based methods.

For your capstone project, you will use these skills to load your data, preprocess it correctly, implement a regression model, compare against baselines, and report your results.

In Module 11, we will apply the same principles to **image data and computer vision research**.

---

## 10.13 Additional Resources

### On Tabular Data

- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). "Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data?" *NeurIPS*.
- Shwartz-Ziv, R., & Armon, A. (2022). "Tabular Data: Deep Learning is Not All You Need." *Information Fusion*.
- Borisov, V., et al. (2021). "Deep Neural Networks and Tabular Data: A Survey." *IEEE TNNLS*.

### On Gradient Boosting

- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *KDD*.
- Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS*.
- Prokhorenkova, L., et al. (2018). "CatBoost: Unbiased Boosting with Categorical Features." *NeurIPS*.

### On Regression

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- James, G., et al. (2013). *An Introduction to Statistical Learning*. Springer.

### On Evaluation

- Willmott, C. J., & Matsuura, K. (2005). "Advantages of the Mean Absolute Error (MAE) over the Root Mean Square Error (RMSE)." *Climate Research*.
- Demšar, J. (2006). "Statistical Comparisons of Classifiers over Multiple Data Sets." *JMLR*.

### On Datasets

- UCI Machine Learning Repository: https://archive.ics.uci.edu/
- Kaggle Datasets: https://www.kaggle.com/datasets
- OpenML: https://www.openml.org/

---

*End of Module 10*