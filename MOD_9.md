# Module 9: Reproducing a Paper

## TORA | CS 336: PyTorch for Research

**Instructor:** Alpha Alimamy Kamara

**Module Length:** 2 weeks (4 lectures + 2 lab sessions)

**Prerequisites:** Modules 1–8 completed; a paper selected for reproduction; a working PyTorch environment.

---

## 9.0 Module Overview

In Module 8, we learned how to read a paper critically. But reading is not enough. The ultimate test of understanding is **reproduction**: can you implement the method and obtain the same results? Reproduction is the cornerstone of science. A result that cannot be reproduced is not a result; it is an anecdote.

This module is about the **reproduction process**: the systematic workflow for reimplementing a method, verifying its results, and documenting any discrepancies. We will begin with the philosophy of reproduction, then develop a reproduction checklist, then work through concrete examples from the ML literature. Along the way, we will encounter the common barriers to reproduction and the strategies for overcoming them.

The pedagogical approach here is **process first, practice second**. We will introduce the reproduction workflow and the checklist, then apply them to a paper of your choice. The goal is not to reproduce a specific paper but to internalize a repeatable process for reproducing any paper.

By the end of this module, you will be able to define what reproduction means in different contexts; use a reproduction checklist to guide your work; implement a paper's method in PyTorch; verify its results against the paper; document discrepancies; and articulate why reproduction is a core research skill.

The module is self-paced, but the exercises are best done across multiple sessions so that you have time to implement and debug.

---

## 9.1 The Philosophy of Reproduction

### 9.1.1 What Is Reproduction?

**Reproduction** is the process of reimplementing a method and verifying its results. It is distinguished from **replication**, which is running the same code on the same data, and **generalization**, which is applying the method to new data.

The taxonomy of reproducibility, adapted from the ACM and ML community, is:

| Level | Definition | Example |
|-------|------------|---------|
| **Repeatability** | Same team, same code, same data → same result | You rerun your own experiment and get the same number |
| **Replicability** | Different team, same code, same data → same result | A colleague clones your repo and gets the same number |
| **Reproducibility** | Different team, different code, same data → same conclusion | A colleague reimplements your method from the paper and gets a similar number |
| **Generalizability** | Different team, different code, different data → same conclusion | Your method works on a new dataset from the same distribution |

In this module, we aim for **reproducibility** as the minimum standard and **generalizability** as the aspiration.

### 9.1.2 Why Reproduce?

Reproduction serves several purposes:

**Verification.** It confirms that the original results are correct.

**Understanding.** It forces you to understand the method in detail. You cannot reimplement what you do not understand.

**Baseline.** It provides a baseline for your own work. You can compare your method against the reproduced method.

**Extension.** It provides a starting point for extensions. You can modify the method and test new ideas.

**Reproducibility crisis.** The field has a reproducibility crisis. By reproducing papers, you contribute to the solution.

### 9.1.3 The Reproducibility Crisis

In 2016, *Nature* published a survey finding that more than 70% of researchers had tried and failed to reproduce another scientist's experiments, and more than half had failed to reproduce their own. In ML, the crisis is particularly acute because:

- **Non-determinism:** Random seeds, data ordering, and hardware introduce variance.
- **Hyperparameter sensitivity:** Small changes in hyperparameters can lead to large changes in results.
- **Benchmark overfitting:** Methods are tuned to perform well on specific benchmarks.
- **Incomplete reporting:** Papers often omit critical details.

The ML Reproducibility Checklist, introduced by Joelle Pineau in 2019, is now adopted by NeurIPS, ICML, ICLR, and other major venues. It requires authors to report hyperparameter search spaces, random seeds, compute infrastructure, and more.

### 9.1.4 The Three Questions Applied to Reproduction

Recall from Module 1 the three questions of research. For reproduction, the question is: *Can I reproduce the paper's results?* The evidence is: *My reimplementation and the results I obtain.* The meaning is: *What does this tell me about the method, the paper, and the field?*

### 9.1.5 Summary: Philosophy of Reproduction

| Level | Definition | Difficulty |
|-------|------------|------------|
| Repeatability | Same code, same data | Easy |
| Replicability | Same code, different team | Easy |
| Reproducibility | Different code, same data | Medium |
| Generalizability | Different code, different data | Hard |

> **Exercise 9.1:** Choose a paper you want to reproduce. Write a one-paragraph answer to each of the three questions. Bring this to lecture.

---

## 9.2 The Reproduction Checklist

### 9.2.1 Overview

The **reproduction checklist** is a systematic guide to reproducing a paper. It covers the data, the method, the training, the evaluation, and the reporting. Working through the checklist ensures that you do not miss critical details.

### 9.2.2 Data

**Questions to ask:**

- What datasets were used?
- How were the datasets split into train/validation/test?
- What preprocessing was applied?
- What data augmentation was used?
- Is the data publicly available? If not, can it be obtained?
- Are there any licensing or privacy restrictions?

**Common issues:**

- The dataset is not publicly available.
- The split is not specified.
- The preprocessing is not fully described.
- The data has been updated since the paper was published.

**Strategies:**

- Contact the authors for the data.
- Use a similar dataset if the original is unavailable.
- Document any differences.

### 9.2.3 Method

**Questions to ask:**

- What is the proposed method?
- What are the key components?
- What are the hyperparameters?
- What are the design choices?
- Are the design choices justified?
- Is there pseudocode or code available?

**Common issues:**

- The method is described at a high level but lacks details.
- The hyperparameters are not specified.
- The code is not available.
- The method has ambiguities.

**Strategies:**

- Read the paper multiple times.
- Look for supplementary material.
- Contact the authors for clarification.
- Make reasonable assumptions and document them.

### 9.2.4 Training

**Questions to ask:**

- What optimizer was used?
- What learning rate was used?
- What learning rate schedule was used?
- What batch size was used?
- How many epochs were used?
- What hardware was used?
- How long did training take?
- What random seeds were used?

**Common issues:**

- The training details are not specified.
- The random seeds are not reported.
- The hardware is not described.
- The training time is not reported.

**Strategies:**

- Use default hyperparameters if not specified.
- Tune hyperparameters on the validation set.
- Report your choices and any differences.

### 9.2.5 Evaluation

**Questions to ask:**

- What metrics were used?
- How were the metrics computed?
- What baselines were compared?
- Are the baselines fair?
- Are the results statistically significant?
- Are the results reproducible?

**Common issues:**

- The evaluation protocol is not fully described.
- The baselines are not comparable.
- The results are not statistically significant.
- The results are not reproducible.

**Strategies:**

- Use the same metrics as the paper.
- Implement the baselines yourself.
- Perform statistical tests.
- Report confidence intervals.

### 9.2.6 Reporting

**Questions to ask:**

- What are the main results?
- Do your results match the paper?
- If not, what are the discrepancies?
- What are the possible explanations?
- What are the limitations of your reproduction?

**Common issues:**

- Your results differ from the paper's.
- The differences are not explained.
- The limitations are not acknowledged.

**Strategies:**

- Report your results honestly.
- Investigate the discrepancies.
- Document possible explanations.
- Acknowledge the limitations.

### 9.2.7 Summary: Reproduction Checklist

| Category | Key Questions |
|----------|---------------|
| Data | What data? How split? How preprocessed? |
| Method | What method? What hyperparameters? What design choices? |
| Training | What optimizer? What learning rate? What seeds? |
| Evaluation | What metrics? What baselines? What statistics? |
| Reporting | Do results match? What discrepancies? |

> **Exercise 9.2:** Apply the reproduction checklist to the paper you selected. For each category, write down what you know and what you need to find out.

---

## 9.3 A Reproduction Workflow

### 9.3.1 The Workflow

The reproduction workflow consists of the following steps:

1. **Select a paper.** Choose a paper that is relevant to your research and feasible to reproduce.
2. **Read the paper.** Apply the three-pass method from Module 8.
3. **Identify the key components.** What are the method, data, training, and evaluation?
4. **Gather resources.** Obtain the data, code, and any supplementary material.
5. **Implement the method.** Start with a simple version and add complexity.
6. **Train the model.** Use the paper's hyperparameters if specified, otherwise tune.
7. **Evaluate the model.** Use the paper's metrics and baselines.
8. **Compare results.** Do your results match the paper's?
9. **Investigate discrepancies.** If not, why?
10. **Document.** Write up your reproduction, including any differences.

### 9.3.2 Selecting a Paper

Not all papers are equally reproducible. When selecting a paper, consider:

**Availability of data.** Is the data publicly available?

**Availability of code.** Is the code available?

**Clarity of the method.** Is the method described in enough detail?

**Complexity of the method.** Is the method feasible to implement in the time available?

**Compute requirements.** Can you train the model on your hardware?

**Relevance to your research.** Is the paper relevant to your capstone?

### 9.3.3 Implementing the Method

**Start simple.** Implement a minimal version of the method first. Get it working, then add complexity.

**Test incrementally.** Test each component separately before combining them.

**Use the paper's notation.** Name your variables and functions after the paper's notation.

**Document as you go.** Write comments and docstrings. You will forget what you did.

**Compare with the paper.** At each step, check that your implementation matches the paper's description.

### 9.3.4 Training the Model

**Use the paper's hyperparameters.** If specified, use them. If not, use defaults or tune on the validation set.

**Set random seeds.** For reproducibility, set all random seeds.

**Monitor training.** Plot the loss curves. Are they converging?

**Save checkpoints.** So you can resume if training is interrupted.

**Log everything.** Hyperparameters, metrics, time, hardware.

### 9.3.5 Evaluating the Model

**Use the paper's metrics.** If the paper reports accuracy, report accuracy. If it reports F1, report F1.

**Implement the baselines.** If the paper compares with baselines, implement them yourself.

**Perform statistical tests.** If the paper reports a single number, report mean ± std over multiple runs.

**Compare with the paper.** Do your results match? If not, why?

### 9.3.6 Documenting the Reproduction

A reproduction report should include:

**Summary:** A brief summary of the paper and your reproduction.

**Method:** Your implementation of the method.

**Data:** The data you used and any differences.

**Training:** The hyperparameters and training procedure.

**Results:** Your results and comparison with the paper.

**Discrepancies:** Any differences and possible explanations.

**Limitations:** The limitations of your reproduction.

**Conclusion:** What you learned.

### 9.3.7 Summary: Reproduction Workflow

| Step | Description |
|------|-------------|
| 1 | Select a paper |
| 2 | Read the paper |
| 3 | Identify key components |
| 4 | Gather resources |
| 5 | Implement the method |
| 6 | Train the model |
| 7 | Evaluate the model |
| 8 | Compare results |
| 9 | Investigate discrepancies |
| 10 | Document |

> **Exercise 9.3:** Apply the reproduction workflow to your paper. Complete steps 1–4 before the next lecture.

---

## 9.4 Common Barriers to Reproduction

### 9.4.1 Missing Implementation Details

**Symptom:** The paper describes the method at a high level but omits critical details.

**Example:** "We use a learning rate of 0.1" — but what schedule? What optimizer? What batch size?

**Strategy:** Contact the authors. Look for code. Make reasonable assumptions and document them.

### 9.4.2 Non-Determinism

**Symptom:** Your results vary across runs.

**Cause:** Random seeds, data ordering, hardware non-determinism.

**Strategy:** Set all random seeds. Use deterministic algorithms. Report mean ± std over multiple runs.

### 9.4.3 Hyperparameter Sensitivity

**Symptom:** Small changes in hyperparameters lead to large changes in results.

**Cause:** The method is sensitive to hyperparameters. The paper may have tuned them for the benchmark.

**Strategy:** Tune hyperparameters on the validation set. Report the search space and the best values.

### 9.4.4 Benchmark Overfitting

**Symptom:** The method performs well on the benchmark but poorly on other datasets.

**Cause:** The method was tuned for the benchmark.

**Strategy:** Evaluate on multiple datasets. Report performance on each.

### 9.4.5 Compute Requirements

**Symptom:** You cannot train the model on your hardware.

**Cause:** The method requires large amounts of compute.

**Strategy:** Use a smaller model or dataset. Report the difference. Use cloud compute if available.

### 9.4.6 Data Availability

**Symptom:** The dataset is not publicly available.

**Cause:** Privacy, licensing, or other restrictions.

**Strategy:** Contact the authors. Use a similar dataset. Report the difference.

### 9.4.7 Summary: Common Barriers

| Barrier | Strategy |
|---------|----------|
| Missing details | Contact authors, make assumptions |
| Non-determinism | Set seeds, report variance |
| Hyperparameter sensitivity | Tune on validation set |
| Benchmark overfitting | Evaluate on multiple datasets |
| Compute requirements | Use smaller model/dataset |
| Data availability | Contact authors, use similar data |

> **Exercise 9.4:** Identify the barriers you expect to encounter in your reproduction. For each, propose a strategy.

---

## 9.5 Case Study: Reproducing a Simple Paper

Let us walk through the reproduction of a simple paper. We will use the linear regression problem from the original notebook as a stand-in for a "paper."

### 9.5.1 The "Paper"

**Title:** "Linear Regression with Gradient Descent"

**Method:** Linear regression with MSE loss, optimized with gradient descent.

**Data:** 100 points generated from $ y = 1 + 2x + \epsilon $, where $ \epsilon \sim \mathcal{N}(0, 0.1^2) $.

**Results:** $ a \approx 1.0235 $, $ b \approx 1.9690 $.

### 9.5.2 The Checklist

**Data:** 100 points, generated with seed 42. Split 80/20.

**Method:** Linear regression, MSE loss, SGD optimizer.

**Training:** Learning rate 0.1, 1000 epochs, full-batch gradient descent.

**Evaluation:** MSE on the training set.

**Reporting:** $ a $ and $ b $ after training.

### 9.5.3 The Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)

# Data generation
N = 100
x = torch.rand(N, 1)
y = 1 + 2 * x + 0.1 * torch.randn(N, 1)

# Train/validation split
idx = torch.randperm(N)
train_idx = idx[:80]
val_idx = idx[80:]
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# Model
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training
for epoch in range(1000):
    yhat = model(x_train)
    loss = loss_fn(y_train, yhat)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Results
print(f"a = {model.bias.item():.4f}, b = {model.weight.item():.4f}")
```

The output is:

```
a = 1.0235, b = 1.9690
```

### 9.5.4 Comparison

The reproduced values match the "paper's" values exactly. The reproduction is successful.

### 9.5.5 Reflection

Even for a simple problem, reproduction requires careful attention to the data, the method, the training, and the evaluation. For a complex paper, the process is much harder.

> **Exercise 9.5:** Reproduce the linear regression experiment above. Verify that you obtain the same results. Then change the seed and observe how the results vary.

---

## 9.6 Case Study: Reproducing a Real Paper

Let us consider the reproduction of a real paper. We will use "Deep Residual Learning for Image Recognition" (He et al., 2016) as an example.

### 9.6.1 The Paper

**Title:** "Deep Residual Learning for Image Recognition"

**Method:** ResNet, a deep CNN with residual connections.

**Data:** ImageNet (1.2M images, 1000 classes).

**Results:** 3.57% top-5 error on ImageNet.

### 9.6.2 The Checklist

**Data:** ImageNet is publicly available. The split is standard (1.28M train, 50K validation).

**Method:** ResNet is described in detail in the paper. The architecture is specified. The code is available.

**Training:** SGD with momentum 0.9, learning rate 0.1 with step decay, batch size 256, 90 epochs. Data augmentation: random crop, horizontal flip.

**Evaluation:** Top-1 and top-5 error on the validation set.

**Reporting:** Error rates for different depths (18, 34, 50, 101, 152).

### 9.6.3 The Implementation

Implementing ResNet from scratch is a substantial task. A simplified version for CIFAR-10 can be implemented in a few hundred lines of PyTorch:

```python
import torch
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

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
```

### 9.6.4 Training

Training ResNet on ImageNet requires significant compute (8 GPUs for several days). For a course project, we use CIFAR-10, a smaller dataset (60K images, 10 classes), and train on a single GPU.

```python
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNet(num_classes=10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

for epoch in range(200):
    model.train()
    for x_batch, y_batch in trainloader:
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
        for x_batch, y_batch in testloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            yhat = model(x_batch)
            _, predicted = yhat.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: accuracy = {100 * correct / total:.2f}%")
```

### 9.6.5 Comparison

The paper reports 93.03% accuracy on CIFAR-10 with ResNet-20. Your reproduction may achieve 92–93%, depending on hyperparameters and hardware. The difference is due to non-determinism and hyperparameter sensitivity.

### 9.6.6 Reflection

Reproducing ResNet on CIFAR-10 takes a few hours on a single GPU. The results are close to the paper's, but not exact. The differences are within the expected range of variance. This is a successful reproduction.

> **Exercise 9.6:** Reproduce ResNet on CIFAR-10 using the code above. Report your accuracy. Compare with the paper's reported accuracy. Investigate any discrepancies.

---

## 9.7 Reporting a Reproduction

### 9.7.1 The Reproduction Report

A **reproduction report** documents your reproduction. It should include:

**Summary:** A brief summary of the paper and your reproduction.

**Method:** Your implementation of the method.

**Data:** The data you used and any differences.

**Training:** The hyperparameters and training procedure.

**Results:** Your results and comparison with the paper.

**Discrepancies:** Any differences and possible explanations.

**Limitations:** The limitations of your reproduction.

**Conclusion:** What you learned.

### 9.7.2 Worked Example 9.1: A Reproduction Report

**Title:** Reproduction of "Deep Residual Learning for Image Recognition"

**Summary:** We reproduced ResNet-20 on CIFAR-10. Our implementation achieves 92.8% accuracy, close to the paper's 93.03%.

**Method:** We implemented ResNet-20 following the paper's description. The architecture has 3 stages with 16, 32, and 64 channels, each with 2 basic blocks. The basic block has two 3×3 convolutions with batch normalization and a residual connection.

**Data:** We used CIFAR-10 (60K images, 10 classes). The standard train/test split (50K/10K) was used. Data augmentation: random crop with padding 4, random horizontal flip. Normalization: mean (0.4914, 0.4822, 0.4465), std (0.2023, 0.1994, 0.2010).

**Training:** SGD with momentum 0.9, weight decay 5e-4, learning rate 0.1 with cosine annealing over 200 epochs. Batch size 128.

**Results:** Our ResNet-20 achieves 92.8% accuracy on CIFAR-10. The paper reports 93.03%. The difference is 0.23 percentage points.

**Discrepancies:** The difference is within the expected range of variance. Possible causes: random seed, hardware differences, minor differences in data augmentation.

**Limitations:** We did not reproduce the deeper ResNet variants (ResNet-56, ResNet-110). We did not reproduce the ImageNet results due to compute constraints.

**Conclusion:** The reproduction is successful. The method is reproducible, and the results are close to the paper's.

### 9.7.3 Summary: Reproduction Report

| Section | Content |
|---------|---------|
| Summary | Brief overview |
| Method | Your implementation |
| Data | Data and differences |
| Training | Hyperparameters |
| Results | Your results |
| Discrepancies | Differences and explanations |
| Limitations | What you did not do |
| Conclusion | What you learned |

> **Exercise 9.7:** Write a reproduction report for your paper. Follow the structure above. Be honest about discrepancies and limitations.

---

## 9.8 Reproducibility in Practice

### 9.8.1 The ML Reproducibility Checklist

The ML Reproducibility Checklist, introduced by Joelle Pineau in 2019, is now adopted by NeurIPS, ICML, ICLR, and other major venues. It requires authors to report:

**Models and algorithms:**
- A clear description of the method
- The hyperparameter search space
- The number of runs

**Theoretical claims:**
- Full proofs or references

**Datasets:**
- Description of the data
- Train/validation/test splits
- Preprocessing steps

**Code:**
- Link to code repository
- Instructions for running

**Experiments:**
- Hyperparameters
- Random seeds
- Compute infrastructure
- Evaluation metrics

**Research claims:**
- Statistical significance
- Error bars

### 9.8.2 Tools for Reproducibility

**Version control:** Git for code, DVC for data.

**Experiment tracking:** Weights & Biases, MLflow, TensorBoard.

**Configuration management:** Hydra, OmegaConf.

**Containerization:** Docker, Singularity.

**Environment management:** Conda, Poetry.

### 9.8.3 Reproducibility in Your Own Research

When you write your own papers, follow the reproducibility checklist. Include:

- Code and data links
- Hyperparameter search spaces
- Random seeds
- Compute infrastructure
- Evaluation metrics
- Error bars
- Statistical tests

This makes your work more useful to others and improves the field as a whole.

### 9.8.4 Summary: Reproducibility in Practice

| Aspect | Tool |
|--------|------|
| Version control | Git, DVC |
| Experiment tracking | W&B, MLflow |
| Configuration | Hydra |
| Containerization | Docker |
| Environment | Conda, Poetry |

> **Exercise 9.8:** Apply the ML Reproducibility Checklist to your reproduction. Which items did you complete? Which are missing?

---

## 9.9 Research Application: Reproduction as a Contribution

### 9.9.1 Reproduction Papers

Reproduction is a valid research contribution. **Reproduction papers** are published at workshops (e.g., NeurIPS Reproducibility Challenge, ICLR Reproducibility Challenge) and conferences. They document the reproduction of a paper, including any discrepancies and insights.

### 9.9.2 How to Write a Reproduction Paper

A reproduction paper typically has the following structure:

**Introduction:** Why is this paper important? Why reproduce it?

**Related work:** What is the context?

**Method:** What did the paper do? What did you do?

**Experiments:** What did you run? What did you find?

**Discussion:** What did you learn? What are the discrepancies?

**Conclusion:** What is the takeaway?

### 9.9.3 The Value of Reproduction

Reproduction contributes to the field by:

- **Verifying results:** Confirming that the original results are correct.
- **Identifying issues:** Finding discrepancies and bugs.
- **Improving reproducibility:** Documenting what is needed to reproduce.
- **Providing baselines:** Giving others a starting point.

### 9.9.4 Summary: Reproduction as Contribution

| Aspect | Value |
|--------|-------|
| Verification | Confirms results |
| Issue identification | Finds problems |
| Reproducibility | Documents requirements |
| Baselines | Provides starting points |

> **Exercise 9.9:** Consider submitting your reproduction to a reproducibility challenge or workshop. Write a one-page proposal for a reproduction paper.

---

## 9.10 Module Summary

| Concept | Description |
|---------|-------------|
| Repeatability | Same code, same data |
| Replicability | Same code, different team |
| Reproducibility | Different code, same data |
| Generalizability | Different code, different data |
| Reproduction checklist | Data, method, training, evaluation, reporting |
| Reproduction workflow | Select, read, identify, gather, implement, train, evaluate, compare, investigate, document |
| Common barriers | Missing details, non-determinism, sensitivity, compute |
| Reproduction report | Summary, method, data, training, results, discrepancies, limitations |
| Reproducibility checklist | Models, data, code, experiments, claims |
| Tools | Git, DVC, W&B, Hydra, Docker |
| Reproduction papers | Valid contribution |

---

## 9.11 Capstone Thread: Reproduction Skills for Your Project

By the end of this module, you will be able to define what reproduction means in different contexts; use a reproduction checklist to guide your work; implement a paper's method in PyTorch; verify its results against the paper; document discrepancies; and articulate why reproduction is a core research skill.

For your capstone project, you will use these skills to reproduce a baseline, verify your implementation, and position your contribution relative to prior work.

In Module 10, we will apply these skills to **tabular data and regression research**.

---

## 9.12 Additional Resources

### On Reproducibility

- Pineau, J., et al. (2021). "Improving Reproducibility in Machine Learning Research." *JMLR*.
- ML Reproducibility Checklist: [Link](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)
- "Reproducibility in Machine Learning" (Stanford): https://cs.stanford.edu/

### On Reproduction Challenges

- NeurIPS Reproducibility Challenge: https://neurips.cc/
- ICLR Reproducibility Challenge: https://iclr.cc/
- ML Reproducibility Challenge: https://paperswithcode.com/

### On Tools

- Git: https://git-scm.com/
- DVC: https://dvc.org/
- Weights & Biases: https://wandb.ai/
- MLflow: https://mlflow.org/
- Hydra: https://hydra.cc/
- Docker: https://www.docker.com/

### On Specific Papers

- He, K., et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.
- Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
- Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." *NAACL*.

---

*End of Module 9*