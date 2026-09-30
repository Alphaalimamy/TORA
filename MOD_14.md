# Module 14: Capstone Project Options

## TORA CS 336: PyTorch for Research

**Institution:** The Open Research Academy (TORA)

**Instructor:** Professor [Your Name]

**Module Length:** 1 week (1 lecture + 1 lab session + 1 individual meeting)

**Prerequisites:** Module 13 completed; a working PyTorch environment; access to at least one dataset; familiarity with the capstone rubric.

---

## 14.0 Module Overview

In Module 13, we established the guidelines for the capstone project: what is expected, how it will be evaluated, and how to manage your time. But guidelines alone do not tell you *what* to do. The hardest part of research is often choosing a question.

This module provides **concrete project options** — five templates that you can adapt, extend, or use as inspiration. Each option includes a research question, a dataset, a method, experiments, metrics, and a timeline. The options span the three applied modules (tabular, image, and advanced architectures) and include a reproducibility study for those who prefer to build on prior work.

The pedagogical approach here is **example first, adaptation second**. We present complete project templates, then guide you through the process of adapting one to your own interests and resources. The goal is not to prescribe a project but to demonstrate what a well-scoped project looks like.

By the end of this module, you will be able to choose a capstone project; adapt a template to your interests; identify the resources you need; write a project proposal; and articulate why your project is feasible within the 12-week timeline.

The module is self-paced, but the individual meeting is scheduled. Come with a draft proposal.

---

## 14.1 How to Choose a Project

### 14.1.1 The Three Criteria

A good capstone project satisfies three criteria:

**Interest.** You should be genuinely curious about the question. You will spend 12 weeks on it; if you are bored, the project will suffer.

**Feasibility.** The project should be completable within 12 weeks with the resources you have. A project that requires 1,000 GPUs is not feasible.

**Contribution.** The project should make a small but real contribution to the field. It should answer a question that has not been definitively answered.

### 14.1.2 The Decision Framework

When choosing a project, ask yourself:

**What data do I have access to?** Tabular, image, text, audio, multimodal?

**What compute do I have?** CPU only, single GPU, multi-GPU, cloud?

**What skills do I want to develop?** Model design, data pipeline, evaluation, writing?

**What papers have I read that inspired me?** What gaps did I notice?

**What would I be excited to work on for 12 weeks?**

### 14.1.3 Common Mistakes in Project Selection

**Too ambitious.** "Train a GPT-4 from scratch" is not feasible.

**Too vague.** "Improve deep learning" is not a project.

**Too derivative.** "Reproduce ResNet" is not original research.

**Too disconnected.** A project that does not connect to your interests or skills will be a slog.

### 14.1.4 Summary: Choosing a Project

| Criterion | Question |
|-----------|----------|
| Interest | Am I genuinely curious? |
| Feasibility | Can I finish in 12 weeks? |
| Contribution | Does it answer a new question? |
| Data | What data do I have? |
| Compute | What compute do I have? |
| Skills | What do I want to learn? |

> **Exercise 14.1:** Write down your answers to the six questions above. Based on your answers, identify which of the five project options (below) is the best fit.

---

## 14.2 Option A: Tabular Data Research

### 14.2.1 The Research Question

**Question:** Under what conditions do neural networks outperform gradient boosting on tabular regression tasks?

This question is central to the tabular data literature. The 2022 paper by Grinsztajn et al. showed that tree-based models outperform neural networks on most tabular datasets, but the conditions under which neural networks win are not fully understood.

### 14.2.2 The Data

Use a collection of tabular datasets with varying characteristics:

| Dataset | Samples | Features | Task |
|---------|---------|----------|------|
| California Housing | 20,640 | 8 | Regression |
| Concrete Strength | 1,030 | 8 | Regression |
| Energy Efficiency | 768 | 8 | Regression |
| Welding | 480 | 4 | Regression |
| Protein Structure | 45,730 | 9 | Regression |

All are publicly available from the UCI ML Repository.

### 14.2.3 The Method

Train the following models on each dataset:

**Baselines:** Linear regression, Ridge, Lasso, Random Forest, XGBoost, LightGBM, CatBoost.

**Neural networks:** MLP with 2–3 hidden layers, varying width (64, 128, 256), dropout, and weight decay.

**Hybrid:** A neural network with feature engineering (e.g., interactions, polynomial features).

### 14.2.4 The Experiments

For each dataset and model:

1. Split into train/val/test (70/15/15).
2. Normalize features (fit on train).
3. Train with 3 random seeds.
4. Report RMSE, MAE, R² on the test set.
5. Perform paired t-tests between models.

### 14.2.5 The Metrics

**Primary:** RMSE.

**Secondary:** MAE, R².

**Statistical:** Paired t-test, Wilcoxon signed-rank test.

### 14.2.6 The Contribution

A decision framework: "Use neural networks when the dataset has X samples, Y features, and Z characteristics; use gradient boosting otherwise."

### 14.2.7 Summary: Option A

| Aspect | Description |
|--------|-------------|
| Question | When do NNs beat GBMs? |
| Data | UCI regression datasets |
| Method | MLP vs. GBM |
| Experiments | 5 datasets × 8 models × 3 seeds |
| Metrics | RMSE, MAE, R² |
| Contribution | Decision framework |

> **Exercise 14.2:** Adapt Option A to a tabular dataset you have access to. Write a one-page proposal.

---

## 14.3 Option B: Image Classification Research

### 14.3.1 The Research Question

**Question:** How does transfer learning performance vary with dataset size and domain similarity?

This question is central to applied computer vision. Transfer learning is standard practice, but the relationship between dataset size, domain similarity, and performance is not fully understood.

### 14.3.2 The Data

Use datasets with varying degrees of similarity to ImageNet:

| Dataset | Classes | Images | Similarity to ImageNet |
|---------|---------|--------|------------------------|
| CIFAR-10 | 10 | 60,000 | Medium |
| CIFAR-100 | 100 | 60,000 | Medium |
| Oxford Flowers-102 | 102 | 8,000 | Medium |
| Stanford Cars | 196 | 16,185 | Medium |
| Fashion-MNIST | 10 | 70,000 | Low |

### 14.3.3 The Method

Fine-tune a pre-trained ResNet-50 on each dataset with three strategies:

**Feature extraction:** Freeze the feature extractor, train only the classifier.

**Fine-tuning:** Unfreeze all layers, train with a small learning rate.

**Discriminative fine-tuning:** Use different learning rates for different layers.

### 14.3.4 The Experiments

For each dataset and strategy:

1. Subsample the training set to 10%, 25%, 50%, 100%.
2. Train with 3 random seeds.
3. Report top-1 accuracy, macro F1.
4. Perform paired t-tests between strategies.

### 14.3.5 The Metrics

**Primary:** Top-1 accuracy.

**Secondary:** Macro F1, confusion matrix.

**Statistical:** Paired t-test.

### 14.3.6 The Contribution

A systematic study of transfer learning regimes: "Feature extraction is sufficient when the dataset is small and the domain is similar; fine-tuning is necessary when the dataset is large or the domain is dissimilar."

### 14.3.7 Summary: Option B

| Aspect | Description |
|--------|-------------|
| Question | How does TL vary with size and similarity? |
| Data | CIFAR, Flowers, Cars, Fashion-MNIST |
| Method | ResNet-50 with 3 TL strategies |
| Experiments | 5 datasets × 4 sizes × 3 strategies |
| Metrics | Top-1 accuracy, F1 |
| Contribution | Transfer learning decision framework |

> **Exercise 14.3:** Adapt Option B to an image dataset you have access to. Write a one-page proposal.

---

## 14.4 Option C: Object Detection or Segmentation Research

### 14.4.1 The Research Question

**Question:** What is the accuracy-efficiency trade-off between different object detection architectures?

Object detection is a core computer vision task. Different architectures (YOLO, Faster R-CNN, DETR) have different accuracy-efficiency trade-offs. A systematic comparison is valuable for practitioners.

### 14.4.2 The Data

Use COCO or PASCAL VOC:

| Dataset | Images | Classes | Annotations |
|---------|--------|---------|-------------|
| COCO | 123,287 | 80 | Bounding boxes, masks |
| PASCAL VOC | 11,540 | 20 | Bounding boxes |

### 14.4.3 The Method

Train the following detection architectures:

**Two-stage:** Faster R-CNN with ResNet-50 FPN.

**One-stage:** YOLOv5, YOLOv8, RetinaNet.

**Transformer-based:** DETR, Deformable DETR.

### 14.4.4 The Experiments

For each architecture:

1. Train on COCO train2017.
2. Evaluate on COCO val2017.
3. Report mAP@[0.5:0.95], mAP@0.5, mAP@0.75.
4. Measure inference time (FPS) on a fixed GPU.
5. Measure model size (parameters, FLOPs).

### 14.4.5 The Metrics

**Primary:** mAP@[0.5:0.95].

**Secondary:** mAP@0.5, mAP@0.75, FPS, parameters, FLOPs.

### 14.4.6 The Contribution

A benchmark study: "YOLOv8 achieves X mAP at Y FPS; DETR achieves Z mAP at W FPS. The trade-off is..."

### 14.4.7 Summary: Option C

| Aspect | Description |
|--------|-------------|
| Question | Accuracy-efficiency trade-off |
| Data | COCO, PASCAL VOC |
| Method | YOLO, Faster R-CNN, DETR |
| Experiments | 5 architectures × 2 datasets |
| Metrics | mAP, FPS, parameters, FLOPs |
| Contribution | Benchmark study |

> **Exercise 14.4:** Adapt Option C to a detection or segmentation task you have access to. Write a one-page proposal.

---

## 14.5 Option D: Domain-Specific Application

### 14.5.1 The Research Question

**Question:** Can deep learning models achieve clinically meaningful performance on a specific medical imaging task?

Medical imaging is a high-impact domain for deep learning. But the performance of deep learning models varies widely across tasks. A careful study on a specific task is valuable.

### 14.5.2 The Data

Use a public medical imaging dataset:

| Dataset | Modality | Task | Images |
|---------|----------|------|--------|
| ChestX-ray14 | X-ray | Multi-label classification | 112,120 |
| ISIC | Dermoscopy | Skin lesion classification | 25,331 |
| Camelyon17 | Histopathology | Metastasis detection | 1,000 |
| BraTS | MRI | Brain tumor segmentation | 1,251 |

### 14.5.3 The Method

Fine-tune a pre-trained model (ResNet-50, EfficientNet, ViT) on the task. Compare with a baseline trained from scratch.

### 14.5.4 The Experiments

For each model:

1. Split into train/val/test (respecting patient-level splits).
2. Train with 3 random seeds.
3. Report accuracy, sensitivity, specificity, AUC.
4. Perform statistical tests.

### 14.5.5 The Metrics

**Primary:** AUC.

**Secondary:** Accuracy, sensitivity, specificity, F1.

**Clinical:** Decision curve analysis.

### 14.5.6 The Contribution

A clinically-oriented study: "Our model achieves AUC X on task Y, which is comparable to human experts."

### 14.5.7 Summary: Option D

| Aspect | Description |
|--------|-------------|
| Question | Clinical performance |
| Data | Public medical imaging |
| Method | Fine-tuning |
| Experiments | Model comparison |
| Metrics | AUC, sensitivity, specificity |
| Contribution | Clinical validation |

> **Exercise 14.5:** Adapt Option D to a domain-specific application you have access to (medical, industrial, agricultural). Write a one-page proposal.

---

## 14.6 Option E: Reproducibility Study

### 14.6.1 The Research Question

**Question:** Can a published paper's results be reproduced, and if not, why?

Reproducibility is a core issue in ML. A reproducibility study is a valid contribution.

### 14.6.2 The Paper

Choose a paper with:

- Publicly available code (or detailed methods).
- Publicly available data.
- Results that can be verified.
- Moderate compute requirements.

Suggested papers:
- "Deep Residual Learning for Image Recognition" (ResNet)
- "Attention Is All You Need" (Transformer)
- "Adam: A Method for Stochastic Optimization" (Adam)
- "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (Dropout)

### 14.6.3 The Method

Reimplement the method from the paper. Train on the same data. Compare results.

### 14.6.4 The Experiments

1. Reimplement the method.
2. Train with the paper's hyperparameters.
3. Train with your own hyperparameters.
4. Compare results with the paper.
5. Investigate discrepancies.

### 14.6.5 The Metrics

**Primary:** The paper's primary metric.

**Secondary:** Training time, compute.

**Statistical:** Confidence intervals, significance tests.

### 14.6.6 The Contribution

A reproducibility report: "We reproduce the main result of paper X with an accuracy of Y (paper reports Z). The discrepancy is due to..."

### 14.6.7 Summary: Option E

| Aspect | Description |
|--------|-------------|
| Question | Can the paper be reproduced? |
| Data | Paper's dataset |
| Method | Reimplementation |
| Experiments | Paper's hyperparameters vs. own |
| Metrics | Paper's primary metric |
| Contribution | Reproducibility report |

> **Exercise 14.6:** Adapt Option E to a paper you have read. Write a one-page proposal.

---

## 14.7 Option F: Custom Project

### 14.7.1 When to Choose a Custom Project

Choose a custom project if:

- You have a specific research question that does not fit any of the templates.
- You have access to a unique dataset.
- You have a novel idea you want to test.
- You want to combine multiple options.

### 14.7.2 How to Design a Custom Project

Follow the same structure as the templates:

**Research question.** Specific, falsifiable, novel, feasible.

**Data.** What data will you use?

**Method.** What approach will you take?

**Experiments.** What experiments will you run?

**Metrics.** How will you evaluate?

**Contribution.** What will you contribute?

### 14.7.3 Common Pitfalls

**Too broad.** "Deep learning for X" is not a research question.

**Too vague.** "Improve Y" is not falsifiable.

**Too ambitious.** "Solve Z" is not feasible.

**No baseline.** Without a baseline, you cannot evaluate.

### 14.7.4 Summary: Option F

| Aspect | Description |
|--------|-------------|
| When to choose | Unique question, data, or idea |
| Structure | Same as templates |
| Pitfalls | Too broad, vague, ambitious |

> **Exercise 14.7:** If none of the templates fit, write a one-page proposal for a custom project.

---

## 14.8 Choosing Your Project

### 14.8.1 The Decision Matrix

Use the following matrix to compare options:

| Option | Interest | Feasibility | Contribution | Data | Compute | Total |
|--------|----------|-------------|--------------|------|---------|-------|
| A: Tabular | | | | | | |
| B: Image | | | | | | |
| C: Detection | | | | | | |
| D: Domain | | | | | | |
| E: Repro | | | | | | |
| F: Custom | | | | | | |

Rate each criterion from 1 (low) to 5 (high). Sum the scores.

### 14.8.2 Sample Projects from Past Students

**Project 1:** "Data Augmentation for Fine-Grained Flower Classification"
- Option B (image classification)
- Dataset: Oxford Flowers-102
- Finding: Random crop and horizontal flip are the most effective augmentations.

**Project 2:** "Neural Networks vs. Gradient Boosting for Welding Process Prediction"
- Option A (tabular)
- Dataset: Welding (480 samples)
- Finding: XGBoost outperforms MLP on this small dataset.

**Project 3:** "Reproducing ResNet on CIFAR-10"
- Option E (reproducibility)
- Paper: He et al. (2016)
- Finding: Reproduced accuracy is within 0.5% of the paper.

**Project 4:** "Fine-Tuning Vision Transformers for Medical Imaging"
- Option D (domain-specific)
- Dataset: ChestX-ray14
- Finding: ViT matches ResNet performance with more data.

### 14.8.3 Summary: Choosing Your Project

| Step | Action |
|------|--------|
| 1 | Rate each option on the matrix |
| 2 | Choose the highest-scoring option |
| 3 | Write a one-page proposal |
| 4 | Meet with your advisor |
| 5 | Refine the proposal |

> **Exercise 14.8:** Fill out the decision matrix. Choose your project. Write a one-page proposal.

---

## 14.9 Writing Your Proposal

### 14.9.1 The Proposal Template

Use the template from Module 13:

**Title.** Descriptive.

**Research question.** One or two sentences.

**Motivation.** Why it matters.

**Literature review.** Prior work.

**Gap.** What is missing.

**Method.** Your approach.

**Data.** What data.

**Experiments.** What experiments.

**Metrics.** How to evaluate.

**Timeline.** 12 weeks.

**Resources.** What you need.

### 14.9.2 Worked Example 14.1: A Proposal for Option B

**Title:** Transfer Learning for Fine-Grained Flower Classification: How Much Does Dataset Size Matter?

**Research question:** How does the performance of transfer learning for fine-grained flower classification vary with the amount of training data?

**Motivation:** Fine-grained classification is challenging because classes are visually similar. Transfer learning is standard, but the relationship between dataset size and performance is not fully understood.

**Literature review:** Kornblith et al. (2019) showed that better ImageNet models transfer better. Zhai et al. (2019) studied large-scale transfer learning. However, the effect of dataset size on fine-grained classification is not well-studied.

**Gap:** The relationship between dataset size and transfer learning performance for fine-grained classification is unclear.

**Method:** Fine-tune a pre-trained ResNet-50 on Oxford Flowers-102 with varying amounts of training data (10%, 25%, 50%, 100%). Compare feature extraction and fine-tuning.

**Data:** Oxford Flowers-102 (102 classes, ~8,000 images).

**Experiments:** For each data size and strategy, train with 3 random seeds. Report top-1 accuracy.

**Metrics:** Top-1 accuracy, macro F1.

**Timeline:** 12 weeks.

**Resources:** 1 GPU, Oxford Flowers-102 dataset.

### 14.9.3 Summary: Writing Your Proposal

| Section | Content |
|---------|---------|
| Title | Descriptive |
| Research question | Specific, falsifiable |
| Motivation | Why it matters |
| Literature review | Prior work |
| Gap | What is missing |
| Method | Your approach |
| Data | What data |
| Experiments | What experiments |
| Metrics | How to evaluate |
| Timeline | 12 weeks |
| Resources | What you need |

> **Exercise 14.9:** Write your project proposal using the template. Bring it to your individual meeting.

---

## 14.10 Module Summary

| Option | Question | Data | Method | Contribution |
|--------|----------|------|--------|--------------|
| A: Tabular | When do NNs beat GBMs? | UCI datasets | MLP vs. GBM | Decision framework |
| B: Image | How does TL vary with size? | CIFAR, Flowers | ResNet-50 | TL framework |
| C: Detection | Accuracy-efficiency trade-off? | COCO, VOC | YOLO, DETR | Benchmark |
| D: Domain | Clinical performance? | Medical imaging | Fine-tuning | Clinical validation |
| E: Repro | Can the paper be reproduced? | Paper's data | Reimplementation | Repro report |
| F: Custom | Your question | Your data | Your method | Your contribution |

---

## 14.11 Capstone Thread: Choosing Your Project

By the end of this module, you will be able to choose a capstone project; adapt a template to your interests; identify the resources you need; write a project proposal; and articulate why your project is feasible within the 12-week timeline.

In Module 15, we will cover **writing and publishing** — how to communicate your results to the research community.

---

## 14.12 Additional Resources

### On Project Ideas

- "Papers With Code" (SOTA tracking): https://paperswithcode.com/
- "ML Reproducibility Challenge": https://paperswithcode.com/rc2022
- "Awesome Lists" (curated project ideas): https://github.com/ChristosChristofidis/awesome-deep-learning

### On Datasets

- UCI ML Repository: https://archive.ics.uci.edu/
- Kaggle Datasets: https://www.kaggle.com/datasets
- Papers With Code Datasets: https://paperswithcode.com/datasets
- Hugging Face Datasets: https://huggingface.co/datasets

### On Specific Domains

- Medical imaging: https://www.kaggle.com/datasets?search=medical
- Satellite imagery: https://www.kaggle.com/datasets?search=satellite
- Agriculture: https://www.kaggle.com/datasets?search=agriculture
- Industrial: https://www.kaggle.com/datasets?search=industrial

### On Reproducibility

- ML Reproducibility Checklist: https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf
- "Reproducibility in Machine Learning" (Stanford): https://cs.stanford.edu/

---

*End of Module 14*