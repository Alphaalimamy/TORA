# Module 1: The Research Mindset

## TORA | CS 336: PyTorch for Research

**Instructor:** Alpha Alimamy Kamara

**Module Length:** 1 week (2 lectures + 1 discussion section)

**Prerequisites:** None for this module; students should have read the course syllabus and completed the pre-course survey.


## 1.0 Module Overview

Before we write a single line of PyTorch code, we must confront a fundamental question: **What separates a researcher from an engineer?**

An engineer builds systems that work. A researcher builds systems *and* generates knowledge about why they work, when they fail, and what that means for the field. This distinction is not semantic. It shapes every decision you will make in this course, from how you structure your code to how you report your results.

This module establishes the intellectual foundation for everything that follows. We will examine the scientific method as it applies to machine learning, the reproducibility crisis in our field, and the habits of mind that distinguish rigorous research from ad-hoc experimentation.

> This module contains no PyTorch code. That is intentional. The most common failure mode of graduate-level ML courses is rushing to implementation without first establishing the conceptual and methodological scaffolding. We will not make that mistake.


## 1.1 What Makes Deep Learning Research Different from Engineering

### 1.1.1 The Engineering Mindset

In software engineering, success is defined by functionality. A program that correctly sorts a list is correct. A web application that serves requests without crashing is working. The criteria are clear, binary, and verifiable.

Engineering values:
- **Correctness:** Does it do what it is supposed to do?
- **Reliability:** Does it do so consistently?
- **Efficiency:** Does it do so within resource constraints?
- **Maintainability:** Can others understand, modify, and extend it?

These values are not wrong. They are simply incomplete for research.

### 1.1.2 The Research Mindset

Research adds a fifth criterion that subsumes the 
others: **knowledge generation**. 
A research contribution is not merely a working system.
it is a working system that teaches us something we did not know before.

This means that a system which *fails* can still be a successful research contribution, provided the failure is informative. The history of deep learning is full of "negative results" that reshaped the field:

- The original perceptron's inability to learn XOR (Minsky & Papert, 1969) motivated multi-layer networks.
- The observation that deeper networks *performed worse* on training data (He et al., 2016) led to residual connections.
- The finding that attention alone, without recurrence, could match or exceed RNNs (Vaswani et al., 2017) restructured the entire field of NLP.

In each case, the research value came not from building something that worked, but from understanding *why* something worked or failed.

### 1.1.3 The Three Questions of Research

Every research project in this course must answer three
questions. If you cannot answer all three, you do not 
have a research project, you have a hobby.

| Question | What It Means | How to Answer It |
|----------|---------------|------------------|
| **What is the question?** | A specific, falsifiable hypothesis about the world | Literature review; gap analysis |
| **What is the evidence?** | Empirical results that support or refute the hypothesis | Controlled experiments; statistical testing |
| **What does it mean?** | Interpretation of the evidence in the context of prior work | Critical analysis; discussion of limitations |

> **Exercise 1.1 (Pre-Class):** Write a one-paragraph answer to each of the three questions for a topic of your choice. Bring this to lecture. We will critique them together.

### 1.1.4 Why This Matters for PyTorch

PyTorch is a tool. A powerful tool, but a tool nonetheless. The danger of any powerful tool is that it can seduce you into using it without thinking. The `torch.nn` module makes it trivially easy to build a neural network. That ease is a gift and a curse.

The gift: you can prototype ideas in minutes rather than weeks.
The curse: you can prototype *bad* ideas in minutes rather than weeks, and never notice.

Throughout this course, we will constantly return to the three questions. Every time you write `optimizer.step()`, you should be able to explain why. Every time you choose `nn.MSELoss()` over `nn.L1Loss()`, you should be able to justify it. Every time you report a number, you should be able to defend it.


## 1.2 The Scientific Method in Machine Learning

### 1.2.1 Classical Scientific Method

The classical scientific method, as formalized by Bacon, Popper, and others, proceeds through a cycle:

1. **Observation:** Notice a phenomenon.
2. **Hypothesis:** Propose an explanation.
3. **Prediction:** Derive testable consequences of the hypothesis.
4. **Experiment:** Test the predictions.
5. **Analysis:** Interpret the results.
6. **Iteration:** Refine the hypothesis and repeat.

The critical insight due to Popper is **falsifiability**. A hypothesis that cannot, in principle, be falsified is not scientific. "All swans are white" is scientific because a single black swan refutes it. "The universe is governed by invisible, undetectable forces" is not, because no observation could refute it.

### 1.2.2 The ML Research Cycle

Machine learning research follows the same cycle, but with domain-specific instantiations:

| Classical Step | ML Instantiation |
|----------------|------------------|
| Observation | A model underperforms on a task; a new architecture is proposed; a dataset reveals an anomaly |
| Hypothesis | "Adding attention to the decoder will improve translation quality" |
| Prediction | "The BLEU score will increase by at least 2 points on WMT-14" |
| Experiment | Train baseline and attention-augmented models on identical data |
| Analysis | Compare BLEU scores with statistical significance testing |
| Iteration | Refine the architecture, test on additional languages, ablate components |

The ML instantiation is messier than the classical version because:
- **Stochasticity:** Random initialization, data ordering, and dropout introduce variance.
- **Confounding:** Many factors change simultaneously (architecture, hyperparameters, data).
- **Cost:** Each experiment is computationally expensive, limiting the number of trials.
- **Benchmarking culture:** The field often conflates benchmark performance with scientific understanding.

### 1.2.3 Falsifiability in ML

What does it mean for an ML hypothesis to be falsifiable?

Consider two hypotheses:

- **H1:** "Transformer models are better than RNNs for machine translation."
- **H2:** "The self-attention mechanism is the key component that makes Transformers effective."

H1 is testable: train both, compare BLEU scores, report the difference with confidence intervals.

H2 is *harder* to test but still falsifiable: ablate self-attention (replace it with something else), retrain, and see if performance degrades. If it doesn't, H2 is refuted.

Now consider:

- **H3:** "Transformers work because they capture long-range dependencies better than RNNs."

H3 is *not* directly falsifiable without a precise operationalization of "capture long-range dependencies." You could measure attention distances, probe representations, or test on synthetic long-range tasks. But the hypothesis as stated is too vague to be tested rigorously. It is a *research direction*, not a *hypothesis*.

**The lesson:** Precision in hypothesis formulation is not pedantry. It is the difference between a research project and a wish.

> **Exercise 1.2 :** Convert the following vague hypotheses into falsifiable ones:
> 1. "Data augmentation helps with small datasets."
> 2. "Larger models are better."
> 3. "Batch normalization makes training more stable."
>
> For each, specify: the independent variable, the dependent variable, the control condition, and the statistical test you would use.



## 1.3 Reproducibility as a First-Class Concern

### 1.3.1 The Reproducibility Crisis

In 2016, *Nature* published a survey finding that **more than 70% of researchers** had tried and failed to reproduce another scientist's experiments, and more than half had failed to reproduce their *own* experiments. This is not a problem unique to machine learning, but ML has its own flavor of it.

In 2019, Joelle Pineau (Meta AI, McGill) introduced the **ML Reproducibility Checklist**, now adopted by NeurIPS, ICML, ICLR, and other major venues. The checklist requires authors to report:

- Hyperparameter search spaces
- Random seeds and number of runs
- Compute infrastructure (GPU type, training time)
- Data preprocessing steps
- Evaluation metrics and statistical tests
- Code and data availability

The fact that such a checklist is necessary tells you something about the state of the field.

### 1.3.2 Levels of Reproducibility

Not all reproducibility is created equal. The following taxonomy, adapted from the ACM and ML community, is useful:

| Level | Definition | Example |
|-------|------------|---------|
| **Repeatability** | Same team, same code, same data → same result | You rerun your own experiment and get the same number |
| **Replicability** | Different team, same code, same data → same result | A colleague clones your repo and gets the same number |
| **Reproducibility** | Different team, different code, same data → same conclusion | A colleague reimplements your method from the paper and gets a similar number |
| **Generalizability** | Different team, different code, different data → same conclusion | Your method works on a new dataset from the same distribution |

In this course, we aim for **reproducibility** as the minimum standard and **generalizability** as the aspiration.

### 1.3.3 Sources of Non-Reproducibility in PyTorch

PyTorch, by default, is **not** deterministic. This is not a bug — it is a deliberate design choice for performance. But it means that unless you take explicit steps, your experiments will not be reproducible.

The sources of non-determinism include:

1. **Random seed initialization:** `torch.manual_seed()`, `torch.cuda.manual_seed()`, and `numpy.random.seed()` must all be set.
2. **CUDA operations:** Some CUDA kernels are non-deterministic by default (e.g., `atomicAdd` in certain reduction operations).
3. **Data loading order:** `DataLoader` with `shuffle=True` uses a random permutation; the seed for this must be controlled.
4. **Dropout and other stochastic layers:** These use the global random state.
5. **Floating-point non-associativity:** `(a + b) + c` may not equal `a + (b + c)` in floating point, and the order of operations may vary across runs.
6. **Multi-GPU training:** Gradient reduction order is non-deterministic.

We will address each of these in Module 9 when we discuss reproducibility in depth. For now, the key takeaway is: **reproducibility is not automatic; it is engineered.**

### 1.3.4 The Reproducibility Checklist for This Course

Every experiment you run in this course must satisfy the following:

- All random seeds are set at the beginning of the script
- The exact command to reproduce the experiment is documented
- Hyperparameters are logged (not hardcoded and forgotten)
- The dataset version and preprocessing steps are recorded
- The random seed is reported in the results
- Results are reported as mean ± standard deviation over at least 3 runs
- Code is committed to version control with a tag for the final experiment

> **Exercise 1.3 (Lab):** Take a simple PyTorch script (we will provide one) that trains a linear regression model. Run it three times without setting any seeds. Record the final loss each time. Then set all seeds and run it three more times. Report the variance in each case. What does this tell you about the importance of seed control?



## 1.4 How This Course Differs from Tutorials

### 1.4.1 The Tutorial Trap

There is no shortage of PyTorch tutorials. The official PyTorch tutorials, the Hugging Face course, the fast.ai course, and countless blog posts all teach you how to build models. So why this course?

Tutorials are optimized for **accessibility**. They want you to succeed quickly, to feel the satisfaction of a working model, to move on to the next tutorial. This is a noble goal, and tutorials are excellent for what they do.

But tutorials have a structural limitation: **they teach you what to do, not why or when.**

A tutorial will show you:

```python
model = nn.Sequential(nn.Linear(1, 1)).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

It will not tell you:

- Why `MSELoss` and not `L1Loss`? Under what conditions would `L1Loss` be preferable?
- Why `SGD` and not `Adam`? What are the trade-offs in convergence speed, generalization, and memory?
- Why `lr=0.1`? How would you determine this systematically rather than by trial and error?
- Why `nn.Sequential`? When would you need a custom `nn.Module`?

These are research questions. They are what separate a practitioner from a researcher.

### 1.4.2 The Research Approach

This course takes a different approach. For every construct we introduce, we ask:

1. **What problem does this solve?** (Motivation)
2. **How does it work?** (Mechanism)
3. **When should you use it?** (Decision criteria)
4. **When should you *not* use it?** (Limitations)
5. **How would you test whether it's working?** (Evaluation)
6. **What does the literature say?** (Context)

We will not always have complete answers. Some of these questions are open research problems. That is the point.

### 1.4.3 A Concrete Example: Loss Functions

Consider the choice between `MSELoss` and `L1Loss` for a regression problem.

**Tutorial answer:** "MSELoss is standard for regression. Use it."

**Research answer:**

- MSE penalizes large errors quadratically, making it sensitive to outliers. L1 penalizes linearly, making it robust to outliers.
- MSE is differentiable everywhere; L1 is not differentiable at 0, requiring subgradient methods.
- MSE corresponds to a Gaussian likelihood assumption; L1 corresponds to a Laplace likelihood assumption.
- If your data has outliers (e.g., sensor failures), L1 may be preferable. If your data is well-behaved, MSE may converge faster.
- The choice can be tested empirically: train with both, compare validation loss and test metrics.

This is the level of depth we aim for.

> **Exercise 1.4 (Discussion Section):** Pick a PyTorch construct you have used before (e.g., `nn.Dropout`, `nn.BatchNorm`, `Adam`). Write a one-page analysis addressing the six questions above. Bring it to discussion section. We will critique them in small groups.

---

## 1.5 The Research Lifecycle: A Roadmap

This course is structured around the research lifecycle. The following diagram shows the phases we will cover:

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE RESEARCH LIFECYCLE                       │
└─────────────────────────────────────────────────────────────────┘

   ┌──────────────┐
   │  LITERATURE  │  Module 7: Finding and Managing Papers
   │    REVIEW    │  Module 8: How to Read a Scientific Paper
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  HYPOTHESIS  │  Module 1: The Research Mindset
   │ FORMULATION  │  (You are here)
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  EXPERIMENT  │  Modules 2–6: PyTorch Fundamentals
   │    DESIGN    │  Modules 10–12: Applied Research
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │    DATA      │  Module 6: Data Pipelines
   │  PIPELINE    │  Module 10: Tabular Data
   │              │  Module 11: Image Data
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │    MODEL     │  Module 4: Building Models
   │  DEVELOPMENT │  Module 5: Training Loops
   │              │  Module 12: Advanced Architectures
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  EVALUATION  │  Module 9: Evaluation and Validation
   │              │  (statistical testing, ablations)
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  ANALYSIS &  │  Module 15: Writing and Publishing
   │   WRITING    │  (including negative results)
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │   CAPSTONE   │  Modules 13–14: Capstone Projects
   │   PROJECT    │  (original contribution)
   └──────────────┘
```

### 1.5.1 Where You Are Now

You are at the beginning: **hypothesis formulation**. Before you can design experiments, you need a question worth asking. Before you can build models, you need to know what you are trying to learn.

This is the hardest part of research. It is also the most important. A well-posed question is half the battle.

### 1.5.2 The Capstone Thread

Throughout this course, you will work toward a capstone project (Modules 13–14). The project is not a final exam. It is the *point* of the course. Everything we do in Modules 2–12 is in service of equipping you to conduct original research.

To that end, you should begin thinking about your capstone topic *now*. By the end of Module 3, you should have a rough idea. By the end of Module 6, you should have a specific question and a dataset. By the end of Module 9, you should have preliminary results.

We will revisit this thread at the end of each module.


## 1.6 Reading Assignment

For the next lecture, read the following:

**Required:**

1. Pineau, J., et al. (2021). "Improving Reproducibility in Machine Learning Research." *Journal of Machine Learning Research*, 22(1), 1–20. [Link](https://jmlr.org/papers/v22/20-303.html)

2. Sculley, D., et al. (2018). "Winner's Curse? On Pace, Progress, and Empirical Rigor." *ICLR Workshop*. [Link](https://openreview.net/forum?id=rJWF0Fywf)

3. Bouthillier, X., et al. (2021). "Accounting for Variance in Machine Learning Benchmarks." *MLSys*. [Link](https://proceedings.mlsys.org/paper/2021/hash/01882513d5fa7c329e940dda99b12147-Abstract.html)

**Recommended:**

4. Henderson, P., et al. (2018). "Deep Reinforcement Learning That Matters." *AAAI*. [Link](https://ojs.aaai.org/index.php/AAAI/article/view/11694)

5. Recht, B., et al. (2019). "Do ImageNet Classifiers Generalize to ImageNet?" *ICML*. [Link](http://proceedings.mlr.press/v97/recht19a.html)

**Reading guide:** As you read, focus on the following questions:

- What is the central claim of the paper?
- What evidence supports it?
- What are the limitations?
- How does this change how you would conduct your own research?

We will discuss these in the next lecture.



## 1.7 Exercises

### Exercise 1.1: The Three Questions

Write a one-paragraph answer to each of the three questions (What is the question? What is the evidence? What does it mean?) for a topic of your choice. This can be anything, a paper you have read, a problem you have encountered, a hypothesis you have about a model's behavior.

**Deliverable:** One page, typed, brought to lecture.

### Exercise 1.2 (In-Class): Falsifiable Hypotheses

Convert the following vague hypotheses into falsifiable ones. For each, specify:

- The independent variable
- The dependent variable
- The control condition
- The statistical test you would use

1. "Data augmentation helps with small datasets."
2. "Larger models are better."
3. "Batch normalization makes training more stable."

**Deliverable:** In-class worksheet, submitted at the end of lecture.

### Exercise 1.3 (Lab): Reproducibility Experiment

We will provide a simple PyTorch script that trains a linear regression model. Your task:

1. Run the script three times without setting any seeds. Record the final loss each time.
2. Set all seeds (`torch.manual_seed`, `torch.cuda.manual_seed`, `numpy.random.seed`, `random.seed`). Run the script three more times. Record the final loss each time.
3. Compute the variance in each case.
4. Write a short reflection (1 paragraph): What does this tell you about the importance of seed control? What are the implications for research?

**Deliverable:** A Jupyter notebook with your code, results, and reflection. Submit to the course GitHub repository.

### Exercise 1.4 (Discussion Section): Construct Analysis

Pick a PyTorch construct you have used before (e.g., `nn.Dropout`, `nn.BatchNorm`, `Adam`). Write a one-page analysis addressing the six questions:

1. What problem does this solve?
2. How does it work?
3. When should you use it?
4. When should you *not* use it?
5. How would you test whether it's working?
6. What does the literature say?

**Deliverable:** One page, typed, brought to discussion section. Be prepared to present your analysis to your small group.



## 1.8 Looking Ahead

In Module 2, we will begin our technical work with **tensor computing**. But we will not leave the research mindset behind. Every tensor operation we discuss will be framed in terms of its role in research: why it matters, when to use it, and how to test it.

You should now have a clear understanding of:

- The difference between engineering and research
- The scientific method as applied to ML
- The importance of reproducibility
- How this course differs from tutorials
- The research lifecycle and your place in it

If any of these points are unclear, now is the time to ask questions. The rest of the course builds on this foundation.



## 1.9 Additional Resources

### On the Scientific Method

- Popper, K. (1959). *The Logic of Scientific Discovery*. Routledge.
- Kuhn, T. (1962). *The Structure of Scientific Revolutions*. University of Chicago Press.
- Chamberlin, T. C. (1890). "The Method of Multiple Working Hypotheses." *Science*, 15(366), 92–96.

### On Reproducibility in ML

- Pineau, J., et al. (2021). "Improving Reproducibility in Machine Learning Research." *JMLR*.
- The ML Reproducibility Checklist: [Link](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)
- Trending Papers: [Link](https://paperswithcode.com/) for tracking SOTA and finding implementations.

### On Research Methodology

- Dodge, J., et al. (2019). "Show Your Work: Improved Reporting of Experimental Results." *EMNLP*.
- Bouthillier, X., et al. (2021). "Accounting for Variance in Machine Learning Benchmarks." *MLSys*.
- Musgrave, K., et al. (2020). "A Metric Learning Reality Check." *ECCV*.

### On Reading Papers

- Keshav, S. (2007). "How to Read a Paper." *ACM SIGCOMM Computer Communication Review*, 37(3), 83–84. [Link](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf)
- Rodriguez, J. (2023). "How to Read a Machine Learning Paper." *Towards Data Science*.

### On Writing Papers

- Mensh, B., & Kording, K. (2017). "Ten Simple Rules for Structuring Papers." *PLOS Computational Biology*.
- Lipton, Z. C., & Steinhardt, J. (2018). "Troubling Trends in Machine Learning Scholarship." *ICML Debates*.

---

## 1.10 Module Summary

| Concept | Key Takeaway |
|---------|--------------|
| Engineering vs. Research | Research adds knowledge generation to functionality |
| Three Questions | What is the question? What is the evidence? What does it mean? |
| Falsifiability | A hypothesis must be testable; vague claims are not hypotheses |
| Reproducibility | Not automatic; must be engineered through seed control, logging, and reporting |
| Levels of Reproducibility | Repeatability → Replicability → Reproducibility → Generalizability |
| Tutorial vs. Research | Tutorials teach *what*; research teaches *why*, *when*, and *how to test* |
| Research Lifecycle | Literature → Hypothesis → Experiment → Data → Model → Evaluation → Writing → Capstone |

---

## 1.11 Capstone Thread: Beginning Your Project

By the end of this module, you should have:

- [ ] Read the required papers
- [ ] Completed the four exercises
- [ ] Started thinking about your capstone topic
- [ ] Identified at least one paper you might want to reproduce or extend

In Module 2, we will begin the technical work. But keep the research mindset active. Every line of code you write from now on should be written with the three questions in mind.

---

*End of Module 1*