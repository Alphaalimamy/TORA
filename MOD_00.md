# The Open Research Academy (TORA)

## A Research-Oriented Curriculum for Accessible Deep Learning
**The Open Research Academy (TORA)** is dedicated to making world-class research training accessible to everyone, regardless of institutional affiliation, geographic location, or financial background.

**Mission Statement of TORA:**

> *"To democratize research training in deep learning by providing a rigorous, self-paced, openly available curriculum that equips anyone — from a student in Lagos to a researcher in Lima — with the skills to conduct, reproduce, and publish original research using PyTorch."*


# TORA CS 336: PyTorch for Research

## From Foundations to Original Contribution

**Instructor:** Alpha Alimamy Kamara

**Course Philosophy:** This is not a tutorial. This is a research apprenticeship. You will learn to *read* research, *reproduce* research, and *conduct* research using PyTorch as your primary instrument. The end goal is a publishable-quality capstone project.

**Prerequisites:** Linear algebra, probability, basic deep learning (CS229/CS230 equivalent), Python proficiency.

**Accessibility Commitment:** All materials are freely available. No paywalls. No institutional affiliation required. No prior research experience assumed. Only curiosity, discipline, and access to a computer.

**Course Materials:** All notebooks, datasets, and code are available at [github.com/sori-cs336](https://github.com/Alphaalimamy/tora-cs336) under a Creative Commons license.


# Table of Contents

## Front Matter

- **0.0** Course Overview
- **0.1** How to Use This Course (Self-Paced Guide)
- **0.2** Accessibility Statement
- **0.3** Prerequisites and Setup
- **0.4** Learning Outcomes
- **0.5** Assessment and Capstone Overview

## Part I: Foundations for Research Practice

### Module 1: The Research Mindset
- 1.1 What Makes Deep Learning Research Different from Engineering
- 1.2 The Scientific Method in Machine Learning
- 1.3 Reproducibility as a First-Class Concern
- 1.4 How This Course Differs from Tutorials
- 1.5 The Research Lifecycle: A Roadmap
- 1.6 Reading Assignment
- 1.7 Exercises
- 1.8 Looking Ahead
- 1.9 Additional Resources
- 1.10 Module Summary
- 1.11 Capstone Thread: Beginning Your Project

### Module 2: Tensor Computing with Research Intent
- 2.1 Tensors as Numerical Abstractions
- 2.2 Creating Tensors: The Practical Foundations
- 2.3 Tensor Attributes: Shape, Dtype, Device
- 2.4 Randomness and Reproducibility
- 2.5 Mathematical Operations on Tensors
- 2.6 Aggregation Operations
- 2.7 Indexing and Slicing
- 2.8 NumPy Interoperability
- 2.9 Worked Example: Linear Regression Data Pipeline
- 2.10 Common Errors and Debugging
- 2.11 Research Application: Tensor Operations in a Neural Network
- 2.12 Module Summary
- 2.13 Capstone Thread: Tensor Skills for Your Project
- 2.14 Additional Resources

### Module 3: Autograd and Computation Graphs
- 3.1 The Mathematics of Differentiation
- 3.2 The Autograd Engine
- 3.3 The Dynamic Computation Graph
- 3.4 Manual Gradient Descent vs. Autograd
- 3.5 The Optimizer
- 3.6 Loss Functions
- 3.7 From Manual to Modular: The Training Step
- 3.8 Research Application: Gradient Checking
- 3.9 Common Autograd Errors and Debugging
- 3.10 Worked Example: Full Training Pipeline with Autograd
- 3.11 Module Summary
- 3.12 Capstone Thread: Autograd Skills for Your Project
- 3.13 Additional Resources

### Module 4: Building Models from First Principles
- 4.1 The Mathematics of a Linear Model
- 4.2 From Parameters to `nn.Parameter`
- 4.3 The `nn.Module` Base Class
- 4.4 The `nn.Linear` Layer
- 4.5 Sequential Models
- 4.6 Model Initialization
- 4.7 Worked Example: Full Model Pipeline
- 4.8 Model State: `state_dict()` and Checkpointing
- 4.9 Common Errors and Debugging
- 4.10 Research Application: Model Ablation
- 4.11 Module Summary
- 4.12 Capstone Thread: Model Skills for Your Project
- 4.13 Additional Resources

### Module 5: Training Loops and Optimization
- 5.1 The Mathematics of Optimization
- 5.2 The Canonical Training Loop
- 5.3 Loss Functions
- 5.4 Optimizers
- 5.5 Learning Rate Schedules
- 5.6 Evaluation Loops
- 5.7 Checkpointing and Resuming Training
- 5.8 Worked Example: Full Training Pipeline
- 5.9 Common Errors and Debugging
- 5.10 Research Application: Hyperparameter Tuning
- 5.11 Module Summary
- 5.12 Capstone Thread: Training Skills for Your Project
- 5.13 Additional Resources

### Module 6: Data Pipelines for Research
- 6.1 The Mathematics of Data
- 6.2 The `Dataset` Abstraction
- 6.3 The `DataLoader` Abstraction
- 6.4 Tabular Data: Excel and CSV
- 6.5 Image Data
- 6.6 Text Data
- 6.7 Handling Imbalanced Datasets
- 6.8 Worked Example: Full Data Pipeline
- 6.9 Common Errors and Debugging
- 6.10 Research Application: Data Versioning and Reproducibility
- 6.11 Module Summary
- 6.12 Capstone Thread: Data Skills for Your Project
- 6.13 Additional Resources

## Part II: The Research Lifecycle

### Module 7: Finding and Managing Scientific Literature
- 7.1 The Role of Literature in Research
- 7.2 Where to Find Papers
- 7.3 How to Search Effectively
- 7.4 Staying Current
- 7.5 Reference Management
- 7.6 Building a Personal Knowledge Base
- 7.7 Reading Groups and Collaboration
- 7.8 Worked Example: A Complete Literature Review
- 7.9 Common Pitfalls and Best Practices
- 7.10 Research Application: Positioning Your Contribution
- 7.11 Module Summary
- 7.12 Capstone Thread: Literature Skills for Your Project
- 7.13 Additional Resources

### Module 8: How to Read a Scientific Paper
- 8.1 The Three-Pass Method
- 8.2 Critical Appraisal
- 8.3 Questions to Ask While Reading
- 8.4 Common Pitfalls in Reading Papers
- 8.5 Reading Different Types of Papers
- 8.6 Worked Example: Reading a Paper in Detail
- 8.7 Reading for Different Purposes
- 8.8 Writing a Paper Review
- 8.9 Peer Review
- 8.10 Research Application: Reading for Your Capstone
- 8.11 Module Summary
- 8.12 Capstone Thread: Reading Skills for Your Project
- 8.13 Additional Resources

### Module 9: Reproducing a Paper
- 9.1 The Philosophy of Reproduction
- 9.2 The Reproduction Checklist
- 9.3 A Reproduction Workflow
- 9.4 Common Barriers to Reproduction
- 9.5 Case Study: Reproducing a Simple Paper
- 9.6 Case Study: Reproducing a Real Paper
- 9.7 Reporting a Reproduction
- 9.8 Reproducibility in Practice
- 9.9 Research Application: Reproduction as a Contribution
- 9.10 Module Summary
- 9.11 Capstone Thread: Reproduction Skills for Your Project
- 9.12 Additional Resources

## Part III: Applied Research with PyTorch

### Module 10: Tabular Data and Regression Research
- 10.1 The Tabular Data Problem
- 10.2 Loading Excel/CSV Data into PyTorch
- 10.3 Preprocessing Tabular Data
- 10.4 A Custom Dataset for Tabular Data
- 10.5 Case Study: Welding Process Parameter Prediction
- 10.6 Baselines: Gradient Boosting vs. Neural Networks
- 10.7 Evaluation: RMSE, MAE, R², and Statistical Tests
- 10.8 Worked Example: Full Tabular Regression Pipeline
- 10.9 Common Errors and Debugging
- 10.10 Research Application: When Do Neural Networks Beat GBMs?
- 10.11 Module Summary
- 10.12 Capstone Thread: Tabular Data for Your Project
- 10.13 Additional Resources

### Module 11: Image Data and Computer Vision Research
- 11.1 Images as Tensors
- 11.2 Loading Image Data
- 11.3 Transforms and Augmentation
- 11.4 Convolutional Neural Networks (CNNs)
- 11.5 Case Study: Shape Classification from Generated Images
- 11.6 Evaluation: Accuracy, Precision, Recall, F1
- 11.7 Worked Example: Full Image Classification Pipeline
- 11.8 Common Errors and Debugging
- 11.9 Research Application: Data Augmentation as Regularization
- 11.10 Module Summary
- 11.11 Capstone Thread: Image Data for Your Project
- 11.12 Additional Resources

### Module 12: Advanced Architectures and Transfer Learning
- 12.1 CNNs: ResNet, EfficientNet, and Their Trade-offs
- 12.2 Vision Transformers (ViTs)
- 12.3 Transfer Learning Strategies
- 12.4 Fine-Tuning vs. Feature Extraction
- 12.5 Domain Adaptation
- 12.6 Case Study: Fine-Tuning a Pre-trained Model
- 12.7 Worked Example: Full Transfer Learning Pipeline
- 12.8 Common Errors and Debugging
- 12.9 Research Application: How Much Does Domain Similarity Matter?
- 12.10 Module Summary
- 12.11 Capstone Thread: Advanced Architectures for Your Project
- 12.12 Additional Resources

## Part IV: Capstone Research Projects

### Module 13: Capstone Project Guidelines
- 13.1 Project Requirements
- 13.2 Evaluation Criteria
- 13.3 Project Timeline
- 13.4 Project Milestones
- 13.5 Project Proposal Template
- 13.6 Project Report Template
- 13.7 Project Presentation Guidelines
- 13.8 Common Pitfalls
- 13.9 Resources for Capstone Projects
- 13.10 Module Summary

### Module 14: Capstone Project Options
- 14.1 Option A: Tabular Data Research
- 14.2 Option B: Image Classification Research
- 14.3 Option C: Object Detection or Segmentation Research
- 14.4 Option D: Domain-Specific Application
- 14.5 Option E: Reproducibility Study
- 14.6 Option F: Custom Project
- 14.7 Choosing Your Project
- 14.8 Sample Projects from Past Students
- 14.9 Module Summary

### Module 15: Writing and Publishing Your Research
- 15.1 The Anatomy of an ML Paper
- 15.2 Writing Style: Precision, Honesty, and Avoiding Overclaiming
- 15.3 Figures and Tables: Communicating Results Effectively
- 15.4 Where to Submit: Workshop → Conference → Journal
- 15.5 The Review Process: What Reviewers Look For
- 15.6 Responding to Reviews
- 15.7 Open Science: Preprints, Code, and Data
- 15.8 Exercise: Peer Review a Classmate's Draft
- 15.9 Module Summary
- 15.10 Additional Resources

## Part V: Research Resources

### Module 16: Tools and Infrastructure
- 16.1 Experiment Tracking: Weights & Biases, TensorBoard, MLflow
- 16.2 Hyperparameter Tuning: Optuna, Ray Tune
- 16.3 Distributed Training: DataParallel, DistributedDataParallel
- 16.4 Mixed Precision: `torch.cuda.amp`
- 16.5 Profiling: `torch.profiler`
- 16.6 Environment Management: Conda, Docker
- 16.7 Version Control: Git, DVC
- 16.8 Module Summary
- 16.9 Additional Resources

### Module 17: Community and Continuing Education
- 17.1 Conferences and Workshops: NeurIPS, ICML, ICLR, CVPR, ACL
- 17.2 Reading Groups: How to Run One and Why
- 17.3 Open Source Contribution: Finding a Project and Contributing
- 17.4 Twitter/X, Blogs, and Newsletters for Staying Current
- 17.5 Mentorship and Collaboration
- 17.6 TORA Community: Forums, Discord, and Study Groups
- 17.7 Module Summary
- 17.8 Additional Resources

## Appendices

### Appendix A: Mathematics Refresher
- A.1 Linear Algebra
- A.2 Calculus
- A.3 Probability
- A.4 Statistics

### Appendix B: PyTorch Cheat Sheet
- B.1 Tensor Operations
- B.2 Autograd
- B.3 Models
- B.4 Training
- B.5 Data Loading

### Appendix C: Research Paper Templates
- C.1 NeurIPS Format
- C.2 ICML Format
- C.3 ICLR Format
- C.4 CVPR Format
- C.5 ACL Format

### Appendix D: Datasets for Research
- D.1 Tabular Datasets
- D.2 Image Datasets
- D.3 Text Datasets
- D.4 Audio Datasets
- D.5 Multimodal Datasets

### Appendix E: Glossary of Terms
- E.1 Deep Learning Terms
- E.2 Research Terms
- E.3 PyTorch Terms

### Appendix F: Frequently Asked Questions
- F.1 Course Logistics
- F.2 Technical Issues
- F.3 Research Questions
- F.4 Accessibility

### Appendix G: Contributing to SORI
- G.1 How to Contribute
- G.2 Translation Efforts
- G.3 Code of Conduct
- G.4 Acknowledgments


# Course at a Glance

| Part | Module | Topic | Duration |
|------|--------|-------|----------|
| I | 1 | The Research Mindset | 1 week |
| I | 2 | Tensor Computing with Research Intent | 2 weeks |
| I | 3 | Autograd and Computation Graphs | 2 weeks |
| I | 4 | Building Models from First Principles | 2 weeks |
| I | 5 | Training Loops and Optimization | 2 weeks |
| I | 6 | Data Pipelines for Research | 2 weeks |
| II | 7 | Finding and Managing Scientific Literature | 1 week |
| II | 8 | How to Read a Scientific Paper | 1 week |
| II | 9 | Reproducing a Paper | 2 weeks |
| III | 10 | Tabular Data and Regression Research | 2 weeks |
| III | 11 | Image Data and Computer Vision Research | 2 weeks |
| III | 12 | Advanced Architectures and Transfer Learning | 2 weeks |
| IV | 13 | Capstone Project Guidelines | 1 week |
| IV | 14 | Capstone Project Options | 1 week |
| IV | 15 | Writing and Publishing Your Research | 1 week |
| V | 16 | Tools and Infrastructure | 1 week |
| V | 17 | Community and Continuing Education | 1 week |
| **Total** | | | **24 weeks** |


# SORI Accessibility Commitments

| Commitment | Implementation |
|------------|----------------|
| **Free access** | All materials on GitHub under CC BY 4.0 |
| **No prerequisites** | Self-contained; math refresher in Appendix A |
| **Self-paced** | No deadlines; work at your own speed |
| **Multiple formats** | Jupyter notebooks, PDF, HTML, video walkthroughs |
| **Low-bandwidth options** | Text-first; images optional |
| **Translation** | Community translations in progress (Spanish, French, Arabic, Hindi, Swahili) |
| **Community support** | Discord, GitHub Discussions, study groups |
| **Mentorship** | Volunteer mentors from industry and academia |
| **Hardware access** | Guides for free GPU resources (Colab, Kaggle, SageMaker) |
| **Childcare-friendly** | Asynchronous; no live attendance required |


# SORI Community

| Platform | Purpose | Link |
|----------|---------|------|
| GitHub | Code, issues, discussions | github.com/sori-cs336 |
| Discord | Real-time chat, study groups | discord.gg/sori |
| Forum | Long-form Q&A | forum.sori.org |
| Newsletter | Weekly updates | sori.org/newsletter |
| YouTube | Video walkthroughs | youtube.com/@sori |
| Twitter/X | Announcements | @sori_research |


# Acknowledgments

SORI is inspired by and grateful to:

- **Stanford University** for its commitment to open education
- **The PyTorch team** for building an open-source framework
- **The Hugging Face community** for democratizing NLP
- **Papers With Code** for tracking SOTA and providing implementations
- **The ML Reproducibility community** for raising standards
- **Every researcher** who has shared their code, data, and knowledge

---

*"Research is not a privilege. It is a practice. And practice is available to anyone willing to put in the work."*

— SORI Founding Principles

---

**End of Table of Contents**