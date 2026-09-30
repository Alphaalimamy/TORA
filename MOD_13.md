# Module 13: Capstone Project Guidelines

## TORA CS 336: PyTorch for Research

**Institution:** The Open Research Academy (TORA)

**Instructor:** Alpha Alimamy Kamara

**Module Length:** 2 weeks (2 lectures + 2 lab sessions + 2 individual meetings)

**Prerequisites:** Modules 1–12 completed; a research question in mind; a dataset identified; a working PyTorch environment with GPU access (or a plan for cloud compute).

---

## 13.0 Module Overview

For twelve modules, you have been building the skills of a researcher: reading papers, reproducing results, implementing models, and evaluating them rigorously. Now you will apply those skills to a project of your own design.

The **capstone project** is the culmination of this course. It is not a final exam. It is not a homework assignment. It is an **original research contribution** — a question you pose, an experiment you design, a result you interpret, and a report you write. The goal is not to produce a publishable paper (though some of you will), but to experience the full research lifecycle from start to finish.

This module provides the **guidelines** for the capstone project: what is expected, how it will be evaluated, and how to manage your time. Module 14 provides **project options** — concrete examples of projects you can undertake. Module 15 covers **writing and publishing** — how to communicate your results.

The pedagogical approach here is **structure first, execution second**. A successful capstone project is not the one with the most ambitious idea; it is the one that is well-scoped, well-executed, and well-documented. This module gives you the structure to achieve that.

By the end of this module, you will be able to define a research question; scope a project to fit the available time and resources; plan the project timeline; identify the resources you need; write a project proposal; anticipate common pitfalls; and articulate what makes a successful capstone project.

The module is self-paced, but the individual meetings are scheduled. Come prepared.

---

## 13.1 What Is a Capstone Project?

### 13.1.1 Definition

A **capstone project** is a substantial, independent research project that demonstrates mastery of the skills developed throughout the course. It is the final requirement for completing TORA CS 336.

The capstone project has four components:

**A research question.** A specific, falsifiable question that you will investigate.

**A methodology.** A plan for answering the question: data, models, experiments, metrics.

**A result.** An empirical finding that addresses the question.

**A report.** A written document that communicates the question, methodology, result, and interpretation.

### 13.1.2 What a Capstone Is Not

A capstone is **not** a literature review. It requires original experiments.

A capstone is **not** a tutorial. It requires a novel question, not a re-implementation of an existing method.

A capstone is **not** a software project. The code is a means, not an end. The contribution is the knowledge, not the software.

A capstone is **not** a race to state-of-the-art. It is a careful investigation of a well-posed question, even if the results are modest.

### 13.1.3 The Three Questions Revisited

Recall from Module 1 the three questions of research. For your capstone, you must answer all three:

**What is the question?** A specific, falsifiable hypothesis about the world.

**What is the evidence?** Empirical results from controlled experiments.

**What does it mean?** Interpretation of the evidence in the context of prior work.

If you cannot answer all three, you do not yet have a capstone project.

### 13.1.4 Summary: What Is a Capstone?

| Aspect | Description |
|--------|-------------|
| Research question | Specific, falsifiable |
| Methodology | Data, models, experiments, metrics |
| Result | Empirical finding |
| Report | Written communication |
| Not a literature review | Requires original experiments |
| Not a tutorial | Requires a novel question |
| Not a software project | Code is a means |
| Not a race to SOTA | Careful investigation |

> **Exercise 13.1:** Write a one-paragraph answer to each of the three questions for your proposed capstone project. Bring this to your first individual meeting.

---

## 13.2 Project Requirements

### 13.2.1 The Minimum Requirements

Every capstone project must satisfy the following requirements:

**A clear research question.** Stated in one or two sentences.

**A literature review.** At least five papers reviewed, with a synthesis of the gap your project addresses.

**A reproducible codebase.** A GitHub repository with all code, data (or links to data), and instructions for reproduction.

**Controlled experiments.** Baselines, ablations, and statistical tests.

**A written report.** 8–12 pages in NeurIPS format.

**An oral presentation.** 15 minutes plus 5 minutes of Q&A.

### 13.2.2 The Research Question

A good research question is:

**Specific.** "Does data augmentation improve accuracy on CIFAR-10?" is specific. "Does data augmentation help?" is not.

**Falsifiable.** There must be an experiment that could, in principle, refute the hypothesis.

**Novel.** The question should not have been definitively answered by prior work. (It is fine to reproduce or extend prior work, but the question should be new.)

**Feasible.** The question can be answered within the time and resources available.

**Relevant.** The question matters to the field, even if only in a small way.

### 13.2.3 The Literature Review

The literature review should:

**Summarize** the most relevant prior work (at least five papers).

**Group** the papers by theme.

**Identify** the gap that your project addresses.

**Position** your contribution relative to prior work.

The literature review should be 1–2 pages and should be integrated into the introduction of your report.

### 13.2.4 The Codebase

The codebase should:

**Be reproducible.** All seeds set, all hyperparameters logged, all experiments scripted.

**Be modular.** Separate data loading, model definition, training, and evaluation.

**Be documented.** A README with instructions for reproduction.

**Be versioned.** Git with meaningful commit messages.

**Be licensed.** A permissive license (MIT, Apache, CC BY) for open science.

### 13.2.5 The Experiments

The experiments should include:

**Baselines.** At least one baseline (e.g., linear regression, XGBoost, ResNet-18).

**Ablations.** At least one ablation (e.g., remove a component, vary a hyperparameter).

**Statistical tests.** At least one statistical test (e.g., paired t-test, Wilcoxon).

**Multiple runs.** At least three runs per configuration, with mean ± std reported.

### 13.2.6 The Report

The report should follow the NeurIPS format:

**Abstract.** 150–250 words summarizing the question, method, results, and conclusion.

**Introduction.** Motivation, problem statement, contributions.

**Related Work.** Literature review and gap.

**Method.** Your approach, with mathematical details.

**Experiments.** Datasets, baselines, metrics, implementation details.

**Results.** Main results, ablations, statistical tests.

**Discussion.** Interpretation, limitations, future work.

**Conclusion.** Summary of contributions.

**References.** All cited papers.

**Appendix.** Additional details (hyperparameters, additional results).

### 13.2.7 The Presentation

The presentation should:

**Be 15 minutes.** No more, no less.

**Have 10–15 slides.** One idea per slide.

**Tell a story.** Motivation → question → method → results → conclusion.

**Be practiced.** Rehearse at least twice.

**Include a demo (optional).** If your project has a visual component, show it.

### 13.2.8 Summary: Project Requirements

| Component | Requirement |
|-----------|-------------|
| Research question | Specific, falsifiable, novel, feasible |
| Literature review | At least 5 papers |
| Codebase | Reproducible, modular, documented |
| Experiments | Baselines, ablations, statistical tests |
| Report | 8–12 pages, NeurIPS format |
| Presentation | 15 minutes, 10–15 slides |

> **Exercise 13.2:** Write a one-page project proposal that addresses each of the requirements above. Bring it to your first individual meeting.

---

## 13.3 Evaluation Criteria

### 13.3.1 The Rubric

The capstone project will be evaluated on the following criteria:

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Research question | 15% | Is the question specific, falsifiable, novel, feasible? |
| Literature review | 10% | Is the prior work summarized, grouped, and synthesized? |
| Methodology | 20% | Is the method appropriate, rigorous, and well-justified? |
| Experiments | 20% | Are the experiments controlled, with baselines and ablations? |
| Results | 15% | Are the results statistically significant and well-analyzed? |
| Report | 10% | Is the report clear, well-organized, and well-written? |
| Presentation | 5% | Is the presentation clear, engaging, and well-paced? |
| Reproducibility | 5% | Is the codebase reproducible and well-documented? |

### 13.3.2 What Distinguishes an A from a B

**A project** has a novel question, a rigorous methodology, controlled experiments with statistical tests, a clear report, and a reproducible codebase. It makes a small but real contribution to the field.

**A B project** has a reasonable question, a sound methodology, some experiments, a readable report, and a mostly reproducible codebase. It demonstrates competence but not originality.

**A C project** has a vague question, a questionable methodology, minimal experiments, a poorly written report, or an irreproducible codebase. It demonstrates effort but not mastery.

### 13.3.3 Common Reasons for Low Grades

**Vague question.** "Does deep learning work for X?" is not a research question.

**No baselines.** Without baselines, you cannot know whether your method is an improvement.

**No ablations.** Without ablations, you cannot know which components matter.

**No statistical tests.** A single number is not evidence.

**Irreproducible code.** If we cannot reproduce your results, we cannot evaluate them.

**Overclaiming.** Conclusions that go beyond the evidence.

**Poor writing.** A good project with a bad report is a bad project.

### 13.3.4 Summary: Evaluation Criteria

| Criterion | Weight | Key Question |
|-----------|--------|--------------|
| Research question | 15% | Is it specific, falsifiable, novel? |
| Literature review | 10% | Is prior work synthesized? |
| Methodology | 20% | Is the method rigorous? |
| Experiments | 20% | Are there baselines and ablations? |
| Results | 15% | Are results significant? |
| Report | 10% | Is the report clear? |
| Presentation | 5% | Is the presentation engaging? |
| Reproducibility | 5% | Is the codebase reproducible? |

> **Exercise 13.3:** Evaluate your proposed project against the rubric. Identify the strongest and weakest criteria. Plan how to strengthen the weak ones.

---

## 13.4 Project Timeline

### 13.4.1 The 12-Week Timeline

The capstone project spans 12 weeks. The following timeline is a guide; adjust as needed.

**Week 1: Ideation**
- Choose a research question.
- Conduct a preliminary literature search.
- Identify a dataset.
- Write a one-page proposal.

**Week 2: Literature Review**
- Read at least 10 papers.
- Write a 2-page literature review.
- Refine the research question.
- Identify the gap.

**Week 3: Data Preparation**
- Download and preprocess the data.
- Split into train/val/test.
- Write the data pipeline.
- Verify reproducibility.

**Week 4: Baseline Implementation**
- Implement the simplest baseline.
- Train and evaluate.
- Establish a baseline performance.
- Document the baseline.

**Week 5: Method Implementation**
- Implement your method.
- Train and evaluate.
- Compare with the baseline.
- Debug as needed.

**Week 6: Ablations**
- Design ablation experiments.
- Run ablations.
- Analyze results.
- Identify which components matter.

**Week 7: Statistical Analysis**
- Run multiple seeds.
- Compute mean ± std.
- Perform statistical tests.
- Visualize results.

**Week 8: Additional Experiments**
- Explore edge cases.
- Test on additional datasets.
- Investigate failure modes.
- Refine the method.

**Week 9: Writing the Report**
- Write the introduction.
- Write the related work.
- Write the method.
- Write the experiments.

**Week 10: Refining the Report**
- Write the results.
- Write the discussion.
- Write the conclusion.
- Revise and polish.

**Week 11: Presentation Preparation**
- Create slides.
- Rehearse.
- Get feedback.
- Revise.

**Week 12: Final Submission**
- Submit the report.
- Submit the codebase.
- Present.
- Celebrate.

### 13.4.2 Milestones

The following milestones must be met:

| Milestone | Due | Deliverable |
|-----------|-----|-------------|
| Proposal | Week 1 | 1-page proposal |
| Literature review | Week 2 | 2-page review |
| Data pipeline | Week 3 | Working code |
| Baseline | Week 4 | Baseline results |
| Method | Week 5 | Method results |
| Ablations | Week 6 | Ablation results |
| Draft report | Week 9 | Full draft |
| Final report | Week 12 | Submitted report |
| Presentation | Week 12 | 15-minute talk |

### 13.4.3 Managing Your Time

**Start early.** The most common mistake is starting too late. The capstone is a 12-week project; it cannot be done in the last two weeks.

**Work steadily.** Aim for 10–15 hours per week. Consistent effort beats cramming.

**Set weekly goals.** At the beginning of each week, write down what you will accomplish.

**Meet with your advisor.** Come prepared with specific questions.

**Document as you go.** Write your report incrementally. Do not leave it all to the end.

**Ask for help.** If you are stuck, ask. Do not suffer in silence.

### 13.4.4 Summary: Project Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Ideation | Proposal |
| 2 | Literature review | Review |
| 3 | Data preparation | Pipeline |
| 4 | Baseline | Baseline results |
| 5 | Method | Method results |
| 6 | Ablations | Ablation results |
| 7 | Statistical analysis | Statistics |
| 8 | Additional experiments | Extended results |
| 9 | Writing | Draft report |
| 10 | Refining | Revised report |
| 11 | Presentation prep | Slides |
| 12 | Final submission | Report, code, talk |

> **Exercise 13.4:** Create a detailed timeline for your capstone project. Identify the milestones and the deliverables. Share it with your advisor.

---

## 13.5 Project Proposal

### 13.5.1 The Proposal Template

A project proposal is a 1–2 page document that describes your project. It should include:

**Title.** A descriptive title.

**Research question.** One or two sentences.

**Motivation.** Why does this question matter?

**Literature review.** A brief summary of prior work.

**Gap.** What is missing from prior work?

**Method.** Your approach.

**Data.** What data will you use?

**Experiments.** What experiments will you run?

**Metrics.** How will you evaluate?

**Timeline.** A brief timeline.

**Resources.** What resources do you need?

### 13.5.2 Worked Example 13.1: A Project Proposal

**Title:** Data Augmentation for Fine-Grained Flower Classification: An Ablation Study

**Research question:** Which data augmentation strategies are most effective for fine-grained flower classification, and how do they interact?

**Motivation:** Fine-grained classification is challenging because classes are visually similar. Data augmentation is a standard technique, but the relative effectiveness of different augmentations is unclear.

**Literature review:** Shorten & Khoshgoftaar (2019) survey image data augmentation. Zhang et al. (2018) introduce mixup. DeVries & Taylor (2017) introduce cutout. However, no study systematically compares augmentations for fine-grained classification.

**Gap:** There is no comprehensive ablation of augmentation strategies for fine-grained flower classification.

**Method:** We fine-tune a pre-trained ResNet-50 on the Oxford Flowers-102 dataset. We compare six augmentation strategies: none, horizontal flip, random crop, color jitter, cutout, and mixup. We also test combinations.

**Data:** Oxford Flowers-102 (102 classes, ~8,000 images).

**Experiments:** For each augmentation strategy, train the model with three random seeds. Report mean ± std accuracy on the test set. Perform paired t-tests between strategies.

**Metrics:** Top-1 accuracy, macro F1, confusion matrix.

**Timeline:** 12 weeks, following the standard timeline.

**Resources:** 1 GPU (or Colab Pro), Oxford Flowers-102 dataset (publicly available), PyTorch and torchvision.

### 13.5.3 Common Pitfalls in Proposals

**Too broad.** "Deep learning for medical imaging" is not a research question.

**Too vague.** "Does augmentation help?" is not falsifiable.

**Too ambitious.** "Solve AGI" is not feasible in 12 weeks.

**No gap.** If the question has been answered, it is not research.

**No baseline.** If you cannot compare, you cannot evaluate.

### 13.5.4 Summary: Project Proposal

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

> **Exercise 13.5:** Write a project proposal using the template above. Bring it to your first individual meeting.

---

## 13.6 Project Milestones

### 13.6.1 Milestone 1: Proposal (Week 1)

**Deliverable:** 1-page proposal.

**Evaluation:** Is the question specific, falsifiable, and feasible? Is the gap clearly stated?

**Feedback:** Your advisor will provide written feedback within one week.

### 13.6.2 Milestone 2: Literature Review (Week 2)

**Deliverable:** 2-page literature review.

**Evaluation:** Are at least 5 papers reviewed? Is the gap clearly identified?

**Feedback:** Your advisor will provide written feedback within one week.

### 13.6.3 Milestone 3: Data Pipeline (Week 3)

**Deliverable:** Working code for data loading and preprocessing.

**Evaluation:** Is the pipeline reproducible? Are the splits correct? Is the preprocessing appropriate?

**Feedback:** Your advisor will review the code and provide feedback.

### 13.6.4 Milestone 4: Baseline (Week 4)

**Deliverable:** Baseline results.

**Evaluation:** Is the baseline appropriate? Are the results reasonable?

**Feedback:** Your advisor will review the results and provide feedback.

### 13.6.5 Milestone 5: Method (Week 5)

**Deliverable:** Method results.

**Evaluation:** Does the method improve over the baseline? Is the improvement significant?

**Feedback:** Your advisor will review the results and provide feedback.

### 13.6.6 Milestone 6: Ablations (Week 6)

**Deliverable:** Ablation results.

**Evaluation:** Are the ablations well-designed? Do they isolate the effect of each component?

**Feedback:** Your advisor will review the results and provide feedback.

### 13.6.7 Milestone 7: Draft Report (Week 9)

**Deliverable:** Full draft of the report.

**Evaluation:** Is the report complete? Is it well-written? Are the results clearly presented?

**Feedback:** Your advisor and a peer reviewer will provide feedback.

### 13.6.8 Milestone 8: Final Report (Week 12)

**Deliverable:** Final report, codebase, and presentation.

**Evaluation:** Against the rubric.

### 13.6.9 Summary: Milestones

| Milestone | Due | Deliverable | Feedback |
|-----------|-----|-------------|----------|
| Proposal | Week 1 | 1-page proposal | Written |
| Literature review | Week 2 | 2-page review | Written |
| Data pipeline | Week 3 | Working code | Code review |
| Baseline | Week 4 | Baseline results | Written |
| Method | Week 5 | Method results | Written |
| Ablations | Week 6 | Ablation results | Written |
| Draft report | Week 9 | Full draft | Written |
| Final report | Week 12 | Report, code, talk | Rubric |

> **Exercise 13.6:** Review the milestones. Which are you most concerned about? What will you do to ensure you meet them?

---

## 13.7 Project Report Template

### 13.7.1 The NeurIPS Format

The NeurIPS format is the standard for ML papers. It is a two-column format with specific font, margin, and citation styles. The LaTeX template is available at https://neurips.cc/Conferences/2024/PaperInformation/StyleFiles.

### 13.7.2 The Structure

**Title and Authors.** The title should be descriptive. List all authors.

**Abstract.** 150–250 words. Summarize the question, method, results, and conclusion.

**1. Introduction.** Motivation, problem statement, contributions.

**2. Related Work.** Literature review and gap.

**3. Method.** Your approach, with mathematical details.

**4. Experiments.** Datasets, baselines, metrics, implementation details.

**5. Results.** Main results, ablations, statistical tests.

**6. Discussion.** Interpretation, limitations, future work.

**7. Conclusion.** Summary of contributions.

**References.** All cited papers.

**Appendix.** Additional details.

### 13.7.3 Worked Example 13.2: A Report Outline

**Title:** Data Augmentation for Fine-Grained Flower Classification: An Ablation Study

**Abstract:** We investigate the effectiveness of data augmentation strategies for fine-grained flower classification. We fine-tune a pre-trained ResNet-50 on the Oxford Flowers-102 dataset and compare six augmentation strategies. We find that random crop and horizontal flip are the most effective, while color jitter and cutout provide smaller gains. Mixup provides no benefit in this setting. Our results suggest that simple augmentations are sufficient for fine-grained classification.

**1. Introduction.** Fine-grained classification is challenging because classes are visually similar. Data augmentation is a standard technique, but the relative effectiveness of different augmentations is unclear. We conduct a systematic ablation.

**2. Related Work.** Shorten & Khoshgoftaar (2019) survey image data augmentation. Zhang et al. (2018) introduce mixup. DeVries & Taylor (2017) introduce cutout. However, no study systematically compares augmentations for fine-grained classification.

**3. Method.** We fine-tune a pre-trained ResNet-50 on Oxford Flowers-102. We use a two-phase training strategy: feature extraction (5 epochs) followed by fine-tuning (20 epochs). We compare six augmentation strategies.

**4. Experiments.** Dataset: Oxford Flowers-102 (102 classes, ~8,000 images). Baseline: no augmentation. Metrics: top-1 accuracy, macro F1. Implementation: PyTorch, Adam optimizer, cosine annealing.

**5. Results.** Random crop and horizontal flip provide the largest gains (from 85% to 93%). Color jitter and cutout provide smaller gains (to 94%). Mixup provides no benefit. Combinations of augmentations do not improve over the best single augmentation.

**6. Discussion.** Simple augmentations are sufficient for fine-grained classification. More complex augmentations may require larger datasets or different architectures. Limitations: single dataset, single architecture.

**7. Conclusion.** We provide a systematic ablation of augmentation strategies for fine-grained flower classification. Our results suggest that random crop and horizontal flip are the most effective.

### 13.7.4 Common Writing Mistakes

**Overclaiming.** "Our method solves fine-grained classification" is an overclaim. "Our method improves accuracy by 2%" is accurate.

**Vague language.** "The results are good" is vague. "The accuracy is 93.2%" is specific.

**Missing details.** "We trained a model" is insufficient. "We trained a ResNet-50 with Adam, lr=1e-4, for 20 epochs" is sufficient.

**Poor figures.** A figure with unlabeled axes is useless. Label everything.

**Inconsistent notation.** Use the same symbol for the same concept throughout.

### 13.7.5 Summary: Report Template

| Section | Content | Length |
|---------|---------|--------|
| Abstract | Summary | 150–250 words |
| Introduction | Motivation, contributions | 1–1.5 pages |
| Related Work | Literature, gap | 1–1.5 pages |
| Method | Approach | 2 pages |
| Experiments | Setup | 1–1.5 pages |
| Results | Findings | 2 pages |
| Discussion | Interpretation | 1 page |
| Conclusion | Summary | 0.5 page |
| References | Citations | — |
| Appendix | Details | — |

> **Exercise 13.7:** Write the abstract and introduction for your report. Bring them to your next individual meeting.

---

## 13.8 Project Presentation Guidelines

### 13.8.1 The 15-Minute Talk

The presentation is 15 minutes plus 5 minutes of Q&A. The structure:

**Slide 1: Title.** Title, authors, affiliation.

**Slide 2: Motivation.** Why does this question matter?

**Slide 3: Research question.** One sentence.

**Slide 4: Related work.** Prior work and gap.

**Slide 5: Method.** Your approach.

**Slides 6–8: Experiments.** Setup, results, ablations.

**Slide 9: Discussion.** Interpretation, limitations.

**Slide 10: Conclusion.** Summary.

**Slide 11: Acknowledgments.** Thank collaborators.

**Slide 12: Questions.** Invite questions.

### 13.8.2 Tips for a Good Talk

**Tell a story.** The talk should have a narrative arc: motivation → question → method → results → conclusion.

**One idea per slide.** Do not overload slides.

**Use visuals.** Figures and diagrams are more effective than text.

**Practice.** Rehearse at least twice. Time yourself.

**Speak clearly.** Slow down. Pause between ideas.

**Make eye contact.** Engage the audience.

**Be honest.** Acknowledge limitations.

**Prepare for questions.** Anticipate what the audience will ask.

### 13.8.3 Common Presentation Mistakes

**Too many slides.** 30 slides in 15 minutes is too many.

**Too much text.** Slides are not documents.

**Reading from slides.** The audience can read faster than you can speak.

**No practice.** An unpracticed talk is obvious.

**Running over time.** Respect the time limit.

**Defensive answers.** If you do not know, say so.

### 13.8.4 Summary: Presentation Guidelines

| Aspect | Recommendation |
|--------|----------------|
| Length | 15 minutes |
| Slides | 10–15 |
| Structure | Motivation → question → method → results → conclusion |
| Visuals | Figures over text |
| Practice | At least twice |
| Q&A | 5 minutes, be honest |

> **Exercise 13.8:** Create your presentation slides. Rehearse the talk. Time yourself.

---

## 13.9 Common Pitfalls

### 13.9.1 Pitfall: Vague Question

**Symptom:** "Does deep learning work for X?"

**Problem:** Not falsifiable.

**Fix:** Refine to a specific, testable hypothesis.

### 13.9.2 Pitfall: No Baseline

**Symptom:** You report your method's accuracy but not a baseline's.

**Problem:** Without a baseline, you cannot know whether your method is an improvement.

**Fix:** Implement a simple baseline (linear regression, XGBoost, ResNet-18).

### 13.9.3 Pitfall: No Ablations

**Symptom:** You report your method's accuracy but not ablations.

**Problem:** You cannot know which components matter.

**Fix:** Design ablations that isolate the effect of each component.

### 13.9.4 Pitfall: No Statistical Tests

**Symptom:** You report a single number.

**Problem:** A single number is not evidence.

**Fix:** Run multiple seeds, report mean ± std, perform statistical tests.

### 13.9.5 Pitfall: Irreproducible Code

**Symptom:** Your code does not run on another machine.

**Problem:** The results cannot be verified.

**Fix:** Set seeds, log hyperparameters, write a README, use a Docker container.

### 13.9.6 Pitfall: Overclaiming

**Symptom:** "Our method solves X."

**Problem:** The evidence does not support the claim.

**Fix:** Hedge your claims. "Our method improves accuracy by 2% on this dataset."

### 13.9.7 Pitfall: Poor Writing

**Symptom:** The report is hard to read.

**Problem:** The contribution is obscured.

**Fix:** Revise. Get feedback. Read it aloud.

### 13.9.8 Pitfall: Starting Too Late

**Symptom:** You are cramming in the last two weeks.

**Problem:** The project is rushed.

**Fix:** Start early. Work steadily.

### 13.9.9 Summary: Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Vague question | Specific, falsifiable hypothesis |
| No baseline | Implement a baseline |
| No ablations | Design ablations |
| No statistical tests | Multiple runs, statistical tests |
| Irreproducible code | Seeds, logs, README, Docker |
| Overclaiming | Hedge claims |
| Poor writing | Revise, get feedback |
| Starting too late | Start early |

> **Exercise 13.9:** Review the pitfalls. Which are you most at risk of? What will you do to avoid them?

---

## 13.10 Resources for Capstone Projects

### 13.10.1 Datasets

**Tabular:** UCI ML Repository, Kaggle, OpenML.

**Image:** CIFAR-10/100, ImageNet, Oxford Flowers-102, Stanford Cars, Food-101, COCO.

**Text:** IMDB, AG News, SQuAD, GLUE.

**Audio:** Speech Commands, LibriSpeech, AudioSet.

**Multimodal:** MS COCO, Flickr30K, Conceptual Captions.

### 13.10.2 Compute

**Free:** Google Colab, Kaggle Kernels, SageMaker Studio Lab.

**Paid:** AWS, GCP, Azure.

**Institutional:** Your university's cluster.

### 13.10.3 Tools

**Version control:** Git, DVC.

**Experiment tracking:** Weights & Biases, MLflow, TensorBoard.

**Hyperparameter tuning:** Optuna, Ray Tune.

**Environment:** Conda, Docker.

### 13.10.4 Writing

**LaTeX template:** NeurIPS, ICML, ICLR, CVPR, ACL.

**Reference manager:** Zotero, Mendeley, Paperpile.

**Grammar:** Grammarly, LanguageTool.

### 13.10.5 Mentorship

**Your advisor.** Meet weekly.

**Peer reviewers.** Exchange drafts with classmates.

**Reading group.** Discuss papers.

**Online communities.** TORA Discord, r/MachineLearning.

### 13.10.6 Summary: Resources

| Category | Resources |
|----------|-----------|
| Datasets | UCI, Kaggle, CIFAR, ImageNet, GLUE |
| Compute | Colab, Kaggle, AWS, GCP |
| Tools | Git, W&B, Optuna, Docker |
| Writing | NeurIPS template, Zotero |
| Mentorship | Advisor, peers, reading group |

> **Exercise 13.10:** Identify the resources you need for your capstone project. Make a plan to obtain them.

---

## 13.11 Module Summary

| Concept | Description |
|---------|-------------|
| Capstone project | Original research contribution |
| Research question | Specific, falsifiable, novel, feasible |
| Literature review | At least 5 papers |
| Codebase | Reproducible, modular, documented |
| Experiments | Baselines, ablations, statistical tests |
| Report | 8–12 pages, NeurIPS format |
| Presentation | 15 minutes, 10–15 slides |
| Timeline | 12 weeks, 8 milestones |
| Proposal | 1-page document |
| Evaluation | Rubric with 8 criteria |
| Common pitfalls | Vague question, no baseline, no ablations |

---

## 13.12 Capstone Thread: Beginning Your Project

By the end of this module, you will be able to define a research question; scope a project to fit the available time and resources; plan the project timeline; identify the resources you need; write a project proposal; anticipate common pitfalls; and articulate what makes a successful capstone project.

In Module 14, we will explore **concrete project options** — specific examples you can adapt or use as inspiration.

In Module 15, we will cover **writing and publishing** — how to communicate your results to the research community.

---

## 13.13 Additional Resources

### On Research Projects

- "How to Do Great Research" by Nick Feamster: https://greatresearch.org/
- "The Craft of Research" by Booth, Colomb, and Williams.
- "Writing Science" by Joshua Schimel.

### On Project Management

- "Deep Work" by Cal Newport.
- "Getting Things Done" by David Allen.
- "The Ph.D. Grind" by Philip Guo.

### On Writing

- "How to Write a Great Research Paper" by Simon Peyton Jones: https://www.microsoft.com/en-us/research/
- "Ten Simple Rules for Structuring Papers" by Mensh & Kording.
- "The Elements of Style" by Strunk & White.

### On Presentation

- "TED Talks" by Chris Anderson.
- "Presentation Zen" by Garr Reynolds.
- "How to Give a Great Research Talk" by Simon Peyton Jones.

### On Reproducibility

- ML Reproducibility Checklist: https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf
- "Reproducibility in Machine Learning" (Stanford): https://cs.stanford.edu/

---

*End of Module 13*