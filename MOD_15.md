# Module 15: Writing and Publishing Your Research

## TORA CS 336: PyTorch for Research

**Institution:** The Open Research Academy (TORA)

**Instructor:** Alpha Alimamy Kamara

**Module Length:** 1 week (2 lectures + 1 discussion section + 1 writing workshop)

**Prerequisites:** Modules 1–14 completed; a capstone project underway; a draft of at least the method and results sections.

---

## 15.0 Module Overview

You have posed a research question, reviewed the literature, designed experiments, implemented models, and collected results. Now comes the hardest part: **communicating what you found**. A result that is not communicated is a result that does not exist. Writing is not the final step of research; it is the step that makes research real.

This module is about **writing and publishing research**. We will begin with the anatomy of a research paper, then develop the craft of scientific writing, then examine the publication process from workshop to conference to journal. Along the way, we will encounter the common pitfalls that lead to rejection and the best practices that lead to acceptance.

The pedagogical approach here is **structure first, style second**. A paper with clear structure and honest reporting will be accepted even if the prose is plain. A paper with beautiful prose but muddled structure will be rejected. We will focus on structure, then refine style.

By the end of this module, you will be able to structure a research paper; write each section effectively; create clear figures and tables; avoid overclaiming; choose a venue for submission; respond to reviews; and articulate why writing is a core research skill.

The module is self-paced, but the writing workshop is scheduled. Bring a draft.

---

## 15.1 The Anatomy of a Research Paper

### 15.1.1 The Standard Structure

A research paper in machine learning typically follows this structure:

**Title.** A concise, descriptive phrase that captures the contribution.

**Abstract.** A 150–250 word summary of the question, method, results, and conclusion.

**Introduction.** Motivation, problem statement, contributions, and a roadmap of the paper.

**Related Work.** A synthesis of prior work, grouped by theme, ending with the gap the paper addresses.

**Method.** A precise description of the approach, with mathematical details.

**Experiments.** Datasets, baselines, metrics, implementation details, and experimental setup.

**Results.** The main findings, with tables and figures.

**Discussion.** Interpretation of the results, limitations, and future work.

**Conclusion.** A summary of the contributions.

**References.** All cited papers.

**Appendix.** Additional details that do not fit in the main text.

### 15.1.2 The Hourglass Structure

A well-written paper follows an **hourglass structure**:

- The **introduction** starts broad (motivation) and narrows to the specific contribution.
- The **method** and **experiments** are narrow and technical.
- The **discussion** and **conclusion** broaden again to interpret the results and discuss implications.

This structure guides the reader from the general to the specific and back to the general.

### 15.1.3 The Contribution Statement

Every paper must state its contributions clearly. A contribution statement typically has three parts:

**What:** What did you do?

**Why:** Why does it matter?

**How:** How is it different from prior work?

**Worked example 15.1:** A contribution statement for a paper on transfer learning.

"We propose a systematic study of transfer learning for fine-grained flower classification. Unlike prior work, which focuses on natural image classification, our study examines the effect of dataset size and domain similarity on fine-grained tasks. We find that feature extraction is sufficient when the dataset is small, while fine-tuning is necessary when the dataset is large."

### 15.1.4 Summary: Anatomy of a Paper

| Section | Purpose | Length |
|---------|---------|--------|
| Title | Capture the contribution | 10–15 words |
| Abstract | Summarize | 150–250 words |
| Introduction | Motivate, state contributions | 1–1.5 pages |
| Related Work | Synthesize prior work | 1–1.5 pages |
| Method | Describe the approach | 2 pages |
| Experiments | Describe the setup | 1–1.5 pages |
| Results | Present findings | 2 pages |
| Discussion | Interpret, limitations | 1 page |
| Conclusion | Summarize | 0.5 page |
| References | Cite | — |
| Appendix | Extra details | — |

> **Exercise 15.1:** Outline your capstone paper using the structure above. Write one sentence for each section.

---

## 15.2 Writing Each Section

### 15.2.1 The Title

The title should be:

**Descriptive.** It should tell the reader what the paper is about.

**Concise.** 10–15 words is ideal.

**Specific.** "A Study of Transfer Learning" is vague. "Transfer Learning for Fine-Grained Flower Classification: An Ablation Study" is specific.

**Searchable.** Include keywords that readers will search for.

**Worked example 15.2:** Good and bad titles.

**Bad:** "Deep Learning for Images"

**Good:** "Data Augmentation for Fine-Grained Flower Classification: An Ablation Study"

**Bad:** "A New Method for Classification"

**Good:** "Residual Connections Improve Deep Network Training"

### 15.2.2 The Abstract

The abstract is the most-read part of the paper. It should:

**State the problem.** What question does the paper address?

**State the method.** What approach did you take?

**State the results.** What did you find?

**State the conclusion.** What does it mean?

The abstract should be self-contained: a reader should be able to understand the paper's contribution from the abstract alone.

**Worked example 15.3:** A well-written abstract.

"We investigate the effectiveness of data augmentation strategies for fine-grained flower classification. We fine-tune a pre-trained ResNet-50 on the Oxford Flowers-102 dataset and compare six augmentation strategies. We find that random crop and horizontal flip are the most effective, improving accuracy from 85% to 93%. Color jitter and cutout provide smaller gains, and mixup provides no benefit. Our results suggest that simple augmentations are sufficient for fine-grained classification."

### 15.2.3 The Introduction

The introduction should:

**Motivate the problem.** Why does this question matter?

**State the problem.** What is the specific question?

**Summarize prior work.** What has been done?

**Identify the gap.** What is missing?

**State the contributions.** What did you do?

**Roadmap the paper.** What is in each section?

The introduction is where you convince the reader to keep reading.

### 15.2.4 The Related Work

The related work should:

**Group papers by theme.** Do not list papers chronologically.

**Summarize each group.** What is the common approach?

**Identify the gap.** What is missing from prior work?

**Position your contribution.** How does your work differ?

The related work should end with a paragraph that states the gap and positions your contribution.

### 15.2.5 The Method

The method should:

**Be precise.** A reader should be able to reimplement your method.

**Include mathematics.** Write the equations explicitly.

**Justify design choices.** Why did you choose this architecture, this loss, this optimizer?

**Reference the code.** If code is available, link to it.

The method should be self-contained: a reader should not need to read prior work to understand your method.

### 15.2.6 The Experiments

The experiments should:

**Describe the datasets.** What data did you use? How was it split? How was it preprocessed?

**Describe the baselines.** What did you compare against?

**Describe the metrics.** How did you evaluate?

**Describe the implementation.** What hardware? What hyperparameters? How many runs?

The experiments should be reproducible: a reader should be able to replicate your setup.

### 15.2.7 The Results

The results should:

**Present the main findings.** Use tables and figures.

**Report statistics.** Mean ± std over multiple runs.

**Perform significance tests.** Do not just report numbers.

**Analyze the results.** What do they mean?

The results should be honest: report negative results and limitations.

### 15.2.8 The Discussion

The discussion should:

**Interpret the results.** What do they mean in the context of prior work?

**Acknowledge limitations.** What are the weaknesses of your study?

**Suggest future work.** What are the open questions?

The discussion should be honest and thoughtful.

### 15.2.9 The Conclusion

The conclusion should:

**Summarize the contributions.** What did you do?

**Restate the main finding.** What did you find?

**State the takeaway.** What should the reader remember?

The conclusion should be brief: 0.5 page is sufficient.

### 15.2.10 Summary: Writing Each Section

| Section | Key Question | Common Mistake |
|---------|--------------|----------------|
| Title | What is the contribution? | Too vague |
| Abstract | What did you find? | Too long |
| Introduction | Why does it matter? | Too broad |
| Related Work | What is the gap? | Listing papers |
| Method | How does it work? | Missing details |
| Experiments | What did you run? | Missing hyperparameters |
| Results | What did you find? | No statistics |
| Discussion | What does it mean? | Overclaiming |
| Conclusion | What is the takeaway? | Too long |

> **Exercise 15.2:** Write the abstract and introduction for your capstone paper. Bring them to the writing workshop.

---

## 15.3 Figures and Tables

### 15.3.1 The Purpose of Figures

Figures communicate information that is hard to convey in text. A good figure:

**Is self-contained.** The caption explains what the figure shows.

**Has labeled axes.** Every axis has a label and units.

**Has a legend.** If multiple series are plotted, a legend identifies them.

**Is readable.** Font sizes are large enough to read.

**Is honest.** The y-axis starts at zero, or the truncation is clearly marked.

### 15.3.2 Common Figure Types

**Line plots:** Show trends over time or across a parameter.

**Bar charts:** Compare categories.

**Scatter plots:** Show relationships between variables.

**Heatmaps:** Show matrices (e.g., confusion matrices, attention maps).

**Histograms:** Show distributions.

**Box plots:** Show distributions with quartiles.

**Architecture diagrams:** Show the structure of a model.

### 15.3.3 The Purpose of Tables

Tables present precise numbers that are hard to read from a figure. A good table:

**Has a clear caption.** The caption explains what the table shows.

**Has labeled rows and columns.** Every row and column has a label.

**Reports statistics.** Mean ± std, not just mean.

**Highlights the best result.** Bold or underline the best number.

**Is honest.** Do not cherry-pick.

### 15.3.4 Worked Example 15.4: A Results Table

| Method | Accuracy (%) | F1 (%) | Time (min) |
|--------|-------------|--------|------------|
| ResNet-18 (scratch) | 85.2 ± 0.3 | 84.8 ± 0.4 | 45 |
| ResNet-18 (feature extraction) | 91.5 ± 0.2 | 91.2 ± 0.3 | 10 |
| ResNet-18 (fine-tuning) | **93.2 ± 0.1** | **93.0 ± 0.2** | 60 |
| ResNet-50 (fine-tuning) | 94.1 ± 0.2 | 93.9 ± 0.2 | 120 |

The best result in each column is bolded. The table reports mean ± std over 3 runs. The time column shows the training time.

### 15.3.5 Summary: Figures and Tables

| Aspect | Figure | Table |
|--------|--------|-------|
| Purpose | Show trends | Show precise numbers |
| Caption | Self-contained | Self-contained |
| Labels | Axes, legend | Rows, columns |
| Statistics | Error bars | Mean ± std |
| Best result | — | Bold |
| Honesty | No truncation | No cherry-picking |

> **Exercise 15.3:** Create one figure and one table for your capstone results. Ensure they are self-contained and honest.

---

## 15.4 Writing Style

### 15.4.1 Precision

Scientific writing is precise. Every claim should be specific and supported by evidence.

**Bad:** "The model performs well."

**Good:** "The model achieves 93.2% accuracy on the test set."

**Bad:** "The results are significant."

**Good:** "The improvement is statistically significant (paired t-test, p < 0.01)."

### 15.4.2 Honesty

Scientific writing is honest. Acknowledge limitations, report negative results, and avoid overclaiming.

**Bad:** "Our method solves fine-grained classification."

**Good:** "Our method improves accuracy by 2% on the Oxford Flowers-102 dataset."

**Bad:** "We did not observe any limitations."

**Good:** "Our study is limited to a single dataset and a single architecture."

### 15.4.3 Clarity

Scientific writing is clear. Use plain language, short sentences, and active voice.

**Bad:** "The utilization of data augmentation techniques was implemented in order to facilitate the enhancement of model performance."

**Good:** "We used data augmentation to improve model performance."

### 15.4.4 Conciseness

Scientific writing is concise. Every word should earn its place.

**Bad:** "It is important to note that the results of our experiments demonstrate that..."

**Good:** "Our experiments show that..."

### 15.4.5 The Active Voice

Use the active voice. It is clearer and more direct.

**Passive:** "The model was trained by us."

**Active:** "We trained the model."

### 15.4.6 Tense

Use the present tense for general truths and the past tense for specific experiments.

**Present:** "ResNet uses residual connections."

**Past:** "We trained ResNet-50 for 20 epochs."

### 15.4.7 Summary: Writing Style

| Principle | Bad | Good |
|-----------|-----|------|
| Precision | "Performs well" | "93.2% accuracy" |
| Honesty | "Solves X" | "Improves X by 2%" |
| Clarity | "Utilization of" | "We used" |
| Conciseness | "It is important to note" | (delete) |
| Active voice | "Was trained by us" | "We trained" |
| Tense | "ResNet used" | "ResNet uses" |

> **Exercise 15.4:** Revise a paragraph from your draft to improve precision, honesty, clarity, and conciseness.

---

## 15.5 Where to Submit

### 15.5.1 The Publication Landscape

Research is published in three main venues:

**Workshops:** Small, focused meetings co-located with conferences. Good for early-stage work and feedback.

**Conferences:** Peer-reviewed meetings. The primary venue for ML research.

**Journals:** Peer-reviewed publications. Longer review cycles, more comprehensive papers.

### 15.5.2 Major ML Conferences

| Conference | Focus | Deadline | Acceptance Rate |
|------------|-------|----------|-----------------|
| NeurIPS | General ML | May | ~25% |
| ICML | General ML | January | ~25% |
| ICLR | General ML | September | ~30% |
| CVPR | Computer Vision | November | ~25% |
| ICCV | Computer Vision | March | ~25% |
| ECCV | Computer Vision | March | ~25% |
| ACL | NLP | February | ~25% |
| EMNLP | NLP | May | ~25% |
| AAAI | General AI | August | ~20% |

### 15.5.3 Major ML Journals

| Journal | Focus | Impact Factor |
|---------|-------|---------------|
| JMLR | General ML | 5.0 |
| TPAMI | Pattern Analysis | 24.3 |
| TNNLS | Neural Networks | 10.4 |
| AI | Artificial Intelligence | 14.0 |

### 15.5.4 Workshops

Workshops are a good venue for early-stage work. They have lower acceptance bars and provide feedback from a focused community. Examples:

- NeurIPS Workshops
- ICML Workshops
- ICLR Workshops
- CVPR Workshops

### 15.5.5 Preprints

**arXiv** is the standard preprint server for ML. Posting a preprint:

- Establishes priority.
- Allows others to cite your work before publication.
- Provides a permanent record.

### 15.5.6 Choosing a Venue

When choosing a venue, consider:

**Fit.** Does your paper fit the venue's scope?

**Timeline.** When is the deadline? Can you meet it?

**Audience.** Who will read your paper?

**Prestige.** How competitive is the venue?

**Open access.** Is the venue open access?

### 15.5.7 Summary: Where to Submit

| Venue | Type | Deadline | Acceptance Rate |
|-------|------|----------|-----------------|
| NeurIPS | Conference | May | ~25% |
| ICML | Conference | January | ~25% |
| ICLR | Conference | September | ~30% |
| CVPR | Conference | November | ~25% |
| ACL | Conference | February | ~25% |
| JMLR | Journal | Rolling | — |
| TPAMI | Journal | Rolling | — |
| arXiv | Preprint | Rolling | — |

> **Exercise 15.5:** Identify three venues where your capstone project could be submitted. For each, note the deadline, acceptance rate, and fit.

---

## 15.6 The Review Process

### 15.6.1 How Peer Review Works

The peer review process:

1. **Submission.** Authors submit a paper to a venue.
2. **Assignment.** The editor assigns the paper to 2–4 reviewers.
3. **Review.** Reviewers read the paper and write reviews.
4. **Decision.** The editor makes a decision (accept, reject, revise).
5. **Rebuttal.** Authors respond to the reviews.
6. **Final decision.** The editor makes the final decision.

### 15.6.2 What Reviewers Look For

**Novelty:** Is the contribution new?

**Significance:** Is the contribution important?

**Correctness:** Are the methods and results correct?

**Clarity:** Is the paper well-written?

**Reproducibility:** Can the results be reproduced?

### 15.6.3 The Review Form

A typical review form has:

**Summary:** A brief summary of the paper.

**Strengths:** What the paper does well.

**Weaknesses:** What the paper does poorly.

**Questions:** Questions for the authors.

**Suggestions:** Suggestions for improvement.

**Recommendation:** Accept, reject, or revise.

### 15.6.4 Responding to Reviews

When responding to reviews:

**Be respectful.** Reviewers are volunteering their time.

**Be specific.** Address each point.

**Be concise.** Do not ramble.

**Be gracious.** Thank the reviewers for their feedback.

**Be honest.** If a reviewer is wrong, say so politely.

### 15.6.5 Common Reasons for Rejection

**Incremental contribution.** The paper does not advance the field.

**Weak experiments.** No baselines, no ablations, no statistics.

**Poor writing.** The paper is hard to read.

**Overclaiming.** The conclusions go beyond the evidence.

**Irreproducible.** The code or data is not available.

### 15.6.6 Summary: The Review Process

| Stage | Description |
|-------|-------------|
| Submission | Authors submit |
| Assignment | Editor assigns reviewers |
| Review | Reviewers write reviews |
| Decision | Editor decides |
| Rebuttal | Authors respond |
| Final decision | Editor decides |
| Criteria | Novelty, significance, correctness, clarity, reproducibility |

> **Exercise 15.6:** Simulate a peer review. Exchange drafts with a classmate. Write a review using the structure above. Discuss.

---

## 15.7 Open Science

### 15.7.1 What Is Open Science?

**Open science** is the practice of making research transparent, accessible, and reproducible. It includes:

**Open access:** Making papers freely available.

**Open code:** Sharing code on GitHub.

**Open data:** Sharing datasets.

**Open peer review:** Making reviews public.

### 15.7.2 Why Open Science?

Open science:

**Accelerates research.** Others can build on your work.

**Improves reproducibility.** Others can verify your results.

**Increases impact.** Open papers are cited more.

**Promotes equity.** Researchers without institutional access can read your work.

### 15.7.3 How to Practice Open Science

**Post preprints.** Use arXiv.

**Share code.** Use GitHub with a permissive license.

**Share data.** Use Zenodo, Figshare, or Hugging Face Datasets.

**Document everything.** Write READMEs, data cards, and model cards.

**Use open licenses.** MIT, Apache, CC BY.

### 15.7.4 Summary: Open Science

| Practice | Tool |
|----------|------|
| Preprints | arXiv |
| Code | GitHub |
| Data | Zenodo, Figshare |
| Documentation | README, data cards |
| Licenses | MIT, Apache, CC BY |

> **Exercise 15.7:** Plan how you will practice open science for your capstone. What will you share? Where?

---

## 15.8 The Writing Workshop

### 15.8.1 The Workshop Format

The writing workshop is a 2-hour session where students exchange drafts and provide feedback. The format:

1. **Pair up.** Each student is paired with a partner.
2. **Read.** Read your partner's draft.
3. **Write feedback.** Use the review form from Section 15.6.
4. **Discuss.** Share feedback verbally.
5. **Revise.** Revise your draft based on feedback.

### 15.8.2 The Feedback Template

Use the following template for feedback:

**Summary:** What is the paper about?

**Strengths:** What does the paper do well?

**Weaknesses:** What could be improved?

**Specific comments:** Line-by-line feedback.

**Suggestions:** How to improve the paper.

### 15.8.3 How to Give Good Feedback

**Be specific.** "The introduction is unclear" is vague. "The second paragraph of the introduction does not state the research question" is specific.

**Be constructive.** "This could be improved by adding a baseline" is constructive. "This is wrong" is not.

**Be kind.** Remember that your partner has worked hard.

**Be honest.** Honest feedback is the most valuable.

### 15.8.4 How to Receive Feedback

**Listen.** Do not interrupt.

**Take notes.** Write down every point.

**Ask questions.** If you do not understand, ask.

**Do not argue.** You can disagree later, but first understand.

**Thank your partner.** Feedback is a gift.

### 15.8.5 Summary: The Writing Workshop

| Stage | Action |
|-------|--------|
| Pair up | With a classmate |
| Read | Your partner's draft |
| Write feedback | Using the template |
| Discuss | Share feedback |
| Revise | Improve your draft |

> **Exercise 15.8:** Exchange drafts with a classmate. Provide feedback using the template. Revise your draft.

---

## 15.9 Common Writing Pitfalls

### 15.9.1 Pitfall: Overclaiming

**Symptom:** "Our method solves X."

**Problem:** The evidence does not support the claim.

**Fix:** Hedge. "Our method improves accuracy by 2% on this dataset."

### 15.9.2 Pitfall: Vague Language

**Symptom:** "The results are good."

**Problem:** Not specific.

**Fix:** "The accuracy is 93.2%."

### 15.9.3 Pitfall: Missing Details

**Symptom:** "We trained a model."

**Problem:** Not reproducible.

**Fix:** "We trained ResNet-50 with Adam, lr=1e-4, for 20 epochs."

### 15.9.4 Pitfall: Poor Figures

**Symptom:** A figure with unlabeled axes.

**Problem:** Unreadable.

**Fix:** Label axes, add legends, use readable font sizes.

### 15.9.5 Pitfall: Inconsistent Notation

**Symptom:** Different symbols for the same concept.

**Problem:** Confusing.

**Fix:** Use the same symbol throughout.

### 15.9.6 Pitfall: Passive Voice Overuse

**Symptom:** "The model was trained."

**Problem:** Less clear.

**Fix:** "We trained the model."

### 15.9.7 Pitfall: Long Sentences

**Symptom:** Sentences that run on for 50 words.

**Problem:** Hard to read.

**Fix:** Break into shorter sentences.

### 15.9.8 Pitfall: Jargon

**Symptom:** Terms that only experts understand.

**Problem:** Excludes readers.

**Fix:** Define terms, use plain language.

### 15.9.9 Summary: Common Writing Pitfalls

| Pitfall | Fix |
|---------|-----|
| Overclaiming | Hedge |
| Vague language | Be specific |
| Missing details | Include hyperparameters |
| Poor figures | Label everything |
| Inconsistent notation | Use same symbols |
| Passive voice | Use active |
| Long sentences | Break up |
| Jargon | Define terms |

> **Exercise 15.9:** Review your draft for the pitfalls above. Correct as many as you can.

---

## 15.10 Module Summary

| Concept | Description |
|---------|-------------|
| Anatomy of a paper | Title, abstract, intro, related work, method, experiments, results, discussion, conclusion |
| Hourglass structure | Broad → narrow → broad |
| Contribution statement | What, why, how |
| Title | Descriptive, concise, specific |
| Abstract | Problem, method, results, conclusion |
| Introduction | Motivate, state, summarize, identify gap, contributions |
| Related work | Group by theme, identify gap |
| Method | Precise, mathematical, justified |
| Experiments | Datasets, baselines, metrics, implementation |
| Results | Main findings, statistics, significance |
| Discussion | Interpretation, limitations, future work |
| Conclusion | Summary, takeaway |
| Figures | Self-contained, labeled, honest |
| Tables | Precise, statistics, best result |
| Writing style | Precise, honest, clear, concise, active voice |
| Venues | Workshops, conferences, journals, preprints |
| Peer review | Novelty, significance, correctness, clarity, reproducibility |
| Open science | Preprints, code, data, documentation, licenses |

---

## 15.11 Capstone Thread: Writing Your Paper

By the end of this module, you will be able to structure a research paper; write each section effectively; create clear figures and tables; avoid overclaiming; choose a venue for submission; respond to reviews; and articulate why writing is a core research skill.

For your capstone project, you will use these skills to write your report, create figures and tables, and prepare your presentation.

In Module 16, we will study **tools and infrastructure** for research.

---

## 15.12 Additional Resources

### On Writing

- "How to Write a Great Research Paper" by Simon Peyton Jones: https://www.microsoft.com/en-us/research/
- "Ten Simple Rules for Structuring Papers" by Mensh & Kording: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005619
- "The Elements of Style" by Strunk & White.
- "Writing Science" by Joshua Schimel.
- "Style: Lessons in Clarity and Grace" by Joseph Williams.

### On Figures and Tables

- "The Visual Display of Quantitative Information" by Edward Tufte.
- "Fundamentals of Data Visualization" by Claus Wilke.

### On Peer Review

- "How to Write a Peer Review" (PLOS): https://plos.org/
- "Reviewing a Paper" (Nature): https://www.nature.com/

### On Open Science

- "Open Science" (UNESCO): https://en.unesco.org/science-sustainable-future/open-science
- arXiv: https://arxiv.org/
- GitHub: https://github.com/
- Zenodo: https://zenodo.org/

### On Venues

- NeurIPS: https://neurips.cc/
- ICML: https://icml.cc/
- ICLR: https://iclr.cc/
- CVPR: https://cvpr.thecvf.com/
- ACL: https://aclanthology.org/

---

*End of Module 15*