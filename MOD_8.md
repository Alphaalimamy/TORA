# Module 8: How to Read a Scientific Paper

## TORA | CS 336: PyTorch for Research

**Instructor:** Alpha Alimamy Kamara

**Module Length:** 1 week (2 lectures + 1 discussion section)

**Prerequisites:** Module 7 completed; a set of at least 10 papers from your literature search.

---

## 8.0 Module Overview

In Module 7, we learned where to find papers and how to manage them. But finding a paper is only the first step. The harder skill is **reading** it — extracting its contributions, evaluating its claims, and connecting it to your own work. A researcher who cannot read papers effectively will drown in the literature, waste time on irrelevant work, and miss critical details.

This module is about **critical reading**: the systematic process of understanding a scientific paper, evaluating its claims, and extracting the information you need. We will begin with the three-pass method, then develop a critical appraisal framework, then work through concrete examples from the ML literature. Along the way, we will encounter the common pitfalls that lead researchers to misunderstand or misrepresent prior work.

The pedagogical approach here is **method first, practice second**. We will introduce the three-pass method and the critical appraisal framework, then apply them to real papers. The goal is not to memorize a checklist but to internalize a repeatable process for reading any paper in any field.

By the end of this module, you will be able to read a paper in three passes; identify its contributions, methods, and results; evaluate its internal and external validity; ask critical questions; extract information for your own work; and articulate why critical reading is a core research skill.

The module is self-paced, but the exercises are best done in a single sitting so that you build a complete reading workflow.

---

## 8.1 The Three-Pass Method

### 8.1.1 Overview

The **three-pass method**, introduced by S. Keshav in his 2007 paper "How to Read a Paper," is a systematic approach to reading a scientific paper. The idea is to read the paper three times, with increasing depth each time. This allows you to quickly filter out irrelevant papers and to deeply understand the ones that matter.

The three passes are:

**Pass 1 (5–10 minutes):** Get the gist. Read the title, abstract, introduction, section headings, and conclusions. Look at the figures and tables. Decide if the paper is worth reading.

**Pass 2 (1 hour):** Understand the content. Read the paper carefully, but skip the proofs and derivations. Take notes on the key points. Identify what the authors did and what they found.

**Pass 3 (4–5 hours):** Deep understanding. Read the paper in detail, including the proofs and derivations. Try to reproduce the results. Identify the assumptions, the limitations, and the open questions.

### 8.1.2 Pass 1: The Skim

The goal of Pass 1 is to answer five questions:

1. **Category:** What type of paper is this? (Empirical study? New method? Survey? Theoretical analysis?)
2. **Context:** Which other papers is it related to? Which theories were used as a basis?
3. **Correctness:** Do the assumptions appear valid?
4. **Contributions:** What are the paper's main contributions?
5. **Clarity:** Is the paper well-written?

**How to do Pass 1:**

- Read the **title** carefully. It usually states the contribution.
- Read the **abstract**. It summarizes the problem, method, and results.
- Read the **introduction**. It motivates the problem and states the contributions.
- Read the **section headings** and **subheadings**. They reveal the structure.
- Read the **conclusions**. They summarize the findings.
- Look at the **figures and tables**. They often convey the key results.
- Look at the **references**. They reveal the context.

**Worked example 8.1:** Apply Pass 1 to "Attention Is All You Need."

**Title:** "Attention Is All You Need" — the contribution is a new architecture based solely on attention.

**Abstract:** The paper proposes the Transformer, a new architecture for sequence transduction based entirely on attention mechanisms, dispensing with recurrence and convolutions. It achieves state-of-the-art results on machine translation.

**Introduction:** The problem is that RNNs are slow to train and have difficulty with long-range dependencies. The contribution is the Transformer, which is faster and better.

**Section headings:** Introduction, Background, Model Architecture, Why Self-Attention, Training, Results, Conclusion.

**Figures:** Figure 1 shows the Transformer architecture. Figure 2 shows the attention mechanism. Tables show BLEU scores.

**Conclusion:** The Transformer is the first sequence transduction model relying entirely on attention, and it achieves state-of-the-art results.

**Category:** New method.

**Context:** Related to RNNs, CNNs, attention mechanisms.

**Correctness:** The assumptions (attention is sufficient, recurrence is not necessary) are plausible.

**Contributions:** The Transformer architecture; the multi-head attention mechanism; state-of-the-art results on translation.

**Clarity:** The paper is well-written and clear.

**Pass 1 conclusion:** This paper is worth reading in depth. It is central to modern NLP.

### 8.1.3 Pass 2: The Read

The goal of Pass 2 is to understand the content of the paper. You should be able to summarize the paper's main ideas to someone else.

**How to do Pass 2:**

- Read the paper carefully, but **skip the proofs and derivations**. They can be read later if needed.
- **Take notes** on the key points. Write down the problem, method, results, and limitations.
- **Look at the figures and tables** carefully. What do they show? What are the axes? What are the trends?
- **Mark the references** that you want to read later.
- **Identify the assumptions** the authors make. Are they reasonable?
- **Identify the limitations** the authors acknowledge. Are there others they do not?

**Worked example 8.2:** Apply Pass 2 to "Attention Is All You Need."

**Problem:** Sequence transduction (e.g., machine translation) is dominated by RNNs and CNNs. RNNs are sequential, which prevents parallelization. CNNs have difficulty with long-range dependencies.

**Method:** The Transformer, a new architecture based on attention. It has an encoder and a decoder. The encoder consists of a stack of identical layers, each with a multi-head self-attention mechanism and a feed-forward network. The decoder adds a cross-attention mechanism. Positional encodings are added to the input embeddings.

**Results:** State-of-the-art BLEU scores on WMT 2014 English-to-German (28.4) and English-to-French (41.8). Trained in 3.5 days on 8 GPUs, a fraction of the time of previous models.

**Assumptions:** Attention is sufficient for sequence transduction; positional encodings capture order; multi-head attention captures different types of relationships.

**Limitations:** Quadratic complexity in sequence length; requires large amounts of data; no explicit handling of very long sequences.

**Pass 2 conclusion:** The paper proposes a new architecture that is faster and better than RNNs and CNNs. The key innovation is multi-head self-attention. The results are strong, but the limitations suggest open questions.

### 8.1.4 Pass 3: The Deep Dive

The goal of Pass 3 is to fully understand the paper. You should be able to reproduce the results, identify the strengths and weaknesses, and propose extensions.

**How to do Pass 3:**

- **Read the paper in detail**, including the proofs and derivations.
- **Try to reproduce the results.** Implement the method yourself. Do you get the same results?
- **Identify the assumptions** and test them. What happens if you relax them?
- **Identify the limitations** and think about how to address them.
- **Compare with related work.** How does this paper differ from others?
- **Think about extensions.** What new questions does this paper raise?

**Worked example 8.3:** Apply Pass 3 to "Attention Is All You Need."

**Deep understanding:** The Transformer's key innovation is the self-attention mechanism, which computes a weighted sum of values based on query-key similarity. Multi-head attention applies multiple attention functions in parallel and concatenates the results. The feed-forward network is a two-layer MLP with ReLU. Positional encodings are sinusoidal functions of position.

**Reproduction:** Implementing the Transformer from scratch is a substantial task. A simplified version can be implemented in a few hundred lines of PyTorch. The results on a small translation task (e.g., IWSLT) can be compared with the paper.

**Assumptions tested:** What if we remove positional encodings? Performance drops significantly. What if we use a single attention head? Performance drops. What if we replace self-attention with a fixed random matrix? Performance drops.

**Limitations addressed:** The quadratic complexity can be reduced with sparse attention (Longformer, BigBird) or linear attention (Performer, Linformer).

**Related work:** The Transformer builds on attention mechanisms (Bahdanau et al., 2015) and has inspired a family of models (BERT, GPT, T5, ViT).

**Extensions:** The Transformer has been applied to vision (ViT), speech (Whisper), reinforcement learning (Decision Transformer), and biology (AlphaFold).

**Pass 3 conclusion:** The Transformer is a foundational paper. Its key innovation is self-attention, which has proven to be a general-purpose mechanism. Its limitations have inspired a large body of follow-up work.

### 8.1.5 Summary: The Three-Pass Method

| Pass | Time | Goal | Output |
|------|------|------|--------|
| 1 | 5–10 min | Get the gist | Decide if worth reading |
| 2 | 1 hour | Understand the content | Summarize to someone else |
| 3 | 4–5 hours | Deep understanding | Reproduce, critique, extend |

> **Exercise 8.1:** Choose a paper from your literature search. Apply Pass 1. Write down the five questions and your answers. Decide if the paper is worth reading in depth.

---

## 8.2 Critical Appraisal

### 8.2.1 What Is Critical Appraisal?

**Critical appraisal** is the process of evaluating the quality and validity of a research paper. It goes beyond understanding what the paper says to asking whether the paper's claims are justified.

The key questions of critical appraisal are:

1. **Internal validity:** Was the study designed correctly? Are the results trustworthy?
2. **External validity:** Do the results generalize? To what populations, settings, or tasks?
3. **Statistical validity:** Are the results statistically significant? Are the effect sizes meaningful?
4. **Construct validity:** Do the measurements actually measure what they claim to measure?
5. **Conclusion validity:** Do the conclusions follow from the results?

### 8.2.2 Internal Validity

**Internal validity** is the extent to which the results of a study are caused by the variables being studied, rather than by confounding factors.

Threats to internal validity in ML research:
- **Data leakage:** Training on data that should be held out.
- **Hyperparameter tuning on the test set:** Selecting hyperparameters based on test performance.
- **Cherry-picking:** Reporting only the best results.
- **Confounding:** Changing multiple variables at once.
- **Non-determinism:** Results that vary across runs.

**Questions to ask:**
- Was the train/validation/test split done correctly?
- Were hyperparameters tuned on the validation set?
- Were all results reported, or only the best?
- Were ablations performed to isolate the effect of each component?
- Were multiple runs performed? What is the variance?

### 8.2.3 External Validity

**External validity** is the extent to which the results generalize to other populations, settings, or tasks.

Threats to external validity in ML research:
- **Overfitting to a benchmark:** Tuning the method to perform well on a specific dataset.
- **Narrow evaluation:** Evaluating on only one or two datasets.
- **Unrepresentative data:** Using data that does not reflect the real-world distribution.
- **Domain shift:** The test data comes from a different distribution than the training data.

**Questions to ask:**
- Was the method evaluated on multiple datasets?
- Are the datasets representative of the real world?
- Does the method generalize to other domains?
- Are there any datasets where the method fails?

### 8.2.4 Statistical Validity

**Statistical validity** is the extent to which the results are statistically significant and the effect sizes are meaningful.

Threats to statistical validity in ML research:
- **Multiple comparisons:** Testing many hypotheses without correction.
- **Small sample sizes:** Results that are not statistically significant.
- **Ignoring variance:** Reporting a single number without confidence intervals.
- **P-hacking:** Selecting results that are statistically significant.

**Questions to ask:**
- Are the results reported with confidence intervals or standard deviations?
- How many runs were performed?
- Are the differences between methods statistically significant?
- Were multiple comparisons corrected?

### 8.2.5 Construct Validity

**Construct validity** is the extent to which the measurements actually measure what they claim to measure.

Threats to construct validity in ML research:
- **Proxy metrics:** Using accuracy as a proxy for real-world performance.
- **Biased benchmarks:** Benchmarks that are not representative.
- **Gaming the metric:** Optimizing for the metric rather than the underlying goal.

**Questions to ask:**
- Do the metrics actually measure what matters?
- Are there alternative metrics that might tell a different story?
- Is the benchmark representative of the real-world task?
- Could the method be gaming the metric?

### 8.2.6 Conclusion Validity

**Conclusion validity** is the extent to which the conclusions follow from the results.

Threats to conclusion validity in ML research:
- **Overclaiming:** Conclusions that go beyond the results.
- **Correlation vs. causation:** Confusing correlation with causation.
- **Ignoring limitations:** Not acknowledging the limitations of the study.
- **Overgeneralization:** Applying results to settings they do not apply to.

**Questions to ask:**
- Do the conclusions follow from the results?
- Are the claims supported by the evidence?
- Are the limitations acknowledged?
- Are the conclusions appropriately hedged?

### 8.2.7 Summary: Critical Appraisal

| Validity Type | Question | Threats |
|---------------|----------|---------|
| Internal | Was the study designed correctly? | Data leakage, cherry-picking |
| External | Do the results generalize? | Overfitting to benchmark |
| Statistical | Are the results significant? | Small samples, p-hacking |
| Construct | Do the measurements measure what they claim? | Proxy metrics |
| Conclusion | Do the conclusions follow? | Overclaiming |

> **Exercise 8.2:** Choose a paper from your literature search. Apply the critical appraisal framework. Write down your answers to the questions for each type of validity.

---

## 8.3 Questions to Ask While Reading

### 8.3.1 The Master List

The following questions are a master list for critical reading. Not all questions apply to all papers, but the list provides a comprehensive framework.

**Motivation:**
- What problem does the paper address?
- Why is this problem important?
- What is the current state of the art?
- What are the limitations of existing methods?

**Contributions:**
- What are the paper's main contributions?
- Are the contributions novel?
- Are the contributions significant?
- Are the contributions clearly stated?

**Method:**
- What is the proposed method?
- How does it work?
- What are the key assumptions?
- What are the design choices?
- Are the design choices justified?

**Experiments:**
- What datasets were used?
- What metrics were used?
- What baselines were compared?
- Are the baselines fair?
- Are the results statistically significant?
- Are the results reproducible?

**Results:**
- What are the main results?
- Do the results support the claims?
- Are there any surprising results?
- Are there any negative results?

**Limitations:**
- What are the limitations of the method?
- What are the failure modes?
- What are the open questions?

**Related work:**
- How does this paper relate to prior work?
- What is the key difference?
- What is the key improvement?

**Writing:**
- Is the paper well-written?
- Are the figures and tables clear?
- Is the notation consistent?
- Are the claims appropriately hedged?

### 8.3.2 The Five Questions

If you only have time for five questions, ask these:

1. **What is the problem?** (Motivation)
2. **What is the solution?** (Method)
3. **What is the evidence?** (Experiments)
4. **What are the limitations?** (Discussion)
5. **What does it mean?** (Significance)

These are the same three questions from Module 1, plus two more.

### 8.3.3 Summary: Questions to Ask

| Category | Key Question |
|----------|--------------|
| Motivation | What problem does it solve? |
| Contributions | What is new? |
| Method | How does it work? |
| Experiments | What is the evidence? |
| Results | What did they find? |
| Limitations | What are the weaknesses? |
| Related work | How does it compare? |
| Writing | Is it clear? |

> **Exercise 8.3:** Choose a paper from your literature search. Answer the five questions above. Share your answers with a classmate and discuss.

---

## 8.4 Common Pitfalls in Reading Papers

### 8.4.1 Pitfall: Reading Only the Abstract

**Symptom:** You cite a paper based only on its abstract.

**Problem:** The abstract is a summary, not a substitute for the paper. It may omit important details, overstate the results, or gloss over limitations.

**Fix:** Read at least the introduction and conclusions. If you cite the paper, read the method and experiments.

### 8.4.2 Pitfall: Trusting the Results

**Symptom:** You accept the paper's results without question.

**Problem:** Papers can have errors, biases, or misleading claims. Even peer-reviewed papers are not immune.

**Fix:** Apply the critical appraisal framework. Look for the evidence behind the claims.

### 8.4.3 Pitfall: Ignoring the Limitations

**Symptom:** You skip the limitations section.

**Problem:** The limitations section is often the most important part of the paper. It tells you what the method does not do and where it might fail.

**Fix:** Read the limitations carefully. Think about whether they apply to your use case.

### 8.4.4 Pitfall: Overgeneralizing

**Symptom:** You assume the method works for all tasks.

**Problem:** Methods are often tuned for specific benchmarks. They may not generalize.

**Fix:** Check the external validity. Look for evaluations on multiple datasets and domains.

### 8.4.5 Pitfall: Misunderstanding the Method

**Symptom:** You implement the method based on a superficial reading and get different results.

**Problem:** The method may have important details that are easy to miss.

**Fix:** Read the method section carefully. Implement it step by step. Compare with the paper.

### 8.4.6 Pitfall: Confusing Correlation with Causation

**Symptom:** You assume that because A and B are correlated, A causes B.

**Problem:** Correlation does not imply causation. There may be a confounding variable.

**Fix:** Look for controlled experiments. Ablations isolate the effect of each component.

### 8.4.7 Summary: Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Reading only the abstract | Read the method and experiments |
| Trusting the results | Apply critical appraisal |
| Ignoring limitations | Read the limitations section |
| Overgeneralizing | Check external validity |
| Misunderstanding the method | Read carefully, implement step by step |
| Confusing correlation with causation | Look for controlled experiments |

> **Exercise 8.4:** Review a paper you have read recently. Did you commit any of the pitfalls above? How would you correct your understanding?

---

## 8.5 Reading Different Types of Papers

### 8.5.1 Empirical Papers

**Empirical papers** present new experimental results. They typically have the structure: Introduction, Related Work, Method, Experiments, Results, Discussion, Conclusion.

**How to read:**
- Focus on the method and experiments.
- Check the experimental design: datasets, baselines, metrics.
- Look for ablations that isolate the effect of each component.
- Check for statistical significance.

### 8.5.2 Theoretical Papers

**Theoretical papers** present new theorems or analyses. They typically have the structure: Introduction, Related Work, Preliminaries, Main Results, Proofs, Discussion.

**How to read:**
- Focus on the assumptions and the main theorem.
- Check the proof: is it correct? Are the assumptions reasonable?
- Look for the intuition behind the theorem.
- Consider the implications for practice.

### 8.5.3 Survey Papers

**Survey papers** summarize the state of the art in a field. They typically have the structure: Introduction, Taxonomy, Methods, Datasets, Challenges, Future Directions.

**How to read:**
- Focus on the taxonomy and the challenges.
- Use the references to find the key papers.
- Look for the open questions.

### 8.5.4 Position Papers

**Position papers** argue for a particular point of view. They typically have the structure: Introduction, Argument, Evidence, Counterarguments, Conclusion.

**How to read:**
- Identify the central claim.
- Evaluate the evidence.
- Consider the counterarguments.
- Decide whether you agree.

### 8.5.5 Summary: Types of Papers

| Type | Structure | Focus |
|------|-----------|-------|
| Empirical | Method, experiments, results | Experiments |
| Theoretical | Assumptions, theorem, proof | Assumptions, proof |
| Survey | Taxonomy, methods, challenges | Taxonomy, references |
| Position | Argument, evidence, counterarguments | Central claim |

> **Exercise 8.5:** Find one paper of each type (empirical, theoretical, survey, position) in your literature search. Apply the appropriate reading strategy to each.

---

## 8.6 Worked Example: Reading a Paper in Detail

Let us apply the full framework to a paper from the ML literature.

### 8.6.1 The Paper

**Title:** "Deep Residual Learning for Image Recognition"

**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

**Year:** 2016

**Venue:** CVPR

### 8.6.2 Pass 1: Skim

**Category:** New method (architecture).

**Context:** Related to VGG, GoogLeNet, and other deep CNNs.

**Correctness:** The assumption (deeper networks are harder to train due to degradation, not overfitting) is supported by experiments.

**Contributions:** The ResNet architecture with residual connections; state-of-the-art results on ImageNet.

**Clarity:** The paper is well-written and clear.

**Decision:** Worth reading in depth.

### 8.6.3 Pass 2: Read

**Problem:** Deeper neural networks are harder to train. This is not due to overfitting (training error is also higher) but to the **degradation problem**: as depth increases, accuracy saturates and then degrades.

**Method:** The ResNet architecture. Each block computes \( \mathcal{F}(\mathbf{x}) + \mathbf{x} \), where \( \mathcal{F} \) is a residual function. The identity shortcut \( \mathbf{x} \) allows gradients to flow directly through the network, mitigating the degradation problem.

**Results:** ResNet-152 achieves 3.57% top-5 error on ImageNet, winning the ILSVRC 2015 competition. It is 8× deeper than VGG but has lower complexity.

**Assumptions:** The degradation problem is due to optimization difficulty, not capacity; residual connections make optimization easier.

**Limitations:** The method adds parameters and computation; very deep networks still require careful initialization; the optimal depth is task-dependent.

### 8.6.4 Pass 3: Deep Dive

**Deep understanding:** The residual block is a general-purpose building block. The identity shortcut is parameter-free. The bottleneck design (1×1, 3×3, 1×1 convolutions) reduces computation. The architecture can be scaled to hundreds or thousands of layers.

**Reproduction:** Implementing ResNet from scratch is a moderate task. A simplified version can be implemented in a few hundred lines of PyTorch. The results on CIFAR-10 can be compared with the paper.

**Assumptions tested:** What if we remove the identity shortcut? Performance degrades. What if we use a 1×1 convolution shortcut instead of identity? Performance is slightly worse. What if we use a different nonlinearity? Performance changes.

**Limitations addressed:** The residual connection has inspired a family of architectures (DenseNet, Highway Networks, Transformers with residual connections).

**Related work:** ResNet builds on VGG and GoogLeNet and has inspired DenseNet, Wide ResNet, ResNeXt, and EfficientNet.

**Extensions:** ResNet has been applied to object detection (Faster R-CNN with ResNet), segmentation (FCN with ResNet), and video understanding.

### 8.6.5 Critical Appraisal

**Internal validity:** The experiments are well-designed. Ablations isolate the effect of residual connections. Multiple depths are tested.

**External validity:** Results are reported on ImageNet and CIFAR-10. The method generalizes to other vision tasks.

**Statistical validity:** The results are reported with error bars. The improvements are substantial.

**Construct validity:** Top-1 and top-5 accuracy are standard metrics for image classification. They measure what they claim to measure.

**Conclusion validity:** The conclusions follow from the results. The authors acknowledge the limitations.

### 8.6.6 Reflection

Reading ResNet in depth took approximately 3 hours. The paper is clear, well-organized, and rigorous. The key insight — that identity shortcuts mitigate the degradation problem — is simple but powerful. The paper has had enormous impact, with over 100,000 citations.

> **Exercise 8.6:** Choose a foundational paper in your field. Apply the full framework: three passes, critical appraisal, and the five questions. Write a one-page summary.

---

## 8.7 Reading for Different Purposes

### 8.7.1 Reading for a Literature Review

When reading for a literature review, focus on:
- The problem the paper addresses
- The method
- The results
- How it relates to other papers

Take structured notes that you can use in your writing.

### 8.7.2 Reading for Implementation

When reading to implement a method, focus on:
- The method details
- The hyperparameters
- The training procedure
- The evaluation protocol

Take detailed notes, including equations and pseudocode.

### 8.7.3 Reading for Critique

When reading to critique a paper, focus on:
- The assumptions
- The experimental design
- The statistical validity
- The conclusions

Take notes on the strengths and weaknesses.

### 8.7.4 Reading for Inspiration

When reading for inspiration, focus on:
- The open questions
- The future work
- The limitations
- The connections to other fields

Take notes on ideas for your own research.

### 8.7.5 Summary: Reading Purposes

| Purpose | Focus | Notes |
|---------|-------|-------|
| Literature review | Problem, method, results | Structured |
| Implementation | Method details, hyperparameters | Detailed |
| Critique | Assumptions, design, conclusions | Strengths/weaknesses |
| Inspiration | Open questions, future work | Ideas |

> **Exercise 8.7:** Choose three papers. Read each with a different purpose (literature review, implementation, critique). Compare your notes.

---

## 8.8 Writing a Paper Review

### 8.8.1 The Structure of a Review

A **paper review** is a structured critique of a paper. It typically has the following sections:

**Summary:** A brief summary of the paper's contributions.

**Strengths:** What the paper does well.

**Weaknesses:** What the paper does poorly.

**Questions:** Questions for the authors.

**Suggestions:** Suggestions for improvement.

**Recommendation:** Accept, reject, or revise.

### 8.8.2 How to Write a Good Review

**Be specific.** Instead of "the experiments are weak," say "the experiments do not compare with the state-of-the-art method X."

**Be constructive.** Instead of "this is wrong," say "this could be improved by..."

**Be fair.** Acknowledge the strengths as well as the weaknesses.

**Be clear.** Use plain language. Avoid jargon.

**Be concise.** A good review is 1–2 pages.

### 8.8.3 Worked Example 8.4: A Review of "Attention Is All You Need"

**Summary:** The paper proposes the Transformer, a new architecture for sequence transduction based entirely on attention mechanisms. It achieves state-of-the-art results on machine translation.

**Strengths:**
- The architecture is novel and elegant.
- The results are strong and reproducible.
- The paper is well-written and clear.
- The method is general and has been widely adopted.

**Weaknesses:**
- The quadratic complexity in sequence length limits scalability.
- The method requires large amounts of data.
- The paper does not address very long sequences.

**Questions:**
- How does the method perform on low-resource languages?
- Can the quadratic complexity be reduced without loss of performance?
- How does the method compare with RNNs on other tasks?

**Suggestions:**
- Evaluate on additional tasks (e.g., speech recognition, image captioning).
- Compare with RNNs on low-resource settings.
- Discuss the limitations more thoroughly.

**Recommendation:** Accept.

### 8.8.4 Summary: Writing a Review

| Section | Content |
|---------|---------|
| Summary | Brief overview |
| Strengths | What works |
| Weaknesses | What does not |
| Questions | For the authors |
| Suggestions | For improvement |
| Recommendation | Accept/reject/revise |

> **Exercise 8.8:** Write a review of a paper from your literature search. Follow the structure above. Share it with a classmate and discuss.

---

## 8.9 Peer Review

### 8.9.1 How Peer Review Works

**Peer review** is the process by which papers are evaluated by experts before publication. The typical workflow:

1. **Submission:** Authors submit a paper to a conference or journal.
2. **Assignment:** The editor assigns the paper to 2–4 reviewers.
3. **Review:** Reviewers read the paper and write reviews.
4. **Decision:** The editor makes a decision (accept, reject, revise).
5. **Rebuttal:** Authors respond to the reviews.
6. **Final decision:** The editor makes the final decision.

### 8.9.2 What Reviewers Look For

**Novelty:** Is the contribution new?

**Significance:** Is the contribution important?

**Correctness:** Are the methods and results correct?

**Clarity:** Is the paper well-written?

**Reproducibility:** Can the results be reproduced?

### 8.9.3 How to Respond to Reviews

**Be respectful.** Reviewers are volunteering their time.

**Be specific.** Address each point.

**Be concise.** Do not ramble.

**Be gracious.** Thank the reviewers for their feedback.

### 8.9.4 Summary: Peer Review

| Aspect | Description |
|--------|-------------|
| Purpose | Evaluate papers before publication |
| Reviewers | 2–4 experts |
| Criteria | Novelty, significance, correctness, clarity, reproducibility |
| Response | Respectful, specific, concise |

> **Exercise 8.9:** Simulate a peer review. Exchange papers with a classmate. Write a review of their paper. Discuss your reviews.

---

## 8.10 Research Application: Reading for Your Capstone

### 8.10.1 The Role of Reading in Your Capstone

For your capstone project, you will read dozens of papers. The skills in this module will help you:

- **Find the gap:** Identify what has not been done.
- **Position your contribution:** Show how your work differs from prior work.
- **Learn methodology:** Adopt best practices from prior work.
- **Avoid pitfalls:** Learn from the limitations of prior work.

### 8.10.2 A Reading Plan for Your Capstone

**Week 1:** Read 10 papers broadly to map the landscape.

**Week 2:** Read 5 papers deeply to identify the gap.

**Week 3:** Read 3 papers on methodology.

**Week 4:** Read 2 papers on evaluation.

**Week 5+:** Monitor new papers and read as needed.

### 8.10.3 Summary: Reading for Capstone

| Week | Focus | Papers |
|------|-------|--------|
| 1 | Landscape | 10 |
| 2 | Gap | 5 |
| 3 | Methodology | 3 |
| 4 | Evaluation | 2 |
| 5+ | Monitoring | As needed |

> **Exercise 8.10:** Create a reading plan for your capstone project. List the papers you will read and the order in which you will read them.

---

## 8.11 Module Summary

| Concept | Description | Tool |
|---------|-------------|------|
| Three-pass method | Skim, read, deep dive | Keshav (2007) |
| Pass 1 | Get the gist | 5–10 min |
| Pass 2 | Understand the content | 1 hour |
| Pass 3 | Deep understanding | 4–5 hours |
| Critical appraisal | Evaluate validity | Framework |
| Internal validity | Was the study designed correctly? | Check design |
| External validity | Do the results generalize? | Check datasets |
| Statistical validity | Are the results significant? | Check stats |
| Construct validity | Do the measurements measure what they claim? | Check metrics |
| Conclusion validity | Do the conclusions follow? | Check claims |
| Five questions | Problem, solution, evidence, limitations, significance | Checklist |
| Common pitfalls | Abstract-only, trusting, overgeneralizing | Avoid |
| Paper types | Empirical, theoretical, survey, position | Adapt reading |
| Reading purposes | Literature review, implementation, critique, inspiration | Adapt focus |
| Paper review | Summary, strengths, weaknesses, questions, suggestions | Structure |
| Peer review | Evaluate papers before publication | Process |

---

## 8.12 Capstone Thread: Reading Skills for Your Project

By the end of this module, you will be able to read a paper in three passes; identify its contributions, methods, and results; evaluate its internal and external validity; ask critical questions; extract information for your own work; and articulate why critical reading is a core research skill.

For your capstone project, you will use these skills to conduct a literature review, identify a gap, position your contribution, and write the related work section.

In Module 9, we will learn **how to reproduce a paper** — the process of reimplementing a method and verifying its results.

---

## 8.13 Additional Resources

### On Reading Papers

- Keshav, S. (2007). "How to Read a Paper." *ACM SIGCOMM Computer Communication Review*, 37(3), 83–84. [Link](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf)
- Rodriguez, J. (2023). "How to Read a Machine Learning Paper." *Towards Data Science*.

### On Critical Appraisal

- Critical Appraisal Skills Programme (CASP): https://casp-uk.net/
- "Critical Appraisal of a Research Paper" (Stanford): https://med.stanford.edu/

### On Peer Review

- "How to Write a Peer Review" (PLOS): https://plos.org/
- "Reviewing a Paper" (Nature): https://www.nature.com/

### On Specific Papers

- He, K., et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.
- Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
- Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." *NAACL*.

---

*End of Module 8*