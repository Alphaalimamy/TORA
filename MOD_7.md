# Module 7: Finding and Managing Scientific Literature

## TORA | CS 336: PyTorch for Research

**Instructor:** Alpha Alimamy Kamara

**Module Length:** 1 week (2 lectures + 1 discussion section)

**Prerequisites:** Module 1 completed; access to a university library account (for paywalled papers); a reference manager installed (Zotero, Mendeley, or Paperpile).

---

## 7.0 Module Overview

In Modules 2–6, we built the technical foundations of PyTorch research: tensors, autograd, models, training loops, and data pipelines. But technical skill alone does not make a researcher. A researcher must also know **what has already been done**, **what is currently being done**, and **what remains to be done**. This knowledge comes from the scientific literature.

This module is about the **literature review process**: how to find papers, how to organize them, how to stay current, and how to build a personal knowledge base that supports your research. It is a departure from the technical modules, but it is no less important. A researcher who cannot navigate the literature will reinvent wheels, miss critical prior work, and fail to position their contributions.

The pedagogical approach here is **process first, tools second**. We will begin with the research workflow, then introduce the specific tools and databases that support it. The goal is not to memorize a list of websites but to internalize a repeatable process for discovering and organizing knowledge.

By the end of this module, you will be able to search the major ML literature databases; use citation graphs to find related work; set up alerts for new papers; organize your references with a reference manager; build a personal knowledge base; and articulate why literature management is a core research skill.

The module is self-paced, but the exercises are best done in a single sitting so that you build a complete workflow.

---

## 7.1 The Role of Literature in Research

### 7.1.1 Why Read Papers?

Research is a conversation. Each paper is a contribution to an ongoing dialogue about a problem. To participate in that dialogue, you must first understand what has been said. This is why every research paper begins with a **literature review** — a summary of the relevant prior work.

Reading papers serves several purposes:

**Avoiding reinvention.** Many ideas have been tried before. Reading the literature prevents you from spending months on a problem that was solved a decade ago.

**Finding inspiration.** Papers often contain suggestions for future work, open questions, and limitations that you can address.

**Building context.** You need to know where your work fits in the broader landscape. What are the current state-of-the-art methods? What are the open problems?

**Learning methodology.** Papers describe experimental designs, evaluation metrics, and statistical tests. Reading them teaches you how to conduct your own experiments.

**Developing critical judgment.** Not all papers are equally good. Reading widely teaches you to distinguish rigorous work from sloppy work.

### 7.1.2 The Literature Review as a Process

A literature review is not a one-time event. It is an ongoing process that spans your entire research career. The process has several stages:

**Exploration.** When you begin a new project, you survey the landscape. You read broadly to understand the problem space.

**Deep dive.** Once you have identified a specific question, you read deeply on that question. You find the seminal papers, the state-of-the-art methods, and the most recent work.

**Monitoring.** After you have a working knowledge of the field, you monitor for new papers. This is a lifelong habit.

**Synthesis.** You organize what you have learned into a coherent narrative that motivates your own work.

### 7.1.3 The Three Questions Applied to Literature

Recall from Module 1 the three questions of research. For literature, the question is: *What has been done, what is currently being done, and what remains to be done?* The evidence is: *The papers themselves, their methods, and their results.* The meaning is: *How does this prior work inform your own research question?*

### 7.1.4 Summary: The Role of Literature

| Purpose | Description | Example |
|---------|-------------|---------|
| Avoiding reinvention | Learn what has been done | "This was tried in 2015" |
| Finding inspiration | Identify open questions | "Future work could..." |
| Building context | Understand the landscape | "SOTA is 95% accuracy" |
| Learning methodology | Study experimental design | "They used 5-fold CV" |
| Developing judgment | Distinguish good from bad | "This paper is rigorous" |

> **Exercise 7.1 (Pre-Class):** Write a one-paragraph answer to each of the following. Bring this to lecture.
>
> 1. What is a research question you are interested in?
> 2. What papers have you already read on this topic?
> 3. What gaps remain in the literature?

---

## 7.2 Where to Find Papers

### 7.2.1 arXiv

**arXiv** (https://arxiv.org/) is the primary preprint server for machine learning, computer science, physics, and mathematics. It hosts over 2 million papers, with new submissions daily.

The relevant categories for ML research are:
- **cs.LG** — Machine Learning
- **cs.CV** — Computer Vision
- **cs.CL** — Computation and Language (NLP)
- **cs.AI** — Artificial Intelligence
- **stat.ML** — Machine Learning (Statistics)

arXiv papers are preprints — they have not necessarily been peer-reviewed. This is both a feature and a bug. It means you get access to cutting-edge work quickly, but it also means you must be more critical.

**Searching arXiv:** The web interface supports keyword search, author search, and category browsing. The advanced search allows you to filter by date, category, and other fields.

**Worked example 7.1:** Search arXiv for papers on "vision transformers" published in the last year.

1. Go to https://arxiv.org/
2. Enter "vision transformers" in the search box
3. Select "Computer Vision and Pattern Recognition (cs.CV)" in the category filter
4. Sort by "Announcement date (newest first)"
5. Review the titles and abstracts

### 7.2.2 Papers With Code

**Papers With Code** (https://paperswithcode.com/) is a website that tracks state-of-the-art results across ML tasks. It links papers to their implementations and provides leaderboards.

The key features are:
- **SOTA tracking:** See the best-performing methods for each task
- **Code links:** Find implementations of papers
- **Datasets:** Browse datasets by task
- **Methods:** Explore methods by category

**Research relevance:** Papers With Code is the fastest way to find the current state of the art for a given task. When you begin a project, this is often the first place to look.

**Worked example 7.2:** Find the state of the art for image classification on ImageNet.

1. Go to https://paperswithcode.com/
2. Search for "ImageNet"
3. Click on the "Image Classification on ImageNet" task
4. Review the leaderboard and the associated papers

### 7.2.3 Semantic Scholar

**Semantic Scholar** (https://www.semanticscholar.org/) is an AI-powered academic search engine. It provides:
- **Citation graphs:** See which papers cite a given paper
- **Author profiles:** Track an author's publications
- **Recommendations:** Get paper recommendations based on your reading
- **TLDR summaries:** Short summaries of papers

**Research relevance:** Semantic Scholar is particularly useful for finding related work. If you find a paper that is central to your topic, you can use the citation graph to find papers that cite it (downstream work) and papers it cites (upstream work).

**Worked example 7.3:** Find papers that cite "Attention Is All You Need."

1. Go to https://www.semanticscholar.org/
2. Search for "Attention Is All You Need"
3. Click on the paper
4. Click on "Citations" to see papers that cite it
5. Filter by year, venue, or author

### 7.2.4 Google Scholar

**Google Scholar** (https://scholar.google.com/) is the most widely used academic search engine. It indexes papers, theses, books, and conference proceedings.

Key features:
- **Broad search:** Covers all disciplines
- **Citation tracking:** See who cites a paper
- **Alerts:** Get notified when new papers match your query
- **Library links:** Access paywalled papers through your institution

**Research relevance:** Google Scholar is a good starting point for broad searches. The alerts feature is essential for staying current.

### 7.2.5 Conference Proceedings

The major ML conferences are the primary venues for peer-reviewed research. The most important are:

| Conference | Focus | Website |
|------------|-------|---------|
| NeurIPS | General ML | https://neurips.cc/ |
| ICML | General ML | https://icml.cc/ |
| ICLR | General ML | https://iclr.cc/ |
| CVPR | Computer Vision | https://cvpr.thecvf.com/ |
| ICCV | Computer Vision | https://iccv.thecvf.com/ |
| ECCV | Computer Vision | https://eccv.ecva.net/ |
| ACL | NLP | https://aclanthology.org/ |
| EMNLP | NLP | https://aclanthology.org/ |
| NAACL | NLP | https://aclanthology.org/ |
| AAAI | General AI | https://aaai.org/ |

**Research relevance:** Conference proceedings contain peer-reviewed papers, which are generally more reliable than preprints. When you cite a paper, prefer the peer-reviewed version if available.

### 7.2.6 Awesome Lists

**Awesome lists** are curated collections of papers, code, and resources on specific topics. They are hosted on GitHub and maintained by the community.

Examples:
- **Awesome Deep Learning:** https://github.com/ChristosChristofidis/awesome-deep-learning
- **Awesome Computer Vision:** https://github.com/jbhuang0604/awesome-computer-vision
- **Awesome NLP:** https://github.com/keon/awesome-nlp
- **Awesome Transformers:** https://github.com/abidelgad/awesome-transformers

**Research relevance:** Awesome lists are a good starting point for a new topic. They provide a curated set of papers that you can use as a foundation.

### 7.2.7 Industry Lab Publications

Industry labs publish technical reports and blog posts that often describe state-of-the-art methods before they appear in conferences. The most important are:

| Lab | Website |
|-----|---------|
| OpenAI | https://openai.com/research/ |
| DeepMind | https://deepmind.google/research/ |
| Meta AI | https://ai.meta.com/research/ |
| Google Research | https://research.google/ |
| Microsoft Research | https://www.microsoft.com/en-us/research/ |
| Anthropic | https://www.anthropic.com/research |

**Research relevance:** Industry lab publications are often at the cutting edge, but they are not peer-reviewed. Treat them as preprints.

### 7.2.8 Summary: Where to Find Papers

| Source | Type | Best For |
|--------|------|----------|
| arXiv | Preprints | Cutting-edge work |
| Papers With Code | Aggregator | SOTA, code |
| Semantic Scholar | Search engine | Citation graphs |
| Google Scholar | Search engine | Broad search, alerts |
| Conference proceedings | Peer-reviewed | Reliable results |
| Awesome lists | Curated | Starting a new topic |
| Industry labs | Reports | Cutting-edge methods |

> **Exercise 7.2:** For a topic of your choice, find at least three papers from each of the following sources:
>
> 1. arXiv
> 2. Papers With Code
> 3. Semantic Scholar
> 4. A conference proceedings (NeurIPS, ICML, ICLR, CVPR, ACL)
>
> For each paper, record the title, authors, year, and a one-sentence summary.

---

## 7.3 How to Search Effectively

### 7.3.1 Keyword Selection

Effective search begins with effective keywords. The keywords you use determine what you find. Poor keywords return irrelevant results; good keywords return the papers you need.

**Strategies for keyword selection:**

**Start broad, then narrow.** Begin with a general term (e.g., "image classification") and add modifiers (e.g., "few-shot image classification," "self-supervised image classification").

**Use synonyms.** Different authors use different terms for the same concept. Try "neural network," "deep learning," "deep neural network," and "multi-layer perceptron."

**Use Boolean operators.** Most search engines support AND, OR, and NOT. For example, "transformer AND vision" finds papers that mention both; "transformer OR attention" finds papers that mention either.

**Use field-specific terms.** Each field has its own vocabulary. In NLP, terms like "tokenization," "embedding," and "attention" are common. In vision, terms like "convolution," "pooling," and "receptive field" are common.

### 7.3.2 The Query Formulation Process

**Worked example 7.4:** Formulate a query for "self-supervised learning for medical image segmentation."

1. **Identify the core concepts:** self-supervised learning, medical imaging, image segmentation
2. **List synonyms for each:**
   - Self-supervised: "self-supervised," "contrastive learning," "pretext task"
   - Medical imaging: "medical image," "clinical image," "radiology," "pathology"
   - Segmentation: "segmentation," "pixel-wise classification," "dense prediction"
3. **Combine with Boolean operators:**
   - `("self-supervised" OR "contrastive learning") AND ("medical image" OR "clinical image") AND ("segmentation" OR "dense prediction")`
4. **Refine based on results:** If too many results, add more specific terms. If too few, remove terms.

### 7.3.3 Citation Chaining

**Citation chaining** is the process of following citations to find related work. There are two directions:

**Backward chaining:** Look at the references of a paper to find the work it builds on.

**Forward chaining:** Look at the papers that cite a paper to find the work that builds on it.

**Worked example 7.5:** Perform citation chaining on "Attention Is All You Need."

1. **Backward chaining:** The paper cites "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al., 2015) and "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997).
2. **Forward chaining:** Papers that cite "Attention Is All You Need" include "BERT" (Devlin et al., 2019), "GPT-3" (Brown et al., 2020), and "Vision Transformer" (Dosovitskiy et al., 2021).

### 7.3.4 Systematic Search

For a formal literature review, a **systematic search** is preferred. This involves:

1. **Defining the research question** (e.g., using PICO: Population, Intervention, Comparison, Outcome)
2. **Defining inclusion and exclusion criteria**
3. **Searching multiple databases**
4. **Screening results** (title/abstract, then full text)
5. **Extracting data** from included papers
6. **Synthesizing findings**

**Research relevance:** Systematic reviews are the gold standard in evidence-based medicine. In ML, they are less common but increasingly used. The PRISMA guidelines provide a framework for reporting systematic reviews.

### 7.3.5 Summary: Search Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| Keyword search | Use terms in search box | General exploration |
| Boolean operators | AND, OR, NOT | Refining queries |
| Citation chaining | Follow references | Finding related work |
| Systematic search | Structured, comprehensive | Formal reviews |
| Alerts | Notifications for new papers | Staying current |

> **Exercise 7.3:** Formulate a search query for a topic of your choice using the process above. Search at least three databases. Record the number of results and the top 5 papers from each.

---

## 7.4 Staying Current

### 7.4.1 The Problem of Scale

Machine learning is a fast-moving field. Thousands of papers are published every month. Keeping up with all of them is impossible. The challenge is to filter the signal from the noise.

### 7.4.2 Alerts

**Google Scholar alerts:** Create an alert for a specific query. Google Scholar will email you when new papers match.

**arXiv alerts:** Subscribe to daily or weekly digests for specific categories.

**Semantic Scholar alerts:** Follow authors, topics, or papers.

**Twitter/X:** Follow researchers and labs. Many post their papers when they are released.

### 7.4.3 Curated Newsletters

Several newsletters curate the most important papers each week:

- **The Batch** (DeepLearning.AI): Weekly AI news
- **Import AI** (Jack Clark): Weekly AI research summary
- **ML Street Talk:** Interviews with researchers
- **The Gradient:** Long-form articles on ML research

### 7.4.4 Reading Groups

A **reading group** is a regular meeting where a group of researchers discusses a paper. Reading groups are a powerful way to stay current and to develop critical reading skills.

**How to run a reading group:**
1. Choose a paper one week in advance.
2. Assign a presenter who prepares slides.
3. Each participant reads the paper and prepares questions.
4. The presenter summarizes the paper and leads discussion.
5. Record notes and action items.

### 7.4.5 Conferences and Workshops

Attending conferences is an immersive way to stay current. You hear talks, meet researchers, and see posters. Even if you cannot attend in person, most conferences post their proceedings online.

### 7.4.6 Summary: Staying Current

| Method | Frequency | Effort | Signal |
|--------|-----------|--------|--------|
| Alerts | Continuous | Low | Medium |
| Newsletters | Weekly | Low | High |
| Reading groups | Weekly | Medium | High |
| Conferences | Annual | High | High |
| Twitter/X | Continuous | Low | Variable |

> **Exercise 7.4:** Set up at least two alerts (Google Scholar, arXiv, or Semantic Scholar) for topics relevant to your research. Subscribe to at least one newsletter. Join or form a reading group.

---

## 7.5 Reference Management

### 7.5.1 Why Reference Management?

A researcher reads hundreds of papers over a career. Without a system for organizing them, you will forget what you have read, lose track of citations, and waste time re-finding papers.

A **reference manager** is software that stores, organizes, and formats your references. It integrates with your word processor to insert citations and generate bibliographies.

### 7.5.2 Zotero

**Zotero** (https://www.zotero.org/) is a free, open-source reference manager. It is the most popular choice in academia.

Key features:
- **Browser extension:** Save papers with one click
- **PDF storage:** Attach PDFs to references
- **Collections:** Organize references into folders
- **Tags:** Add custom tags for cross-cutting themes
- **Citation styles:** Thousands of styles (APA, MLA, Chicago, BibTeX)
- **Word/LibreOffice integration:** Insert citations as you write
- **Sync:** Sync across devices
- **Groups:** Share collections with collaborators

### 7.5.3 Mendeley

**Mendeley** (https://www.mendeley.com/) is a free reference manager owned by Elsevier. It has similar features to Zotero but is less open.

### 7.5.4 Paperpile

**Paperpile** (https://paperpile.com/) is a paid reference manager integrated with Google Docs. It is popular among researchers who write in Google Docs.

### 7.5.5 BibTeX

**BibTeX** is a file format for storing references. It is used by LaTeX and is supported by all major reference managers.

A BibTeX entry looks like:

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

### 7.5.6 Summary: Reference Managers

| Tool | Cost | Platform | Best For |
|------|------|----------|----------|
| Zotero | Free | Desktop, web | Most researchers |
| Mendeley | Free | Desktop, web | Elsevier users |
| Paperpile | Paid | Web | Google Docs users |
| BibTeX | Free | Text | LaTeX users |

> **Exercise 7.5:** Install Zotero (or another reference manager). Import at least 10 papers from your literature search. Organize them into collections. Add tags for cross-cutting themes. Generate a bibliography in BibTeX format.

---

## 7.6 Building a Personal Knowledge Base

### 7.6.1 Why a Knowledge Base?

Reading papers is not enough. You must **retain** what you read and **connect** it to your own work. A **personal knowledge base** is a system for capturing, organizing, and retrieving what you learn.

### 7.6.2 Note-Taking Systems

**Zettelkasten:** A method where each note is a single idea, linked to other notes. The goal is to build a network of interconnected ideas.

**Literature notes:** A summary of a paper in your own words. Include the key claims, methods, results, and limitations.

**Permanent notes:** A synthesis of multiple papers on a theme. These are your own ideas, informed by the literature.

**Tools:**
- **Obsidian:** A markdown-based note-taking app with bidirectional links
- **Roam Research:** A networked note-taking app
- **Notion:** A general-purpose workspace
- **Logseq:** An open-source alternative to Roam

### 7.6.3 Paper Summaries

For each paper you read, write a structured summary:

**Citation:** Full citation in your preferred format.

**Problem:** What problem does the paper address?

**Method:** What is the proposed method?

**Results:** What are the key results?

**Limitations:** What are the limitations?

**Relevance:** How does this relate to your work?

**Quotes:** Key quotes with page numbers.

**Worked example 7.6:** Write a structured summary for "Attention Is All You Need."

**Citation:** Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.

**Problem:** Recurrent and convolutional neural networks for sequence transduction are slow to train and have difficulty with long-range dependencies.

**Method:** The Transformer, a novel architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

**Results:** State-of-the-art BLEU scores on WMT 2014 English-to-German and English-to-French translation tasks. Trained in a fraction of the time of previous models.

**Limitations:** Quadratic complexity in sequence length; requires large amounts of data; no explicit positional encoding in the original formulation.

**Relevance:** The Transformer is the foundation of modern NLP and increasingly vision.

**Quotes:** "The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution."

### 7.6.4 Synthesis Notes

Synthesis notes combine multiple papers on a theme. For example, a synthesis note on "self-supervised learning" might summarize the key ideas from SimCLR, MoCo, BYOL, and SimSiam, and identify common themes and differences.

### 7.6.5 Summary: Knowledge Base

| Note Type | Purpose | Length |
|-----------|---------|--------|
| Literature note | Summary of one paper | 1 page |
| Permanent note | Synthesis of multiple papers | 1–2 pages |
| Project note | Ideas for a specific project | Variable |
| Reading log | List of papers read | 1 line each |

> **Exercise 7.6:** Write structured summaries for five papers from your literature search. Write one synthesis note that combines them.

---

## 7.7 Reading Groups and Collaboration

### 7.7.1 Why Reading Groups?

Reading groups are a powerful way to:
- **Stay current:** Discuss recent papers
- **Develop critical reading:** Practice critiquing papers
- **Learn from others:** Hear different perspectives
- **Build community:** Meet other researchers
- **Prepare for conferences:** Discuss papers before attending

### 7.7.2 How to Run a Reading Group

**Format:** Weekly or biweekly, 1 hour.

**Structure:**
1. **Presenter:** One person prepares a 20-minute summary of the paper.
2. **Discussion:** 40 minutes of Q&A and critique.
3. **Action items:** Notes, follow-up papers, next week's paper.

**Roles:**
- **Organizer:** Schedules meetings, chooses papers.
- **Presenter:** Prepares the summary.
- **Scribe:** Takes notes.
- **Participants:** Read the paper, prepare questions.

### 7.7.3 Tips for Effective Discussion

**Ask clarifying questions.** "What does the author mean by 'self-attention'?"

**Challenge assumptions.** "Why did they choose this evaluation metric?"

**Connect to prior work.** "How does this relate to the paper we read last week?"

**Identify limitations.** "What are the failure modes of this method?"

**Suggest extensions.** "Could this be applied to a different domain?"

### 7.7.4 Summary: Reading Groups

| Aspect | Recommendation |
|--------|----------------|
| Frequency | Weekly or biweekly |
| Duration | 1 hour |
| Presenter | Rotating |
| Paper | One per session |
| Preparation | Read before meeting |
| Notes | Shared document |

> **Exercise 7.7:** Form a reading group with 3–5 classmates. Choose a paper from your literature search. Prepare a 20-minute summary. Lead a discussion. Take notes.

---

## 7.8 Worked Example: A Complete Literature Review

Let us work through a complete literature review for a hypothetical project on "self-supervised learning for medical image segmentation."

### 7.8.1 Define the Research Question

**Question:** Can self-supervised pre-training improve the performance of medical image segmentation models, particularly in low-data regimes?

### 7.8.2 Search the Literature

**Databases:** arXiv, Papers With Code, Semantic Scholar, Google Scholar.

**Query:** `("self-supervised" OR "contrastive learning") AND ("medical image" OR "clinical image") AND ("segmentation")`

**Results:** 127 papers (after deduplication).

### 7.8.3 Screen the Results

**Title/abstract screening:** 127 → 42 papers.

**Full-text screening:** 42 → 18 papers.

**Inclusion criteria:** Papers that (1) propose or use self-supervised learning, (2) apply it to medical images, (3) evaluate on segmentation tasks.

### 7.8.4 Extract Data

For each of the 18 papers, extract:
- Citation
- Method (self-supervised objective)
- Dataset
- Segmentation task
- Results
- Limitations

### 7.8.5 Synthesize

**Themes:**
1. **Contrastive learning** (SimCLR, MoCo, BYOL) is the dominant approach.
2. **Masked image modeling** (MAE, BEiT) is emerging.
3. **Domain-specific pre-training** outperforms ImageNet pre-training.
4. **Low-data regimes** benefit most from self-supervised pre-training.

**Gaps:**
1. Most papers focus on classification, not segmentation.
2. Few papers address multi-modal medical images (e.g., MRI + CT).
3. Evaluation protocols vary widely, making comparison difficult.

### 7.8.6 Write the Review

A literature review section in a paper typically has this structure:

**Introduction:** Why is this topic important?

**Background:** What are the key concepts?

**Related work:** What has been done?

**Gaps:** What remains to be done?

**Conclusion:** How does your work address the gaps?

### 7.8.7 Reflection

This literature review took approximately 20 hours of work: 4 hours searching, 8 hours reading, 4 hours extracting, 2 hours synthesizing, 2 hours writing. The result is a comprehensive understanding of the state of the art and a clear identification of the gap your research will address.

> **Exercise 7.8:** Conduct a mini literature review (10 papers) on a topic of your choice. Follow the process above. Write a 2-page summary.

---

## 7.9 Common Pitfalls and Best Practices

### 7.9.1 Pitfall: Confirmation Bias

**Symptom:** You only find papers that support your hypothesis.

**Fix:** Actively search for papers that contradict your view. Read them carefully.

### 7.9.2 Pitfall: Citation Padding

**Symptom:** You cite papers you have not read.

**Fix:** Only cite papers you have read. If you must cite a paper you have not read, read at least the abstract and introduction.

### 7.9.3 Pitfall: Over-reliance on Preprints

**Symptom:** You cite arXiv preprints instead of peer-reviewed versions.

**Fix:** Prefer peer-reviewed versions. If a paper has been published, cite the published version.

### 7.9.4 Pitfall: Ignoring Older Work

**Symptom:** You only cite papers from the last 2 years.

**Fix:** Seminal papers remain relevant. Cite them.

### 7.9.5 Best Practices

- **Read the abstract first.** Decide if the paper is worth your time.
- **Take notes as you read.** Do not rely on memory.
- **Summarize in your own words.** This forces understanding.
- **Connect to your work.** How does this paper relate to your research?
- **Cite consistently.** Use a reference manager.
- **Keep a reading log.** Track what you have read.

### 7.9.6 Summary: Pitfalls and Best Practices

| Pitfall | Fix |
|---------|-----|
| Confirmation bias | Seek contradictory evidence |
| Citation padding | Only cite what you have read |
| Over-reliance on preprints | Prefer peer-reviewed versions |
| Ignoring older work | Cite seminal papers |
| Poor note-taking | Use a structured system |
| Inconsistent citations | Use a reference manager |

> **Exercise 7.9:** Review your literature search for the pitfalls above. Correct any issues. Add at least one older seminal paper and one paper that contradicts your hypothesis.

---

## 7.10 Research Application: Positioning Your Contribution

### 7.10.1 The Related Work Section

Every research paper has a **related work** section that positions the contribution in the context of prior work. Writing this section requires a thorough understanding of the literature.

A good related work section:
- **Summarizes** the most relevant prior work
- **Groups** papers by theme
- **Identifies** gaps
- **Positions** your contribution

### 7.10.2 The Contribution Statement

After reviewing the literature, you must articulate your contribution. A contribution statement typically has three parts:

**What:** What did you do?

**Why:** Why does it matter?

**How:** How is it different from prior work?

**Worked example 7.7:** Contribution statement for a paper on self-supervised medical image segmentation.

"We propose a novel self-supervised pre-training method for medical image segmentation that combines contrastive learning with masked image modeling. Unlike prior work, which focuses on classification, our method is designed specifically for dense prediction tasks. We show that our method outperforms ImageNet pre-training and state-of-the-art self-supervised methods on three medical imaging datasets, particularly in low-data regimes."

### 7.10.3 Summary: Positioning

| Element | Description |
|---------|-------------|
| Related work | Summarize prior work |
| Gap | Identify what is missing |
| Contribution | State what you did |
| Significance | Explain why it matters |
| Differentiation | Explain how it differs |

> **Exercise 7.10:** Write a contribution statement for your capstone project. Include what, why, and how.

---

## 7.11 Module Summary

| Concept | Description | Tool |
|---------|-------------|------|
| arXiv | Preprint server | https://arxiv.org/ |
| Papers With Code | SOTA tracking | https://paperswithcode.com/ |
| Semantic Scholar | Citation graphs | https://www.semanticscholar.org/ |
| Google Scholar | Broad search, alerts | https://scholar.google.com/ |
| Conference proceedings | Peer-reviewed papers | NeurIPS, ICML, ICLR, CVPR, ACL |
| Awesome lists | Curated collections | GitHub |
| Industry labs | Technical reports | OpenAI, DeepMind, Meta AI |
| Alerts | New paper notifications | Google Scholar, arXiv |
| Reference manager | Organize citations | Zotero, Mendeley, Paperpile |
| Knowledge base | Notes and synthesis | Obsidian, Roam, Notion |
| Reading group | Discuss papers | Weekly meetings |
| Literature review | Synthesize prior work | Systematic process |
| Related work | Position your contribution | Paper section |

---

## 7.12 Capstone Thread: Literature Skills for Your Project

By the end of this module, you should be able to search the major ML literature databases; use citation graphs to find related work; set up alerts for new papers; organize your references with a reference manager; build a personal knowledge base; and articulate why literature management is a core research skill.

For your capstone project, you will need these skills to conduct a literature review, identify a gap, position your contribution, and write the related work section.

In Module 8, we will learn **how to read a scientific paper** — the skills of critical appraisal, three-pass reading, and extracting the key information from a paper.

---

## 7.13 Additional Resources

### On Literature Search

- Stanford Libraries: https://library.stanford.edu/
- "How to Conduct a Literature Review" (UNC Writing Center): https://writingcenter.unc.edu/tips-and-tools/literature-reviews/
- "Systematic Reviews" (PRISMA): http://www.prisma-statement.org/

### On Reference Management

- Zotero: https://www.zotero.org/
- Zotero documentation: https://www.zotero.org/support/
- Mendeley: https://www.mendeley.com/
- Paperpile: https://paperpile.com/

### On Knowledge Management

- "How to Take Smart Notes" by Sönke Ahrens
- Obsidian: https://obsidian.md/
- Roam Research: https://roamresearch.com/
- Notion: https://www.notion.so/

### On Reading Groups

- "How to Run a Reading Group" (MIT): https://mitcommlab.mit.edu/
- "Running a Paper Reading Group" (Stanford): https://cs.stanford.edu/

### On Writing

- "Writing a Research Paper" (Stanford): https://cs.stanford.edu/
- "How to Write a Great Research Paper" by Simon Peyton Jones: https://www.microsoft.com/en-us/research/

---

*End of Module 7*