# BBC Igbo–Pidgin Gold-Standard NLP Corpus (Sample)

<div align="center">

![Version](https://img.shields.io/badge/version-1.0-blue.svg)
![Type](https://img.shields.io/badge/type-Sample%20Dataset-purple.svg)
![License](https://img.shields.io/badge/license-CC--BY--4.0-green.svg)
![Languages](https://img.shields.io/badge/languages-Igbo%20%7C%20Pidgin-orange.svg)
![Samples](https://img.shields.io/badge/samples-217-brightgreen.svg)

**Sample dataset: High-quality annotated data for Nigerian Igbo and Pidgin English NLP**

[🤗 Hugging Face](https://huggingface.co/datasets/Bytte-AI/BBC_Igbo-Pidgin_Gold-Standard_NLP_Corpus) • [📊 Figshare](https://figshare.com/articles/dataset/BBC_Igbo_Pidgin_Gold-Standard_NLP_Corpus/31249567) • [🌐 Website](https://www.bytte.xyz/) • [📧 Contact](mailto:contact@bytteai.xyz)

</div>

---

## 📋 Overview

The **BBC Igbo–Pidgin Gold-Standard NLP Corpus (Sample)** is a meticulously curated collection of professionally annotated text data designed to advance natural language processing for Nigerian languages. Created by **Bytte AI**, this sample corpus addresses the critical scarcity of high-quality linguistic resources for African low-resource languages.

> **📌 Sample Dataset Notice:** This is a **sample dataset** representing a curated subset of annotated documents. It demonstrates the annotation methodology, quality standards, and multi-task capabilities of the full corpus. Ideal for prototyping, fine-tuning, and quality benchmarking.

### 🎯 Key Features

- **217 annotated documents** across Igbo and Pidgin English (sample size)
- **4 complementary datasets** covering multiple NLP tasks
- **4,851 named entities** with 7-way classification
- **Human-in-the-loop annotation** using Label Studio
- **News domain coverage** from authoritative BBC sources
- **Multi-task learning ready** with aligned document IDs

### 📊 Quick Stats

| Language | Samples | Avg. Words/Doc | Tasks |
|----------|---------|----------------|-------|
| **Igbo** | 63 | 494 | Intent, Sentiment, Segmentation |
| **Pidgin** | 91 | 805 | Intent, Sentiment, NER, Language Quality |

---

## 🗂️ Dataset Composition

### 1️⃣ **BBC Igbo IQS** (Intent, Quality, Sentiment)
```
📄 bbc_igbo_IQS.csv | 219 KB | 63 samples
```
- **Tasks:** Intent classification, tone/sentiment analysis
- **Labels:** 4 intent types, 4 tone types
- **Format:** CSV with metadata and full article text
- **Use Case:** Document-level classification tasks

### 2️⃣ **BBC Pidgin IQS** (Intent, Quality, Sentiment)
```
📄 bbc_pidgin_IQS.csv | 432 KB | 91 samples
```
- **Tasks:** Intent classification, tone/sentiment analysis, language quality assessment
- **Labels:** 4 intent types, 4 tone types, 2 language quality types
- **Format:** CSV with metadata and full article text
- **Special Feature:** Language quality dimension captures Pidgin-English continuum

### 3️⃣ **BBC Igbo Sentence Segmentation**
```
📄 bbc_igbo_sentence_segmentation.json | 482 KB | 63 samples
```
- **Task:** Sentence boundary detection
- **Coverage:** 98.41% of documents (62/63)
- **Format:** Label Studio JSON export
- **Use Case:** Sentence tokenization, text preprocessing

### 4️⃣ **BBC Pidgin NER** (Named Entity Recognition)
```
📄 bbc_pidgin_NER.json | 1.4 MB | 91 samples
```
- **Task:** Named entity recognition and classification
- **Entities:** 4,851 annotations across 7 types
- **Entity Types:** PERSON, LOCATION, DATE, ORGANIZATION, EVENT, PRODUCT, MONEY
- **Format:** Label Studio JSON export with character offsets
- **Density:** 53.31 entities per document on average

---

## 🏷️ Label Taxonomy

### Intent Classification (5 classes)

| Label | Igbo | Pidgin | Description |
|-------|------|--------|-------------|
| **news-reporting** | 38 | 65 | Objective reporting of current events |
| **human-interest** | 22 | 12 | Personal stories and community narratives |
| **analysis** | 2 | 6 | In-depth examination of trends |
| **opinion** | 1 | 0 | Editorial viewpoints |
| **breaking-news** | 0 | 8 | Urgent, time-sensitive updates |

### Tone/Sentiment (4 classes)

| Label | Igbo | Pidgin | Description |
|-------|------|--------|-------------|
| **neutral** | 35 | 36 | Balanced, objective presentation |
| **positive** | 19 | 14 | Optimistic, hopeful framing |
| **negative** | 5 | 28 | Pessimistic, concerning framing |
| **critical** | 4 | 13 | Analytical questioning or criticism |

### Language Quality (Pidgin Only - 2 classes)

| Label | Count | Description |
|-------|-------|-------------|
| **mixed-pidgin-english** | 88 | Pidgin with English lexical items |
| **pure-pidgin** | 3 | Predominantly Pidgin vocabulary |

### Named Entities (Pidgin - 7 classes)

| Entity Type | Count | % | Examples |
|-------------|-------|---|----------|
| **PERSON** | 1,681 | 34.6% | Vladimir Putin, Donald Trump |
| **LOCATION** | 1,427 | 29.4% | Ukraine, Moscow, Nigeria |
| **DATE** | 603 | 12.4% | February 2022, Thursday |
| **ORGANIZATION** | 593 | 12.2% | BBC, World Economic Forum |
| **EVENT** | 244 | 5.0% | Russian invasion, tok-tok |
| **PRODUCT** | 240 | 4.9% | Air Force One |
| **MONEY** | 63 | 1.3% | Financial amounts |

---

## 🔬 Quality Metrics

### Annotation Quality

| Metric | Igbo IQS | Pidgin IQS | Igbo Seg | Pidgin NER |
|--------|----------|------------|----------|------------|
| **Annotators** | 1 | 2 | 1 | 1 |
| **Avg Lead Time** | 291.09s | 127.07s | 25.50s | 77.69s |
| **Median Lead Time** | 23.75s | 22.54s | 5.51s | 5.81s |

*High lead times indicate thoughtful, deliberate annotation rather than rushed work.*

### Class Balance (Shannon Entropy in bits)

| Dataset | Dimension | Entropy | Interpretation |
|---------|-----------|---------|----------------|
| Igbo IQS | Intent | 1.22 | Moderate imbalance |
| Igbo IQS | Tone | 1.54 | Good balance |
| Pidgin IQS | Intent | 1.30 | Moderate imbalance |
| Pidgin IQS | Tone | 1.87 | High balance ✅ |
| Pidgin IQS | Language | 0.21 | Severe imbalance* |
| Pidgin NER | Entities | 2.31 | Good balance ✅ |

*\*Reflects authentic Pidgin usage (naturally mixed with English)*

### Label Distribution Characteristics

| Dataset | Long-tail 80% Coverage | Variance | Avg Labels/Item |
|---------|------------------------|----------|-----------------|
| Igbo IQS (Intent) | 1/4 classes | 313.58 | 2.0 |
| Igbo IQS (Tone) | 1/4 classes | 211.58 | 2.0 |
| Pidgin IQS (Intent) | 1/4 classes | 799.58 | 3.0 |
| Pidgin IQS (Tone) | 2/4 classes | 124.92 | 3.0 |
| Pidgin NER (Entities) | 4/7 types | 333,815.71 | 53.31 |

---

## 🚀 Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/Bytte-AI/BBC_Igbo-Pidgin_Gold-Standard_NLP_Corpus.git
cd BBC_Igbo-Pidgin_Gold-Standard_NLP_Corpus

# Install dependencies
pip install pandas datasets
```

### Quick Load (Hugging Face)

```python
from datasets import load_dataset

# Load from Hugging Face Hub
dataset = load_dataset("Bytte-AI/BBC_Igbo-Pidgin_Gold-Standard_NLP_Corpus")

# Access individual splits
igbo_iqs = dataset['igbo_iqs']
pidgin_iqs = dataset['pidgin_iqs']
igbo_segmentation = dataset['igbo_segmentation']
pidgin_ner = dataset['pidgin_ner']
```

### Load Locally (CSV/JSON)

```python
import pandas as pd
import json

# Load IQS datasets (CSV)
igbo_iqs = pd.read_csv('bbc_igbo_IQS.csv')
pidgin_iqs = pd.read_csv('bbc_pidgin_IQS.csv')

# Load sentence segmentation (JSON)
with open('bbc_igbo_sentence_segmentation.json', 'r', encoding='utf-8') as f:
    igbo_seg = json.load(f)

# Load NER annotations (JSON)
with open('bbc_pidgin_NER.json', 'r', encoding='utf-8') as f:
    pidgin_ner = json.load(f)

print(f"Igbo IQS samples: {len(igbo_iqs)}")
print(f"Pidgin NER samples: {len(pidgin_ner)}")
```

### Example: Intent Classification

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('bbc_pidgin_IQS.csv')

# Prepare features and labels
X = df['raw_text']
y = df['intent']

# Split data (stratified to maintain class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train your model
# ... (use transformers, sklearn, etc.)
```

### Example: Named Entity Recognition

```python
import json

# Load NER data
with open('bbc_pidgin_NER.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract entities from first annotated document
sample = data[0]
text = sample['data']['raw_text']
entities = []

if sample['annotations']:
    for annotation in sample['annotations']:
        for entity in annotation['result']:
            entities.append({
                'text': entity['value']['text'],
                'label': entity['value']['labels'][0],
                'start': entity['value']['start'],
                'end': entity['value']['end']
            })

print(f"Text: {text[:100]}...")
print(f"Found {len(entities)} entities:")
for ent in entities[:5]:
    print(f"  - {ent['text']} ({ent['label']})")
```

---

## 💡 Use Cases

### ✅ Recommended Applications

1. **Benchmarking NLP Models**
   - Evaluate pre-trained multilingual models (mBERT, XLM-R, AfroXLMR)
   - Compare performance across African languages
   - Establish baselines for future research

2. **Model Training & Fine-tuning**
   - Fine-tune language models on Nigerian languages
   - Train specialized NER systems for West African contexts
   - Develop intent and sentiment classifiers for news domains

3. **Low-Resource NLP Research**
   - Study transfer learning to African languages
   - Investigate code-switching in Pidgin English
   - Analyze morphological processing for Igbo

4. **Language Technology Development**
   - Build content moderation tools for Nigerian platforms
   - Create information retrieval systems
   - Support machine translation research

### ❌ Out-of-Scope Uses

- General-purpose models without domain adaptation (news-specific)
- Applications requiring perfectly balanced datasets
- Formal/literary language processing (colloquial news style)
- Production deployment without validation on target domain

---

## ⚠️ Limitations

### Known Constraints

1. **Sample Dataset - Limited Scale**
   - This is a **sample dataset** (217 documents)
   - Not representative of full corpus size
   - Best for: prototyping, fine-tuning, quality demonstration
   - Not recommended for: training large models from scratch

2. **Limited Training Scale**
   - 63 Igbo samples, 91 Pidgin samples
   - Best used for fine-tuning, not training from scratch
   - Combine with other datasets for larger experiments

2. **Domain Specificity**
   - Exclusively news content from BBC
   - May not generalize to social media, technical, or conversational domains

3. **Class Imbalance**
   - News-reporting intent dominates (60-71%)
   - Consider class weighting during training
   - Apply oversampling for minority classes

4. **Single Source**
   - All content from BBC (editorial perspective)
   - Represents one variety of Igbo and Pidgin
   - Combine with diverse sources when possible

5. **Temporal Coverage**
   - Articles up to January 2026
   - May not reflect newer linguistic trends

### Recommended Mitigations

- **Data Augmentation:** Use back-translation, paraphrasing
- **Class Weighting:** Apply inverse frequency weights
- **Ensemble Methods:** Combine with other African language corpora
- **Domain Adaptation:** Fine-tune on target domain after pre-training
- **Stratified Splitting:** Maintain class distribution in train/dev/test

---

## 📖 Data Format Specifications

### IQS CSV Format

| Column | Type | Description |
|--------|------|-------------|
| `annotation_id` | int | Unique annotation identifier |
| `annotator` | str | Annotator ID |
| `created_at` | datetime | Annotation timestamp |
| `id` | str | Document ID (e.g., `bbc_igbo_0001`) |
| `intent` | str | Intent label |
| `language` | str | Language code (`igbo` or `pidgin`) |
| `language_quality` | str | Language quality (Pidgin only) |
| `lead_time` | float | Annotation time in seconds |
| `raw_text` | str | Full article text |
| `source` | str | Source platform (`bbc`) |
| `title` | str | Article headline |
| `tone` | str | Sentiment/tone label |
| `updated_at` | datetime | Last update timestamp |
| `url` | str | Original BBC article URL |

### Label Studio JSON Format

The sentence segmentation and NER datasets use Label Studio's export format:

```json
{
  "id": 405,
  "data": {
    "id": "bbc_pidgin_0001",
    "raw_text": "Full article text...",
    "language": "pidgin",
    "source": "bbc",
    "url": "https://www.bbc.com/pidgin/...",
    "title": "Article headline"
  },
  "annotations": [{
    "id": 171,
    "completed_by": 3,
    "result": [{
      "value": {
        "start": 454,
        "end": 496,
        "text": "di status of Ukraine eastern Donbas region",
        "labels": ["EVENT"]
      },
      "id": "oJCzY13Q8I",
      "from_name": "entities",
      "to_name": "article",
      "type": "labels"
    }],
    "lead_time": 77.69
  }]
}
```

---

## 📚 Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{bytte_ai_bbc_igbo_pidgin_2026,
  author    = {Bytte AI},
  title     = {BBC Igbo–Pidgin Gold-Standard NLP Corpus (Sample)},
  year      = {2026},
  version   = {1.0},
  note      = {Sample dataset},
  publisher = {Hugging Face and Figshare},
  url       = {https://huggingface.co/datasets/Bytte-AI/BBC_Igbo-Pidgin_Gold-Standard_NLP_Corpus},
  license   = {CC-BY-4.0}
}
```

**APA Style:**
```
Bytte AI. (2026). BBC Igbo–Pidgin Gold-Standard NLP Corpus (Version 1.0) [Data set]. 
Hugging Face. https://huggingface.co/datasets/Bytte-AI/BBC_Igbo-Pidgin_Gold-Standard_NLP_Corpus
```

---

## 📜 License

This dataset is released under **CC-BY-4.0 (Creative Commons Attribution 4.0 International)**.

**You are free to:**
- ✅ Share — copy and redistribute the material
- ✅ Adapt — remix, transform, and build upon the material
- ✅ Commercial use — use for commercial purposes

**Under the following terms:**
- 📌 **Attribution** — You must give appropriate credit to Bytte AI

See [LICENSE](LICENSE) for full details.

---

## 🤝 Contributing

We welcome contributions to improve and expand this corpus! Here's how you can help:

### Reporting Issues
- Found annotation errors? [Open an issue](https://github.com/Bytte-AI/BBC_Igbo-Pidgin_Gold-Standard_NLP_Corpus/issues)
- Have suggestions? Share them via [contact@bytteai.xyz](mailto:contact@bytteai.xyz)

### Proposed Enhancements
- Additional annotation layers (syntax, morphology)
- Inter-annotator agreement studies
- Alignment with other African language corpora
- Expanded coverage (more articles, languages)

---

## 🌍 Related Resources

### African Language Datasets
- **JW300** - Parallel corpus including Igbo
- **MENYO-20k** - Yoruba-English parallel corpus
- **AfriSenti** - Sentiment analysis for African languages
- **MasakhaNER** - NER for African languages
- **Lanfrica** - Directory of African language resources

### Tools & Models
- **AfroXLMR** - Multilingual model for African languages
- **IgboAPI** - Tools for Igbo language processing
- **Label Studio** - Annotation platform used for this corpus

---

## 📞 Contact & Support

**Organization:** Bytte AI  
**Website:** https://www.bytte.xyz/  
**Email:** contact@bytteai.xyz  

**Download Links:**
- 🤗 **Hugging Face:** https://huggingface.co/datasets/Bytte-AI/BBC_Igbo-Pidgin_Gold-Standard_NLP_Corpus
- 📊 **Figshare:** https://figshare.com/articles/dataset/BBC_Igbo_Pidgin_Gold-Standard_NLP_Corpus/31249567

---

## 🙏 Acknowledgments

This corpus was created through the dedicated efforts of the Bytte AI annotation team. We acknowledge:

- **BBC Igbo and BBC Pidgin** for providing authoritative journalism in African languages
- **Label Studio** for the annotation platform
- **The African NLP community** for inspiring this work
- **Our annotators** for their meticulous and thoughtful work

---

## 📅 Version History

### v1.0 (February 2026) - Initial Release
- 217 annotated samples across 4 datasets
- Tasks: Intent classification, sentiment analysis, sentence segmentation, NER
- Languages: Nigerian Igbo, Nigerian Pidgin English
- Human-in-the-loop annotation with quality metrics

---

## 🔮 Roadmap

While there are no immediate plans for expansion, potential future directions include:

- 🌱 Additional samples from diverse sources
- 🌱 More annotation tasks (POS tagging, dependency parsing)
- 🌱 Inter-annotator agreement studies
- 🌱 Expansion to other Nigerian languages (Yoruba, Hausa)
- 🌱 Cross-lingual alignment with English

Community feedback will help shape future development priorities.

---

<div align="center">

**By [Bytte AI](https://www.bytte.xyz/) for African language NLP**

[![Twitter](https://img.shields.io/twitter/follow/BytteAI?style=social)](https://twitter.com/BytteAI)
[![GitHub](https://img.shields.io/github/stars/Bytte-AI?style=social)](https://github.com/Bytte-AI)

</div>
