<div align="center">

![Cover Page](images/Resized.png)

# 🤖 **Inside-Medium : The Right Article, at the Right Time**

*Discover trending, relevant reads instantly with AI-powered article matching!*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Features](#features) • [Installation](#installation) • [Documentation](#documentation) • [Usage](#usage) • [Contributing](#contributing)

</div>

---

# 🌟 Overview

**Inside-Medium** is an AI-powered content recommendation engine designed to help readers find the most relevant and high-quality Medium articles based on their interests or selected articles. By leveraging Natural Language Processing (NLP) and Topic Modeling (NMF) techniques, the system extracts hidden topics from articles, encodes them into meaningful vectors, and uses cosine similarity to recommend similar content.

---

# Dataset - Medium Articles Dataset

📎 **Source**: [Medium Articles Dataset – Kaggle](https://www.kaggle.com/datasets/dorianlazar/medium-articles-dataset/data)

The **Medium Articles Dataset** is a curated collection of publicly available articles published on Medium.com. It contains both **textual content and engagement metadata**, making it ideal for tasks like recommendation systems, NLP, and content analysis.

#### 📁 Dataset Highlights:

* **Total Records**: \~8,000 articles
* **Key Columns**:

  * `title`: Title of the article
  * `subtitle`: Subtitle or secondary heading
  * `author`: Author of the article
  * `date`: Publication date
  * `claps`: Number of claps (engagement metric)
  * `reading_time`: Estimated reading time (in minutes)
  * `publication`: Name of the publication (if any)
  * `url`: Link to the original article
  * `article`: Full textual content of the article

#### ✅ Why This Dataset?

* Great for **topic modeling**, **text classification**, and **recommendation systems**
* Contains real-world engagement signals (`claps`) to enrich the model
* Useful for building **AI-driven content discovery platforms** like Inside-Medium

> 📌 **Dataset Link**: [https://www.kaggle.com/datasets/dorianlazar/medium-articles-dataset/data](https://www.kaggle.com/datasets/dorianlazar/medium-articles-dataset/data)

---

# 🚀 Features of *Inside-Medium*

* 🔍 **Content-Based Article Recommendation**
  Recommends articles similar to a user’s query based on textual content and latent topic features.


* 📈 **Similarity Scoring**
  Calculates cosine similarity between articles to identify the most relevant ones.

* 📑 **Interactive Query Support**
  Users can input any article title to retrieve a list of the most similar articles.

* 🧼 **Modular, Clean Codebase**
  Structured using classes for vectorization, normalization, and similarity search with full docstrings and logging.

* 📦 **Reproducible Pipeline**
  Complete workflow from raw data to recommendations—easy to extend or integrate into other systems.

* 🧾 **Logging and Error Handling**
  Built-in logging for debugging and tracking progress/errors in each module.

* 📂 **Scalable Design**
  Easy to adapt for larger datasets or additional features like user profiling or collaborative filtering.

  

