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

## 🌟 Overview

**Inside-Medium** is an AI-powered content recommendation engine designed to help readers find the most relevant and high-quality Medium articles based on their interests or selected articles. By leveraging Natural Language Processing (NLP) and Topic Modeling (NMF) techniques, the system extracts hidden topics from articles, encodes them into meaningful vectors, and uses cosine similarity to recommend similar content.

---

## 📚 Dataset - Medium Articles Dataset

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

## 🚀 Features of *Inside-Medium*

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

---

## 📰 Published Article

Explore other Detailed Fine-Tuning Methods of Large Language Models with Mathematical Calculations:

🔗 Read the article here:
[Inside Medium’s Recommendation Engine: How It Knows What You’ll Love](https://medium.com/@priyampal/inside-mediums-recommendation-engine-how-it-knows-what-you-ll-love-982cc295e0de)

---

## 🛠️ Installation

#### Step - 1: Repository Cloning

```bash
# Clone the repository
git clone https://github.com/priyam-hub/Inside-Medium.git

# Navigate into the directory
cd Inside-Medium
```

#### Step - 2: Enviornmental Setup and Dependency Installation

```bash
# Run env_setup.sh
bash env_setup.sh

# Select 1 to create Python Environment
# Select 2 to create Conda Environment

# Python Version - 3.10

# Make the Project to run as a Local Package
python setup.py
```

#### Step - 3: Creation of Kaggle API

- Log-In to your Kaggle Account
- An API token downloaded from Kaggle Account Settings → Create New Token.
- Manually place your kaggle.json (downloaded from https://www.kaggle.com/settings) into this location:

```plaintext
C:\Users\<Your_Username>\.kaggle\kaggle.json
```

#### Step - 4: Create a .env file in the root directory to add Credentials or (Change the filename ".sample_env" to ".env")

```bash
KAGGLE_USERNAME = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
KAGGLE_API_KEY  = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

#### Step - 5: Run the Full Pipeline

```bash
# Run the Main Python Script
python main.py
```

#### Step - 6: Run the Flask Server (Up-Coming)

```bash
# Run the Web App using Flask Server
python web/app.py
```

**Note** - Upon running, navigate to the provided local URL in your browser to interact with the Inside-Medium Recommendation Engine

---

## 🧰 Technology Stack

**Python** – Core programming language used to build the recommendation pipeline, data processing, and backend logic.
🔗 [Install Python](https://www.python.org/downloads/)

**Pandas & NumPy** – Used for efficient data manipulation, cleaning, and numerical operations.
🔗 [Pandas Documentation](https://pandas.pydata.org/docs/) | [NumPy Documentation](https://numpy.org/doc/)

**Scikit-learn** – Used for feature extraction (TF-IDF), dimensionality reduction (NMF), and similarity computation.
🔗 [Scikit-learn Documentation](https://scikit-learn.org/stable/)

**Flask** – Lightweight Python web framework used to serve the recommendation engine as an API or simple web app.
🔗 [Flask Installation](https://flask.palletsprojects.com/en/latest/installation/)

**Logging** – Python’s built-in `logging` module used for tracking system operations and debugging.
🔗 [Logging Documentation](https://docs.python.org/3/library/logging.html)

**Kaggle API** – Used to automatically fetch and manage the Medium Articles dataset.
🔗 [Kaggle API Setup Guide](https://github.com/Kaggle/kaggle-api)

---

## 📁 Project Structure

```plaintext
Inside-Medium/
├── .env                                      # Store the Kaggle Username and API Key
├── .gitignore                                # Ignoring files for Git
├── env_setup.sh                              # Package installation configuration
├── folder_structure.py                       # Contains the Project Folder Structure
├── LICENCE                                   # MIT License
├── main.py                                   # Full Pipeline of the Project
├── README.md                                 # Project documentation
├── requirements.txt                          # Python dependencies
├── setup.py                                  # Create the Project as Python Package
├── config/                                   # Configuration files
│   ├── __init__.py                           
│   └── config.py/                            # All Configuration Variables of Pipeline
├── data/                                     # Data Directory
│   ├── images/                               # Medium Article Images Directory
│   ├── medium_normalized_data.csv            # Normalized Data of the Medium Articles
│   ├── medium_processed_data.csv             # Processed Data of the Medium Articles
│   └── medium_raw_data.csv                   # Raw Data of the Medium Articles
├── logger/                                   # Logger Setup Directory
│   └── logger.py                             # Format of the Logger Setup of the Project
├── notebooks/                                # Jupyter notebooks for experimentation
│   └── Recommendation_System.ipynb           # Experimented Recommendation Engine in Jupyter Notebook
├── results/                                  # Directory to Store the results of the Project
│   └── eda_results/                          # Directory to Store the EDA Results
├── src/                                      # Source code
│   ├── data_preprocessor/                    # Data Preprocessor Directory
│   │   ├── __init__.py  
│   │   └── data_preprocessor.py              # Python file process the raw data                                       
│   ├── exploratory_data_analysis/            # EDA Directory
│   │   ├── __init__.py   
│   │   └── exploratory_data_analyzer.py      # Python file to perform EDA                                   
│   ├── normalizer/                           # Text Normalizing Directory
│   │   ├── __init__.py                           
│   │   └── nmf_normalizer.py                 # Python File to Normalize the Preprocessed Data                                    
│   ├── recommendation_engine/                # Recommendation Engine Directory
│   │   ├── __init__.py   
│   │   └── similarity_finder.py              # Python file to perform similarity search   
│   ├── vectorizer/                           # Recommendation Engine Directory
│   │   ├── __init__.py   
│   │   └── tfidf_vectorizer.py               # Python file to perform vectorizer                             
│   └── utils/                                # Utility Functions Directory
│       ├── __init__.py                     
│       ├── data_loader.py                    # Load and Save Data from Local
│       ├── download_dataset.py               # Download the Data from Kaggle
│       └── save_plot.py                      # Save the Plot in Specified Path
└── web/
    ├── __init__.py  
    ├── static/                                
    │   ├── styles.css                        # Styling of the Web Page
    │   └── script.js                         # JavaScript File
    ├── templates/                                
    │   └── index.html                        # Default Web Page
    └── app.py/                               # To run the flask server
        
```

---

## 🔮 Future Work Roadmap

The *Inside-Medium* project can be extended significantly to offer a more personalized and intelligent content recommendation system. Here's a proposed roadmap structured in **three development phases**, each with an estimated time frame.

---

### 🚀 **Phase 1: UI & API Integration (1–2 Weeks)**

**Objective:** Transform the backend logic into a user-accessible application.

*  Build a clean and responsive frontend using **HTML/CSS/JS** for user interaction.
*  Deploy the article recommender as a **Flask API**, allowing input of article titles and displaying similar content.
*  Enable users to upload custom datasets (CSV) for analysis and recommendations.
*  Add search bar, loading indicators, and user-friendly error messages.

### 🧠 **Phase 2: Personalization & Topic Modeling (2–3 Weeks)**

**Objective:** Enhance the intelligence of the recommender.

*  Introduce **user profiles** to track reading history and provide personalized recommendations.
*  Apply **LDA or BERTopic** for better topic clustering and diversity in suggestions.
*  Integrate **claps, reading time, and tags** more deeply into the similarity scoring system.
*  Include feedback mechanism to rate recommended articles.

### 🧠 **Phase 3: Embedding Models & LLM Integration (3–4 Weeks)**

**Objective:** Upgrade the recommendation engine with deep learning and language models.

*  Replace TF-IDF + NMF with **sentence embeddings** using `SentenceTransformers` or Hugging Face models.
*  Use **vector databases (e.g., Qdrant, FAISS)** for faster and smarter similarity search.
*  Integrate with **LLMs (e.g., OpenAI, LLaMA via LangChain)** to enable query-based article retrieval using natural language.
*  Package the app into a **Docker container** and deploy to the cloud for scalability.

---


## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

<div align="center">

**Made by Priyam Pal - AI and Data Science Engineer**

[↑ Back to Top]

</div>
