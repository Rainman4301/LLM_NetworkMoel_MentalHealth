# LLM Network Model for Mental Health Analysis

This repository implements a Large Language Model (LLM)-based framework for topic modeling, network analysis, and an empathetic mental health chatbot. The project leverages modern NLP techniques to analyze online mental health discussions, uncover underlying themes, and support users through context-aware conversational AI.

## Project Overview

Online mental health forums contain rich, user-generated text that reflects real struggles, coping strategies, and emotional experiences. This project provides insights by combining:

- **Topic Modeling** to extract themes from mental health posts
- **Network Analysis** to visualize relationships between subtopics
- **Mental Health Chatbot** powered by Retrieval-Augmented Generation (RAG)

Together, these tools aim to support mental health research and explore AI-assisted early emotional support.

---

## Key Features

### 1. Topic Modeling
- Uses BERTopic with MentalBERT embeddings
- Extracts topics/subtopics for Depression, Anxiety, PTSD/Trauma, and Suicidal Thoughts
- Includes hyperparameter tuning for UMAP, HDBSCAN, and CountVectorizer
- Evaluated with topic coherence, and topic diversity
- Includes LLM-assisted topic refinement

### 2. Network Analysis
- Builds subtopic co-occurrence graphs
- Computes network metrics: modularity, centrality, and assortativity
- Produces interactive node-link visualizations
- Reveals how stressors (e.g., work, relationships, finances) influence multiple mental health conditions

### 3. Mental Health Chatbot
- Uses a RAG framework combining retrieval + LLM generation
- Integrates ICD-11 context (non-diagnostic, supportive framing only)
- Adapts tone using prompt engineering guidelines
- Supports multiple query modes: Original, Multi, and HyDE
- Includes sentiment analysis to improve emotional alignment
- Produces empathetic, safe, and context-aware responses

### 4. Data Preprocessing

**Topic Modeling Pipeline:**
- Removes irrelevant content using regular expressions
- Translates emojis and emoticons into text to preserve emotional cues
- Applies lowercasing, stop word removal, lemmatization, and tokenization
- Extracts bigrams and trigrams to capture domain-specific mental health phrases
- For BERT embeddings: minimal preprocessing (filtering, lowercasing, emoji conversion) to preserve sentence context

**RAG Database Pipeline:**
- Processes PDF guideline documents using pdfplumber
- Segments text into overlapping chunks (200 tokens, 40-token overlap) to maintain semantic continuity
- Cleans and normalizes text by removing non-linguistic artifacts, headers, footers, and formatting inconsistencies
- Anonymizes content to eliminate identifiable information from case examples
- Encodes chunks using MentalBERT-based sentence encoder for domain-specific embeddings
- Indexes vectors with FAISS for efficient similarity search

### 5. Human-in-the-Loop Evaluation
- Combines automated metrics with manual review
- Uses LLMs to refine topic labels and ensure semantic fit
- Ensures interpretability and quality of extracted topics

---

## Repository Structure

```
LLM_NetworkModel_MentalHealth/
├── data/                       # Raw and processed datasets
│   ├── reddit_data/            # Reddit scraped mental health data
│   ├── beyondblue_data/        # Beyond Blue forum scraped data
│   ├── network_graph/          # Network data and graphs
│   └── AUS_weather/            # Australian weather data
├── lib/                        # External libraries
├── pic/                        # Images and visualizations
├── rag_system/                 # Retrieval-Augmented Generation framework
│   ├── ICD-11_Data/            # ICD-11 guideline data
│   ├── beyondblue_post_data/   # Beyond Blue post embeddings
│   └── general/                # General clinical guidelines
├── topic_modeling_result/      # Topic modeling outputs
├── chatbot_local.py            # Local chatbot implementation
├── chatbot_API.py              # API-based chatbot implementation
├── hypertune.ipynb             # BERTopic hyperparameter tuning
├── data_network.ipynb          # Network graph generation
├── data_analysis.ipynb         # Topic modeling generation
├── data_collection.ipynb       # Data collection and scraping
├── .env                        # Environment variables (create from template)
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```




---

## Installation

### 1. Clone the repository

**Windows (Command Prompt or PowerShell):**
```cmd
git clone https://github.com/your-username/LLM_NetworkModel_MentalHealth.git
cd LLM_NetworkModel_MentalHealth
```


### 2. Install dependencies

**Windows:**
```cmd
pip install -r requirements.txt
```


### 3. Environment variable setup

Create a `.env` file in the project root directory and add your API credentials. The file should follow this format:

```dotenv
CLIENT_ID=your_reddit_client_id
CLIENT_SECRET=your_reddit_client_secret
USERNAME=your_reddit_username
PASSWORD=your_reddit_password
HF_TOKEN=your_huggingface_token
ICD_CLIENT_ID=your_icd_client_id
ICD_CLIENT_SECRET=your_icd_client_secret
```

**Variable descriptions:**
- `CLIENT_ID`, `CLIENT_SECRET`, `USERNAME`, `PASSWORD`: Reddit API credentials for scraping mental health forum data
- `HF_TOKEN`: Hugging Face authentication token for accessing MentalBERT and other models
- `ICD_CLIENT_ID`, `ICD_CLIENT_SECRET`: ICD-11 API credentials for retrieving clinical guideline data

**How to obtain credentials:**
- **Reddit API**: Create an app at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) and select "script" type
- **Hugging Face**: Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **ICD-11 API**: Request access at [icd.who.int/icdapi](https://icd.who.int/icdapi)

### 4. Download model files

The **MentalBERT** model will be automatically downloaded from Hugging Face when first running the scripts. Ensure your `HF_TOKEN` is configured if accessing gated models.

---


## Usage

## Usage

### 1. Data Collection
Run `data_collection.ipynb` to scrape and preprocess mental health forum data from Reddit and Beyond Blue. This notebook handles data cleaning and prepares datasets for topic modeling and analysis.

### 2. Topic Modeling
**Generate subtopics:**
- Run `data_analysis.ipynb` to extract subtopics using BERTopic with MentalBERT embeddings

**Optimize hyperparameters:**
- Run `hypertune.ipynb` to fine-tune UMAP, HDBSCAN, and CountVectorizer parameters for improved topic quality and coherence

### 3. Network Analysis
Run `data_network.ipynb` to build and visualize subtopic co-occurrence networks, revealing relationships and connections across mental health themes.

### 4. Mental Health Chatbot

**Local Mode (Windows):**
```cmd
python chatbot_local.py
```

**Local Mode (macOS/Linux):**
```bash
python chatbot_local.py
```

**API Mode:**
Run the `chatbot_API.py` script or notebook to use the API-based implementation with external LLM services.

---

## Results

### Topic Modeling
- Extracted themes for Depression, Anxiety, PTSD, and Suicidal Thoughts
- Validated using coherence, diversity, and silhouette scores
- Human review improved topic labels and clarity

### Network Analysis
- Identified central subtopics (e.g., emotional support, coping strategies)
- Revealed cross-condition connections
- Visual node-link diagrams highlight key patterns

### Mental Health Chatbot
- Generates empathetic, clear, and safe responses
- Provides ICD-11 contextual suggestions (non-diagnostic)
- Adapts tone to user emotional state

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **MentalBERT** for domain-specific embeddings
- **BERTopic** for topic modeling
- **Beyond Blue** for mental health forum data
- **ICD-11 API** for contextual mental health resources
