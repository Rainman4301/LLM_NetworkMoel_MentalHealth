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
- Evaluated with topic coherence, topic diversity, and silhouette score
- Includes LLM-assisted topic refinement

### 2. Network Analysis
- Builds subtopic co-occurrence graphs
- Computes network metrics: modularity, centrality, and assortativity
- Produces interactive node-link visualizations
- Reveals how stressors (e.g., work, relationships, finances) influence multiple mental health conditions

### 3. Mental Health Chatbot
- Uses a RAG pipeline combining retrieval + LLM generation
- Integrates ICD-11 context (non-diagnostic, supportive framing only)
- Adapts tone using prompt engineering guidelines
- Supports multiple query modes: Original, Multi, and HyDE
- Includes sentiment analysis to improve emotional alignment
- Produces empathetic, safe, and context-aware responses

### 4. Data Preprocessing
- Cleans posts (removes noise, stopwords, filler terms)
- Generates n-grams (uni-, bi-, tri-grams)
- Uses SentenceTransformers to generate embeddings
- Prepares data for topic modeling and graph construction

### 5. Human-in-the-Loop Evaluation
- Combines automated metrics with manual review
- Uses LLMs to refine topic labels and ensure semantic fit
- Ensures interpretability and quality of extracted topics

---

## Repository Structure

```
LLM_NetworkModel_MentalHealth/
├── data/                       # Raw and processed datasets
│   ├── reddit_data/            # Reddit mental health data
│   ├── beyondblue_data/        # Beyond Blue forum data
│   ├── network_graph/          # Network analysis outputs
│   └── AUS_weather/            # Australian weather data (if applicable)
├── lib/                        # External libraries
├── pic/                        # Images and visualizations
├── rag_system/                 # Retrieval-Augmented Generation pipeline
│   ├── ICD-11_Data/            # ICD-11 related data
│   ├── beyondblue_post_data/   # Beyond Blue post data
│   └── general/                # General utilities
├── topic_modeling_result/      # Topic modeling outputs
├── chatbot_local.py            # Local chatbot implementation
├── chatbot_API.py              # API-based chatbot implementation
├── hypertune.ipynb             # BERTopic hyperparameter tuning
├── data_analysis.ipynb         # Network analysis
├── data_collection.ipynb       # Data collection
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/LLM_NetworkModel_MentalHealth.git
   cd LLM_NetworkModel_MentalHealth
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download additional resources:**
   - **ICD-11 Data**: Set environment variables `ICD_CLIENT_ID` and `ICD_CLIENT_SECRET`
   - **MentalBERT Model**: Download from `mental/mental-bert-base-uncased`

---

## Usage

### 1. Topic Modeling
Run `hypertune.ipynb` and adjust UMAP, HDBSCAN, and vectorizer settings for optimal topic quality.

### 2. Network Analysis
Use `data_analysis.ipynb` to build subtopic networks and generate visualizations.

### 3. Mental Health Chatbot

**Local Mode:**
```bash
python chatbot_local.py
```

**API Mode:**
```bash
python chatbot_API.py
```

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
