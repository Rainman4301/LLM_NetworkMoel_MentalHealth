import os
import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import faiss
from faiss import read_index, write_index
import torch
import logging
import nltk
from openai import OpenAI  # Add this import for API
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pdfplumber  # For PDF extraction
import requests
import json
from dotenv import load_dotenv
import tkinter as tk
from tkinter import Text, Scrollbar, Entry, Button, END, NORMAL, DISABLED

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon', quiet=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Detect device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

# Load environment variables
load_dotenv()

# Set up OpenAI client for Hugging Face API router
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)
model_name = "meta-llama/Llama-3.3-70B-Instruct:fireworks-ai"  # Use a larger model via API

# Load embedder
embedder = SentenceTransformer('cambridgeltl/SapBERT-from-PubMedBERT-fulltext', device=device)

# Configurable paths (change category as needed)
category = 'general'  # 'anxiety', 'depression', 'ptsd', 'suicide', general
pdf_dir = f'./rag_system/{category}/'
rag_chunks_csv_path = f'./rag_system/{category}/chunks.csv'
rag_embeddings_path = f'./rag_system/{category}/chunk_embeddings.npy'
rag_index_path = f'./rag_system/{category}/faiss_index.index'

# Ensure directories exist
os.makedirs(os.path.dirname(rag_chunks_csv_path), exist_ok=True)

# Function to extract and chunk text from a single PDF
def extract_and_chunk_pdf(pdf_path, chunk_size=300, overlap=40):
    chunks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""  # Use layout=True if needed: page.extract_text(layout=True)
                full_text += f"\n\n[Page {page_num}]\n{text}"

        # Simple chunking: Split by words with overlap
        words = full_text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                'content': chunk_text,
                'source_pdf': os.path.basename(pdf_path),
                'page_start': (i // chunk_size) + 1,  # Approximate page
                'category': category
            })

        logger.info(f"Extracted {len(chunks)} chunks from {pdf_path}")

    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
    return chunks

# Step 1: Process all PDFs and save chunks to CSV
if os.path.exists(rag_chunks_csv_path):
    logger.info(f"Loading existing chunks from {rag_chunks_csv_path}")
    chunks_df = pd.read_csv(rag_chunks_csv_path)
else:
    all_chunks = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            pdf_chunks = extract_and_chunk_pdf(pdf_path)
            all_chunks.extend(pdf_chunks)

    if not all_chunks:
        raise ValueError(f"No chunks extracted from PDFs in {pdf_dir}")
    
    chunks_df = pd.DataFrame(all_chunks)
    chunks_df['chunk_id'] = range(len(chunks_df))  # Add unique ID
    chunks_df.to_csv(rag_chunks_csv_path, index=False)
    logger.info(f"Saved {len(chunks_df)} chunks to {rag_chunks_csv_path}")

# Add sentiment if not present
if 'sentiment' not in chunks_df.columns:
    logger.info("Computing sentiment for chunks...")
    sia = SentimentIntensityAnalyzer()
    
    def get_sentiment(text):
        score = sia.polarity_scores(text)['compound']
        if score > 0.05:
            return 'positive'
        elif score < -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    chunks_df['sentiment'] = chunks_df['content'].apply(get_sentiment)
    chunks_df.to_csv(rag_chunks_csv_path, index=False)  # Resave with sentiment
    logger.info("Sentiment added and CSV updated.")

# Step 2: Load or compute embeddings
if os.path.exists(rag_embeddings_path):
    logger.info("Loading precomputed embeddings...")
    chunk_embeddings_np = np.load(rag_embeddings_path)
else:
    logger.info("Computing embeddings...")
    chunk_contents = chunks_df['content'].tolist()
    chunk_embeddings = embedder.encode(
        chunk_contents,
        batch_size=128,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device,
        normalize_embeddings=True  # Built-in normalization
    )
    chunk_embeddings_np = chunk_embeddings.cpu().numpy()
    np.save(rag_embeddings_path, chunk_embeddings_np)
    logger.info(f"Embeddings saved to {rag_embeddings_path}")

# Step 3: Build/Load FAISS Index
d = chunk_embeddings_np.shape[1]  # Embedding dimension
if os.path.exists(rag_index_path):
    logger.info("Loading existing FAISS index...")
    faiss_index = read_index(rag_index_path)
else:
    logger.info("Building FAISS index...")
    index = faiss.IndexFlatIP(d)  # Inner product for normalized vectors
    index.add(chunk_embeddings_np)
    write_index(index, rag_index_path)
    logger.info(f"FAISS index saved to {rag_index_path}")

# Configurable paths for ICD-11 data
icd_dir = './rag_system/ICD-11_Data/'
icd_json_path = f'{icd_dir}/icd_disorders.json'
icd_chunks_csv_path = f'{icd_dir}/icd_chunks.csv'
icd_embeddings_path = f'{icd_dir}/icd_embeddings.npy'
icd_index_path = f'{icd_dir}/faiss_index.index'

# Ensure directories exist
os.makedirs(icd_dir, exist_ok=True)

# ICD-11 API settings (assumes environment variables for credentials)
CLIENT_ID = os.environ.get('ICD_CLIENT_ID')
CLIENT_SECRET = os.environ.get('ICD_CLIENT_SECRET')
SCOPE = 'icdapi_access'
GRANT_TYPE = 'client_credentials'
TOKEN_ENDPOINT = 'https://icdaccessmanagement.who.int/connect/token'
API_BASE = 'https://id.who.int/icd'
RELEASE = '2025-01'  # Update to the latest release if needed
LINEARIZATION = 'mms'  # Mortality and Morbidity Statistics
CHAPTER_ENTITY_ID = '334423054'  # Entity ID for Chapter 6: Mental, behavioural or neurodevelopmental disorders

# Function to get OAuth token
def get_access_token():
    if not CLIENT_ID or not CLIENT_SECRET:
        raise ValueError("ICD_CLIENT_ID and ICD_CLIENT_SECRET must be set in environment variables.")
    
    payload = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'scope': SCOPE,
        'grant_type': GRANT_TYPE
    }
    response = requests.post(TOKEN_ENDPOINT, data=payload)
    response.raise_for_status()
    return response.json()['access_token']

# Function to fetch entity data
def fetch_entity(entity_id, token, is_foundation=False):
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json',
        'Accept-Language': 'en',  # Can change for other languages
        'API-Version': 'v2'
    }
    if is_foundation:
        url = f'{API_BASE}/entity/{entity_id}'
    else:
        url = f'{API_BASE}/release/11/{RELEASE}/{LINEARIZATION}/{entity_id}'
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

# Recursive function to collect all disorders under a parent entity
def collect_disorders(parent_id, token, collected=[], depth=0, max_depth=10):
    if depth > max_depth:
        logger.warning(f"Max depth reached for {parent_id}")
        return collected
    
    try:
        data = fetch_entity(parent_id, token)
        
        # Extract relevant fields
        code = data.get('code', '')
        fully_specified_name = data.get('title', {}).get('@value', '') if isinstance(data.get('title'), dict) else data.get('title', '')
        description = data.get('definition', {}).get('@value', '') if isinstance(data.get('definition'), dict) else data.get('definition', '')
        
        # Exclusions
        exclusions = []
        if 'exclusion' in data:
            for excl in data['exclusion']:
                label = excl.get('label', {}).get('@value', '') if isinstance(excl.get('label'), dict) else excl.get('label', '')
                exclusions.append(label)
        exclusions_str = '; '.join(exclusions)
        
        # Index Terms
        index_terms = []
        if 'indexTerm' in data:
            for term in data['indexTerm']:
                label = term.get('label', {}).get('@value', '') if isinstance(term.get('label'), dict) else term.get('label', '')
                index_terms.append(label)
        index_terms_str = '; '.join(index_terms)
        
        # Check if this is a leaf node (no children)
        has_children = 'child' in data and data['child']
        
        # Only collect if it's a leaf node, has a code, and is not a chapter/block
        if not has_children and code and fully_specified_name and not fully_specified_name.startswith('Chapter') and not fully_specified_name.startswith('Block'):
            collected.append({
                'code': code,
                'fully_specified_name': fully_specified_name,
                'description': description,
                'exclusions': exclusions_str,
                'all_index_terms': index_terms_str,
                'entity_id': parent_id
            })
            logger.info(f"Collected (leaf): {code} - {fully_specified_name}")
        
        # Recurse on children if present
        if has_children:
            for child_uri in data['child']:
                child_id = child_uri.split('/')[-1]  # Extract ID from URI
                collect_disorders(child_id, token, collected, depth + 1, max_depth)
    
    except requests.HTTPError as e:
        logger.error(f"Error fetching {parent_id}: {e}")
    
    return collected

# Step 1: Fetch and save ICD-11 data to CSV and JSON
if os.path.exists(icd_chunks_csv_path):
    logger.info(f"Loading existing ICD data from {icd_chunks_csv_path}")
    icd_df = pd.read_csv(icd_chunks_csv_path)
elif os.path.exists(icd_json_path):
    logger.info(f"Loading existing ICD data from {icd_json_path}")
    with open(icd_json_path, 'r') as f:
        disorders = json.load(f)
    icd_df = pd.DataFrame(disorders)
    icd_df.to_csv(icd_chunks_csv_path, index=False)
    logger.info(f"Saved CSV from JSON to {icd_chunks_csv_path}")
else:
    token = get_access_token()
    disorders = collect_disorders(CHAPTER_ENTITY_ID, token)
    
    if not disorders:
        raise ValueError("No disorders collected from ICD-11 API.")
    
    with open(icd_json_path, 'w') as f:
        json.dump(disorders, f, indent=4)
    logger.info(f"Saved disorders to {icd_json_path}")
    
    icd_df = pd.DataFrame(disorders)
    icd_df.to_csv(icd_chunks_csv_path, index=False)
    logger.info(f"Saved {len(icd_df)} disorders to {icd_chunks_csv_path}")

# Add sentiment if not present (though may not be as relevant for clinical data, but for consistency)
if 'sentiment' not in icd_df.columns:
    logger.info("Computing sentiment for ICD entries...")
    sia = SentimentIntensityAnalyzer()
    
    def get_sentiment(text):
        score = sia.polarity_scores(text)['compound']
        if score > 0.05:
            return 'positive'
        elif score < -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    # Concatenate fields for sentiment
    icd_df['content'] = icd_df.apply(lambda row: f"{row['code']} {row['fully_specified_name']} {row['description']} {row['exclusions']} {row['all_index_terms']}", axis=1)
    icd_df['sentiment'] = icd_df['content'].apply(get_sentiment)
    icd_df.to_csv(icd_chunks_csv_path, index=False)
    logger.info("Sentiment added and CSV updated.")
else:
    # Ensure content column exists
    icd_df['content'] = icd_df.apply(lambda row: f"{row['code']} {row['fully_specified_name']} {row['description']} {row['exclusions']} {row['all_index_terms']}", axis=1)

# Step 2: Load or compute embeddings
if os.path.exists(icd_embeddings_path):
    logger.info("Loading precomputed embeddings...")
    icd_embeddings_np = np.load(icd_embeddings_path)
else:
    logger.info("Computing embeddings...")
    chunk_contents = icd_df['content'].tolist()
    chunk_embeddings = embedder.encode(
        chunk_contents,
        batch_size=128,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device,
        normalize_embeddings=True
    )
    icd_embeddings_np = chunk_embeddings.cpu().numpy()
    np.save(icd_embeddings_path, icd_embeddings_np)
    logger.info(f"Embeddings saved to {icd_embeddings_path}")

# Step 3: Build/Load FAISS Index
if os.path.exists(icd_index_path):
    logger.info("Loading existing FAISS index...")
    icd_index = read_index(icd_index_path)
else:
    logger.info("Building FAISS index...")
    d = icd_embeddings_np.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(icd_embeddings_np)
    write_index(index, icd_index_path)
    logger.info(f"FAISS index saved to {icd_index_path}")

# Loaded resources for all
# Categories (assuming lowercase for paths, but adjust if needed)
# 'anxiety', 'depression', 'ptsd', 'suicide',
categories = ['general']  # Include 'general' as fallback

# Load category-specific resources into a dictionary
category_resources = {}
for cat in categories:
    chunks_csv = f'./rag_system/{cat}/chunks.csv'
    emb_path = f'./rag_system/{cat}/chunk_embeddings.npy'
    idx_path = f'./rag_system/{cat}/faiss_index.index'
    
    if os.path.exists(chunks_csv) and os.path.exists(emb_path) and os.path.exists(idx_path):
        chunks_df = pd.read_csv(chunks_csv)
        # Ensure sentiment is present (fallback compute if missing)
        if 'sentiment' not in chunks_df.columns:
            logger.info(f"Computing sentiment for {cat} chunks...")
            sia = SentimentIntensityAnalyzer()
            def get_sentiment(text):
                score = sia.polarity_scores(text)['compound']
                if score > 0.05: return 'positive'
                elif score < -0.05: return 'negative'
                else: return 'neutral'
            chunks_df['sentiment'] = chunks_df['content'].apply(get_sentiment)
            chunks_df.to_csv(chunks_csv, index=False)
            logger.info(f"Sentiment added for {cat} and CSV updated.")
        
        chunk_emb_np = np.load(emb_path)
        cat_index = read_index(idx_path)
        category_resources[cat] = {
            'df': chunks_df,
            'embeddings': chunk_emb_np,
            'index': cat_index
        }
        logger.info(f"Loaded resources for category: {cat}")
    else:
        logger.warning(f"Missing resources for category: {cat}. Skipping.")

# Load ICD-11 resources
icd_dir = './rag_system/ICD-11_Data/'
icd_chunks_csv_path = f'{icd_dir}/icd_chunks.csv'
icd_embeddings_path = f'{icd_dir}/icd_embeddings.npy'
icd_index_path = f'{icd_dir}/faiss_index.index'

if os.path.exists(icd_chunks_csv_path):
    icd_df = pd.read_csv(icd_chunks_csv_path)
    if 'content' not in icd_df.columns:
        icd_df['content'] = icd_df.apply(lambda row: f"{row['code']} {row['fully_specified_name']} {row['description']} {row['exclusions']} {row['all_index_terms']}", axis=1)
        icd_df.to_csv(icd_chunks_csv_path, index=False)
else:
    raise FileNotFoundError(f"ICD-11 CSV not found at {icd_chunks_csv_path}. Please run the ICD-11 prep script first.")

if os.path.exists(icd_embeddings_path):
    icd_embeddings_np = np.load(icd_embeddings_path)
else:
    raise FileNotFoundError(f"ICD-11 embeddings not found at {icd_embeddings_path}.")

if os.path.exists(icd_index_path):
    icd_index = read_index(icd_index_path)
else:
    raise FileNotFoundError(f"ICD-11 FAISS index not found at {icd_index_path}.")

# API-based generation for RAG (single prompt)
def generate_with_llm_rag(prompt: str, max_tokens: int = 120):
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.8,
        top_p=0.95,
        frequency_penalty=1.2  # Approximate repetition_penalty with frequency_penalty
    )
    return completion.choices[0].message.content.strip()

# Category retrieval with method - refined to use detected category
def category_retrieve(category: str, query: str, method: str = 'original', k: int = 10) -> list[dict]:  # Retrieve more for rerank
    if category not in category_resources:
        logger.warning(f"Category '{category}' not found. Falling back to 'general'.")
        category = 'general'
        if category not in category_resources:
            raise ValueError("No 'general' resources available as fallback.")
    
    res = category_resources[category]
    chunks_df = res['df']
    cat_index = res['index']
    
    queries = [query]
    
    if method == 'multi':
        prompt = f"""
            Strictly follow: Output EXACTLY 5 unique variant queries similar to '{query}', one per line.
            Preserve key elements (events, causes, feelings).
            No reasoning, no steps, no examples, no introductions, no extra text at all.
            Directly start with the first query.

            Example output format (do not include this in output):
            Variant1
            Variant2
            Variant3
            Variant4
            Variant5
            """
        variants_response = generate_with_llm_rag(prompt)
        variants = [line.strip() for line in variants_response.split('\n') if line.strip() and len(line.split()) > 2 and not re.match(r'^(#|Step|Example|For reference|The|To|Output|Preserve|No|Directly)', line, re.I)]
        queries = list(set(v for v in variants if v.lower() != query.lower()))[:5]
        if not queries or len(queries) < 3:  # Fallback if poor generation
            queries = [query, f"{query} coping strategies", f"{query} emotional support"]
        logger.info(f"Generated multi-queries: {queries}")

    elif method == 'hyde':
        prompt = f"""
            Generate a concise hypothetical answer document for the query '{query}'.
            Output only the document text, without any introductions, explanations, numbering,
            or extra formatting.
        """
        hyde_response = generate_with_llm_rag(prompt)
        # Clean up: Join lines into a single document string, remove leading/trailing whitespace, and strip common prefixes like "Document:"
        hyde_doc = ' '.join(line.strip() for line in hyde_response.split('\n') if line.strip()).replace('Document:', '').strip()
        queries = [hyde_doc]  # Use the cleaned doc as "query" for embedding
        logger.info(f"Generated HyDE document: {hyde_doc[:200]}...")
    
    # Embed all queries
    query_embs = embedder.encode(queries, convert_to_tensor=True, device=device, normalize_embeddings=True).cpu().numpy()
    
    # Search for each, collect unique results with max score
    all_results = {}
    for q_emb in query_embs:
        q_emb = q_emb.reshape(1, -1)
        distances, indices = cat_index.search(q_emb, k=k)
        for i in range(len(distances[0])):
            idx = indices[0][i]
            if idx == -1: continue
            score = distances[0][i]
            if idx not in all_results or score > all_results[idx]['score']:
                row = chunks_df.iloc[idx]
                all_results[idx] = {
                    'content': row['content'],
                    'source_pdf': row['source_pdf'],
                    'page_start': row['page_start'],
                    'sentiment': row['sentiment'],
                    'score': score
                }
    
    # Get top k by score first (before rerank)
    sorted_results = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)[:k]
    return sorted_results

# Rerank function
def rerank_results(results: list[dict]) -> list[dict]:
    sentiment_order = {'positive': 0, 'neutral': 1, 'negative': 2}
    # Sort by sentiment order, then by score descending
    sorted_results = sorted(results, key=lambda x: (sentiment_order.get(x['sentiment'], 3), -x['score']))
    return sorted_results[:5]  # Top 5 after rerank

# Full workflow - refined to always use top similar post for category
def full_rag_workflow(query: str, method: str = 'original') -> list[dict]:
    category = 'general'  # Comment out if you want dynamic category

    # Step 2: Retrieve from category-specific database with method
    retrieved = category_retrieve(category, query, method=method)
    
    # Step 3: Rerank
    reranked = rerank_results(retrieved)
    return reranked

# API-based generation for chatbot (messages list)
def generate_with_llm(messages: list[dict], max_tokens: int = 150) -> str:
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.8,
        top_p=0.95,
        frequency_penalty=1.2  # Approximate repetition_penalty
    )
    return completion.choices[0].message.content.strip()

# GUI class for the chatbot
class MentalHealthChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mental Health Chatbot")
        self.root.geometry("600x500")
        self.root.configure(bg="#17202A")

        self.system_prompt = """
            You are a compassionate and empathetic mental health assistant, adapting your role dynamically—like a caring friend for casual chats or a thoughtful guide for deeper reflections—to keep interactions fresh. Your responses should:
            - Analyze input empathetically, tailoring to query, history, and guidelines without rigid patterns (e.g., vary openings beyond name + feeling summary; try questions or shared observations first).
            - Respond conversationally: Mix tones (warm, curious, uplifting) and structures (e.g., short paragraphs, bullets for suggestions) to avoid repetition.
            - Weave in guidelines naturally (e.g., coping ideas as "One thing that helps me is...").
            - Suggest practical steps gently, varying by context; for ICD-11 (if provided), highlight as non-diagnostic ideas and urge professional help.
            - IMPORTANT: Do not mention or reference any 'Doc', 'Guideline', 'Section', scores, sentiments, sources, or internal labels. Integrate as innate knowledge.
            - Responses should be 60-100 words; experiment with styles for natural flow.
        """

        self.history = []  # List of dicts: [{'role': 'user', 'content': ...}, {'role': 'assistant', 'content': ...}]
        self.max_context_tokens = 8192
        self.max_history_pairs = 10
        self.method = 'hyde'  # Changed to match original; can be 'hyde' if preferred

        # Scrollbar for chat (now sibling of text_cons)
        self.scrollbar = Scrollbar(self.root)
        self.scrollbar.place(relheight=0.85, relwidth=0.03, relx=0.0, rely=0.0)

        # Chat display area
        self.text_cons = Text(self.root, bg="#17202A", fg="#EAECEE", font="Helvetica 14", padx=5, pady=5, wrap="word", yscrollcommand=self.scrollbar.set)
        self.text_cons.place(relheight=0.85, relwidth=0.97, relx=0.03, rely=0.0)
        self.text_cons.config(state=DISABLED)

        # Configure scrollbar command
        self.scrollbar.config(command=self.text_cons.yview)

        # Configure tags for alignment
        self.text_cons.tag_config('user', justify='right', foreground="#AED6F1")  # Light blue for user
        self.text_cons.tag_config('bot', justify='left', foreground="#ABEBC6")   # Light green for bot

        # Entry for user message
        self.entry_msg = Entry(self.root, bg="#2C3E50", fg="#EAECEE", font="Helvetica 13")
        self.entry_msg.place(relwidth=0.74, relheight=0.06, rely=0.92, relx=0.011)
        self.entry_msg.focus()
        self.entry_msg.bind("<Return>", self.send_message)  # Bind Enter key to send

        # Send button
        self.button_msg = Button(self.root, text="Send", font="Helvetica 10 bold", width=20, bg="#ABB2B9", command=self.send_message)
        self.button_msg.place(relx=0.77, rely=0.92, relheight=0.06, relwidth=0.22)

        # Initial welcome message
        self.append_message("Chatbot: Welcome to the Mental Health Chatbot. How can I help you today?\n")

    def append_message(self, message):
        self.text_cons.config(state=NORMAL)
        if message.startswith("You:"):
            tag = 'user'
            # For better right alignment, add spaces or use lmargin, but simple justify for now
        else:
            tag = 'bot'
        self.text_cons.insert(END, message + "\n\n", tag)
        self.text_cons.config(state=DISABLED)
        self.text_cons.see(END)

    def send_message(self, event=None):
        query = self.entry_msg.get().strip()
        if not query:
            return
        if query.lower() in ['quit', 'exit']:
            self.append_message("Chatbot: Goodbye! Take care.")
            self.root.quit()
            return

        self.append_message(f"You: {query}")
        self.entry_msg.delete(0, END)

        try:
            # Accumulate user inputs from history + current query
            user_inputs = [h['content'] for h in self.history if h['role'] == 'user'] + [query]
            accumulated_input = ' '.join(user_inputs)

            # Embed accumulated input
            query_embedding = embedder.encode(accumulated_input, convert_to_tensor=True, device=device, normalize_embeddings=True)
            query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)

            # Search ICD-11 index for top 4 matches (adjusted thresholds)
            distances, indices = icd_index.search(query_embedding_np, k=4)
            icd_infos = []
            high_score_disorders = []
            if len(distances[0]) > 0:
                for i in range(len(distances[0])):
                    score = distances[0][i]
                    idx = indices[0][i]
                    if idx == -1: continue
                    if score > 0.8:
                        row = icd_df.iloc[idx]
                        disorder_name = row['fully_specified_name']
                        code = row['code']
                        symptoms = row['description']
                        icd_infos.append(f"""
                            Disorder Name: {disorder_name}
                            Disorder Code: {code}
                            Disorder symptoms: {symptoms}
                            """)
                        if score > 0.9:
                            high_score_disorders.append(f"{disorder_name} ({code})")
            
            icd_prompt_section = ""
            if icd_infos:
                icd_prompt_section = f"\n\nPossible matching disorders from ICD-11 (similarity scores > 0.8):\n{''.join(icd_infos)}\nRemember, this is not a diagnosis; suggest professional help."

            # Get RAG results
            retrieved_docs = full_rag_workflow(query, method=self.method)
            
            # Format retrieved docs as context string
            # doc_context = "\n".join([f"Doc {i+1} (Sentiment: {doc['sentiment']}, Score: {doc['score']:.4f}): {doc['content']}" for i, doc in enumerate(retrieved_docs)])
            # Format retrieved docs as context string (no labels to avoid leakage)
            doc_context = "\n\n---\n\n".join([doc['content'] for doc in retrieved_docs])  # Separate contents with delimiters for readability in prompt


            # Build user prompt
            user_prompt = f"""
                    User's message: {query}

                    Additional relevant guidelines (use these to inform suggestions if they fit the query):
                    {doc_context}

                    Respond empathetically, drawing from guidelines for specific ideas. If the query involves loss or grief, suggest personalized memorials or support resources. Vary your language from previous responses.
                    """ + icd_prompt_section
            
            # Build messages
            messages = [{"role": "system", "content": self.system_prompt}] + self.history + [{"role": "user", "content": user_prompt}]
            
            # Token estimation and truncation
            def estimate_tokens(text: str) -> int:
                return len(text.split()) * 1.3 + 100

            while sum(estimate_tokens(msg['content']) for msg in messages) > self.max_context_tokens:
                if len(self.history) <= 2:
                    logger.warning("Context exceeds limit; proceeding.")
                    break
                self.history = self.history[2:]  # Remove oldest pair
                messages = [{"role": "system", "content": self.system_prompt}] + self.history + [{"role": "user", "content": user_prompt}]
                logger.info(f"Truncated history to {len(self.history)} entries.")
            
            # Generate response
            response = generate_with_llm(messages, max_tokens=100)

            # Append diagnosis suggestion if applicable
            if high_score_disorders:
                suggestions = ', '.join(high_score_disorders)
                response += f"\n\nBased on the provided information, your symptoms suggest possible diagnoses of {suggestions} according to ICD-11."

            self.append_message(f"Chatbot: {response}")

            # Append to history
            self.history.append({"role": "user", "content": query})
            self.history.append({"role": "assistant", "content": response})
        
        except Exception as e:
            logger.error(f"Error in chatbot: {e}")
            self.append_message("Chatbot: Sorry, something went wrong. Please try again.")

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = MentalHealthChatbotGUI(root)
    root.mainloop()