import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import faiss
from faiss import read_index, write_index
import torch
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from openai import OpenAI  # Add this import for API

# Download VADER lexicon if not already downloaded
# nltk.download('vader_lexicon', quiet=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Detect device (still needed for embedder, but not for LLM)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

# Set up OpenAI client for Hugging Face API router
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)
model_name = "meta-llama/Llama-3.3-70B-Instruct:fireworks-ai"  # Use a larger model via API

# RAG paths for Beyond Blue posts (configurable)
bblue_data_path = './rag_system/beyondblue/beyond_df.csv'
bblue_embeddings_path = './rag_system/beyondblue/post_embeddings.npy'
bblue_index_path = './rag_system/beyondblue/faiss_index.index'

# Load the Beyond Blue DataFrame
try:
    beyondblue_df = pd.read_csv(bblue_data_path)
    logger.info(f"Loaded {len(beyondblue_df)} posts from {bblue_data_path}")
except FileNotFoundError:
    raise FileNotFoundError(f"CSV file not found at {bblue_data_path}. Please check the path.")

# Load embedder (for embeddings)
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Load or compute embeddings for posts
if os.path.exists(bblue_embeddings_path):
    logger.info("Loading precomputed embeddings...")
    post_embeddings_np = np.load(bblue_embeddings_path)
else:
    logger.info("Computing embeddings...")
    post_contents = beyondblue_df['clean_title_content_comments'].tolist()
    post_embeddings = embedder.encode(
        post_contents,
        convert_to_tensor=True,
        device=device,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True  # Use built-in normalization
    )
    post_embeddings_np = post_embeddings.cpu().numpy()
    np.save(bblue_embeddings_path, post_embeddings_np)
    logger.info(f"Embeddings saved to {bblue_embeddings_path}")

# Build/Load FAISS Index for posts
if os.path.exists(bblue_index_path):
    logger.info("Loading precomputed FAISS index...")
    post_index = read_index(bblue_index_path)
else:
    logger.info("Building FAISS index...")
    d = post_embeddings_np.shape[1]  # Embedding dimension
    post_index = faiss.IndexFlatIP(d)  # Inner Product for normalized vectors
    post_index.add(post_embeddings_np)
    write_index(post_index, bblue_index_path)
    logger.info(f"FAISS index saved to {bblue_index_path}")

# Categories (assuming lowercase for paths, but adjust if needed)
categories = ['anxiety', 'depression', 'ptsd', 'suicide', 'general']  # Include 'general' as fallback

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

# Function for RAG Query on posts (to get category) - refined for single top match
def rag_search(query: str, k: int = 1) -> dict:
    """Embed query and search FAISS index for the most similar post to determine category."""
    query_embedding = embedder.encode(query, convert_to_tensor=True, device=device, normalize_embeddings=True)
    query_embedding_np = query_embedding.cpu().numpy()
    query_embedding_np = query_embedding_np.reshape(1, -1)  # Reshape for FAISS

    distances, indices = post_index.search(query_embedding_np, k=1)  # Always top 1 for category detection
    if len(distances[0]) == 0:
        raise ValueError("No similar posts found.")
    
    idx = indices[0][0]
    score = distances[0][0]
    post = beyondblue_df.iloc[idx]['clean_title_content_comments']
    category = beyondblue_df.iloc[idx]['Post Category'].lower()  # Normalize to lowercase
    logger.info(f"Most similar post (score {score:.4f}): {post[:200]}... Category: {category}")
    return {
        'post': post,
        'category': category,
        'score': score
    }

# API-based generation for RAG (single prompt)
def generate_with_llm_rag(prompt: str, max_tokens: int = 100):
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
def category_retrieve(category: str, query: str, method: str = 'original', k: int = 20) -> list[dict]:  # Retrieve more for rerank
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
        prompt = f"""Strictly follow: Output EXACTLY 5 unique variant queries similar to '{query}', one per line.
            Preserve key elements (events, causes, feelings).
            No reasoning, no steps, no examples, no introductions, no extra text at all.
            Directly start with the first query.

            Example output format (do not include this in output):
            Variant1
            Variant2
            Variant3
            Variant4
            Variant5"""
        variants_response = generate_with_llm_rag(prompt)
        variants = [line.strip() for line in variants_response.split('\n') if line.strip() and len(line.split()) > 2 and not re.match(r'^(#|Step|Example|For reference|The|To|Output|Preserve|No|Directly)', line, re.I)]
        queries = list(set(v for v in variants if v.lower() != query.lower()))[:5]
        if not queries or len(queries) < 3:  # Fallback if poor generation
            queries = [query, f"{query} coping strategies", f"{query} emotional support"]
        logger.info(f"Generated multi-queries: {queries}")

    elif method == 'hyde':
        prompt = f"Generate a concise hypothetical answer document for the query '{query}'. Output only the document text, without any introductions, explanations, numbering, or extra formatting."
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
def full_rag_workflow(query: str, method: str = 'original') -> tuple[list[dict], dict]:
    # Step 1: Get category from the most similar Beyond Blue post
    post_result = rag_search(query)  # k=1 by default
    category = post_result['category']

    category = 'general'  # Comment out if you want dynamic category

    # Step 2: Retrieve from category-specific database with method
    retrieved = category_retrieve(category, query, method=method)
    
    # Step 3: Rerank
    reranked = rerank_results(retrieved)
    return reranked, post_result

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




# ==============================================================================




# Refined chatbot loop with history and token management
def run_chatbot(method: str = 'original', max_context_tokens: int = 8192, max_history_pairs: int = 10):
    
    system_prompt = """
        You are a compassionate and empathetic mental health assistant. Your responses should:
        - Analyze the user's input with care and understanding.
        - Respond directly in a comforting manner, tailoring to the specific query, history, and provided guidelines.
        - Incorporate relevant details from guidelines naturally (e.g., coping strategies).
        - Offer gentle, practical coping suggestions where relevant, varying them based on context.
        - Keep the tone warm, supportive, and conversational—vary phrasing to avoid repetition.
        - Do not diagnose, give medical advice, or promise cures. Always suggest professional help if needed (e.g., helplines like beyondblue: 1300 22 4636).
        - Responses should be 60-100 words.
        """
    
    history = []  # List of dicts: [{'role': 'user', 'content': ...}, {'role': 'assistant', 'content': ...}]
    
    print("Welcome to the Mental Health Chatbot. Type 'quit' to exit.")
    
    while True:
        try:
            query = input("You: ").strip()
            if query.lower() in ['quit', 'exit']:
                print("Chatbot: Goodbye! Take care.")
                break
            
            if not query:
                print("Chatbot: Please enter a message.")
                continue
            
            # Get RAG results
            retrieved_docs, post_result = full_rag_workflow(query, method=method)
            
            # Format retrieved docs as context string
            doc_context = "\n".join([f"Doc {i+1} (Sentiment: {doc['sentiment']}, Score: {doc['score']:.4f}): {doc['content']}" for i, doc in enumerate(retrieved_docs)])
            
            
            # Build user prompt (added back the similar post context and instruction)
            user_prompt = f"""
                    User's message: {query}

                    Additional relevant guidelines (use these to inform suggestions if they fit the query):
                    {doc_context}

                    Respond empathetically, drawing from guidelines for specific ideas. If the query involves loss or grief, suggest personalized memorials or support resources. Vary your language from previous responses.
                    """
            
            # Build messages
            messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_prompt}]
            
            # Check token length and truncate history proactively if needed
            # Note: Without local tokenizer, approximate token count or skip; for simplicity, assume API handles it or implement a rough estimator
            # For now, we'll skip precise truncation; API has context limits (e.g., 128k for Llama-3.3)
            def estimate_tokens(text: str) -> int:
                return len(text.split()) * 1.3 + 100  # Rough estimate + buffer

            while sum(estimate_tokens(msg['content']) for msg in messages) > max_context_tokens:
                if len(history) <= 2:
                    logger.warning("Context exceeds limit; proceeding.")
                    break
                history = history[2:]  # Remove oldest pair
                messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_prompt}]
                logger.info(f"Truncated history to {len(history)} entries.")
            
            # Generate response
            response = generate_with_llm(messages, max_tokens=150)
            
            print("Chatbot:", response)
            
            # Append to history
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})
        
        except Exception as e:
            logger.error(f"Error in chatbot loop: {e}")
            print("Chatbot: Sorry, something went wrong. Please try again.")

# Run the chatbot
if __name__ == "__main__":
    run_chatbot(method='multi')  # Change method as needed: 'original', 'multi', 'hyde'