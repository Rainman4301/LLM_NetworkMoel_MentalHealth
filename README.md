\section{LLM\_NetworkModel\_MentalHealth}

This project implements a Large Language Model (LLM)--based framework for \textbf{Topic Modeling}, \textbf{Network Analysis}, and an \textbf{Empathetic Mental Health Chatbot}. It applies advanced Natural Language Processing (NLP) methods to analyse mental health discussions, uncover thematic patterns, and support mental-health research and digital intervention tools.

\subsection{Project Overview}

User-generated content from mental health forums offers valuable insights into how individuals express challenges, seek support, and share coping strategies. This project examines these discussions through three core components:

\begin{itemize}
    \item \textbf{Topic Modeling} — discovering key themes and subthemes across mental-health discourse (e.g., Anxiety, Depression, PTSD, Suicidal Thoughts).
    \item \textbf{Network Analysis} — visualising relationships between subtopics and identifying how challenges cluster or overlap.
    \item \textbf{Mental Health Chatbot} — generating empathetic, context-aware responses using a Retrieval-Augmented Generation (RAG) pipeline.
\end{itemize}

\subsection{Key Features}

\subsubsection{Topic Modeling}

\begin{itemize}
    \item Built using \textbf{BERTopic}, a transformer-based topic modeling framework.
    \item Uses \textbf{MentalBERT} for domain-specific embeddings.
    \item Hyperparameter tuning for UMAP, HDBSCAN, and CountVectorizer to improve topic coherence and diversity.
    \item Evaluates topics using coherence, topic diversity, and silhouette scores.
\end{itemize}

\subsubsection{Network Analysis}

\begin{itemize}
    \item Constructs co-occurrence graphs of subtopics.
    \item Computes modularity, centrality, assortativity, and related metrics.
    \item Generates interactive visualisations that reveal subtopic relationships.
\end{itemize}

\subsubsection{Mental Health Chatbot}

\begin{itemize}
    \item Powered by a \textbf{RAG pipeline} combining retrieval and generation.
    \item Integrates \textbf{ICD--11} knowledge in a careful, non-diagnostic manner.
    \item Uses sentiment analysis to adjust tone and emotional sensitivity.
    \item Supports multiple query-handling modes (\texttt{original}, \texttt{multi}, \texttt{hyde}).
\end{itemize}

\subsubsection{Data Preprocessing}

\begin{itemize}
    \item Removes noise, filler terms, and irrelevant content.
    \item Generates uni-/bi-/trigrams for BERTopic modeling.
    \item Computes SentenceTransformer embeddings for all downstream tasks.
\end{itemize}

\subsubsection{Human-in-the-Loop Evaluation}

\begin{itemize}
    \item Combines metric-based evaluation with expert and manual review.
    \item Uses LLM-assisted refinement to verify and improve topic labels.
\end{itemize}

\subsection{Repository Structure}

\begin{verbatim}
.
├── .vscode/
├── data/
│   ├── AUS_weather/
│   ├── beyondblue_data/
│   ├── network_graph/
│   ├── reddit_data/
├── lib/
│   ├── bindings/
│   ├── tom-select/
│   ├── vis-9.1.2/
├── pic/
├── rag_system/
│   ├── beyondblue_post_data/
│   ├── ICD-11_Data/
│   ├── general/
├── topic_modeling_result/
├── chatbot_local.py
├── chatbot_API.py
├── hypertune.ipynb
├── data_analysis.ipynb
├── data_collection.ipynb
├── README.md
├── requirement.txt
\end{verbatim}

\subsection{Installation}

\subsubsection{Clone the Repository}

\begin{verbatim}
git clone https://github.com/your-username/LLM_NetworkModel_MentalHealth.git
cd LLM_NetworkModel_MentalHealth
\end{verbatim}

\subsubsection{Install Dependencies}

\begin{verbatim}
pip install -r requirement.txt
\end{verbatim}

\subsubsection{Download Additional Resources}

\begin{itemize}
    \item \textbf{ICD--11 Data}: Set environment variables
    \texttt{ICD\_CLIENT\_ID} and \texttt{ICD\_CLIENT\_SECRET}.
    \item \textbf{MentalBERT}: Download the \texttt{mental/mental-bert-base-uncased} model.
\end{itemize}

\subsection{Usage}

\subsubsection{Topic Modeling}

Run \texttt{hypertune.ipynb} to train BERTopic and optimise hyperparameters.

\subsubsection{Network Analysis}

Use \texttt{data_analysis.ipynb} to construct and visualise subtopic networks.

\subsubsection{Mental Health Chatbot}

\paragraph{Local Mode}
\begin{verbatim}
python chatbot_local.py
\end{verbatim}

\paragraph{API Mode}
\begin{verbatim}
python chatbot_API.py
\end{verbatim}

\subsection{Results}

\subsubsection{Topic Modeling}
\begin{itemize}
    \item Extracted key topics such as Anxiety, Depression, PTSD, and Suicidal Thoughts.
    \item Achieved strong coherence, diversity, and silhouette scores.
\end{itemize}

\subsubsection{Network Analysis}
\begin{itemize}
    \item Revealed central subtopics including ``coping strategies'' and ``emotional support''.
    \item Node-link diagrams visualise overlapping stressors and condition-specific patterns.
\end{itemize}

\subsubsection{Chatbot}
\begin{itemize}
    \item Provides empathetic, context-aware responses.
    \item Uses ICD--11 information for non-diagnostic, supportive suggestions.
\end{itemize}

\subsection{Contributing}

Contributions are welcome. To contribute:

\begin{enumerate}
    \item Fork the repository.
    \item Create a new branch.
    \item Submit a pull request.
\end{enumerate}

\subsection{License}

This project is released under the \textbf{MIT License}. See the \texttt{LICENSE} file for details.

\subsection{Acknowledgements}

\begin{itemize}
    \item MentalBERT for domain-specific embeddings.
    \item BERTopic for topic modeling tools.
    \item Beyond Blue for mental-health forum data.
    \item ICD--11 API for structured mental-health information.
\end{itemize}
