# 🤖 An Q&A NLP Chatbot

A sophisticated **Retrieval-Augmented Generation (RAG)** based Question-Answering chatbot that leverages vector similarity search and fine-tuned BERT models to provide accurate, context-aware answers from your document repository.

## 📋 Overview

This project implements an intelligent Q&A system combining:
- **Vector Embeddings**: Sentence Transformers for semantic document understanding
- **FAISS Indexing**: Fast approximate nearest neighbor search for efficient retrieval
- **Extractive QA**: Fine-tuned BERT model for answer extraction
- **Streamlit UI**: Interactive web interface for seamless user interaction

## ✨ Features

- 🔍 **Dense Vector Search**: Retrieves semantically similar documents using FAISS
- 🧠 **BERT-Based Extraction**: Extracts precise answers from retrieved context
- 📊 **Confidence Scoring**: Provides confidence metrics for extracted answers
- 🎯 **Configurable Retrieval**: Adjustable document extraction threshold (k parameter)
- 🎨 **Intuitive UI**: Clean, responsive Streamlit interface
- 📚 **Multi-Document Support**: Query across multiple document chunks with source tracking

## 🛠️ Tech Stack

- **Python** 3.x
- **Streamlit** - Web framework for interactive UI
- **PyTorch** - Deep learning framework
- **Transformers** - Pre-trained NLP models (BERT)
- **Sentence Transformers** - Advanced semantic embeddings
- **FAISS** - Efficient similarity search
- **Pandas & NumPy** - Data processing
- **scikit-learn** - Machine learning utilities

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Mayam99/An_Q-A_NLP-Chatbot.git
cd An_Q-A_NLP-Chatbot
