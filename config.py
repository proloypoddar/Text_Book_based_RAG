"""
Configuration file for Bengali RAG Application
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "gpt-3.5-turbo"

# Vector Database Configuration
VECTOR_DB_PATH = "./vector_db"
COLLECTION_NAME = "bengali_literature"

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CHUNKS_TO_RETRIEVE = 5

# Memory Configuration
SHORT_TERM_MEMORY_SIZE = 10  # Number of recent conversations to keep
LONG_TERM_MEMORY_ENABLED = True

# Language Detection
SUPPORTED_LANGUAGES = ["en", "bn"]
DEFAULT_LANGUAGE = "bn"

# Text Processing
BENGALI_STOPWORDS = [
    "এবং", "বা", "কিন্তু", "তবে", "যদি", "তাহলে", "কারণ", "যেহেতু",
    "অথচ", "তথাপি", "সুতরাং", "অতএব", "কিংবা", "অথবা", "না", "নয়",
    "আর", "ও", "এর", "এই", "সেই", "ওই", "যে", "যা", "যার", "যাকে",
    "যাদের", "কে", "কী", "কোন", "কোথায়", "কখন", "কেন", "কীভাবে"
]

# UI Configuration
APP_TITLE = "Bengali Literature RAG System"
APP_DESCRIPTION = "Ask questions about 'অপরিচিতা' by Rabindranath Tagore in Bengali or English"
