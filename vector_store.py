"""
Vector Store Module for Bengali RAG Application
Handles document chunking, embedding, and vector database operations
"""
import os
import json
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from text_preprocessor import BengaliTextPreprocessor
from config import (
    EMBEDDING_MODEL, VECTOR_DB_PATH, COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_TO_RETRIEVE
)

class BengaliVectorStore:
    def __init__(self):
        self.preprocessor = BengaliTextPreprocessor()
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=['\n\n', '\n', '।', '!', '?', ' ', '']
        )
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=VECTOR_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(COLLECTION_NAME)
            print(f"Loaded existing collection: {COLLECTION_NAME}")
        except:
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Bengali Literature RAG Collection"}
            )
            print(f"Created new collection: {COLLECTION_NAME}")
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts"""
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """Split text into chunks using LangChain text splitter"""
        chunks = self.text_splitter.split_text(text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata['chunk_id'] = i
            doc_metadata['chunk_size'] = len(chunk)
            
            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))
        
        return documents
    
    def process_and_store_documents(self, json_file_path: str):
        """Process JSON content and store in vector database"""
        print("Processing documents...")
        
        # Preprocess the JSON content
        processed_data = self.preprocessor.preprocess_json_content(json_file_path)
        
        # Create searchable chunks
        chunks = self.preprocessor.create_searchable_chunks(processed_data)
        
        # Prepare data for vector storage
        documents = []
        texts = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            # Create document ID
            doc_id = f"{chunk['type']}_{i}"
            
            # Prepare text content
            content = chunk['content']
            
            # Create metadata
            metadata = chunk['metadata'].copy()
            metadata['type'] = chunk['type']
            metadata['doc_id'] = doc_id
            
            # Add type-specific metadata
            if chunk['type'] == 'story':
                metadata['title'] = chunk['title']
                metadata['section'] = chunk['section']
            elif chunk['type'] == 'mcq':
                metadata['question'] = chunk['question']
                metadata['answer'] = chunk['answer']
            elif chunk['type'] == 'character':
                metadata['character_name'] = chunk['character_name']
            elif chunk['type'] == 'word_meaning':
                metadata['word'] = chunk['word']
                metadata['meaning'] = chunk['meaning']
            
            texts.append(content)
            metadatas.append(metadata)
            ids.append(doc_id)
        
        # Create embeddings
        print(f"Creating embeddings for {len(texts)} chunks...")
        embeddings = self.create_embeddings(texts)
        
        # Store in ChromaDB
        print("Storing in vector database...")
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Successfully stored {len(texts)} document chunks")
        
        # Save processed data for reference
        with open('processed_data.json', 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        return len(texts)
    
    def similarity_search(self, query: str, k: int = MAX_CHUNKS_TO_RETRIEVE, 
                         filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        # Create query embedding
        query_embedding = self.create_embeddings([query])[0]
        
        # Prepare where clause for filtering
        where_clause = filter_metadata if filter_metadata else None
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_clause
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                result = {
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results['distances'] else None,
                    'id': results['ids'][0][i]
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        count = self.collection.count()
        
        # Get sample of metadata to understand content types
        sample_results = self.collection.get(limit=min(100, count))
        
        content_types = {}
        if sample_results['metadatas']:
            for metadata in sample_results['metadatas']:
                content_type = metadata.get('type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
        
        return {
            'total_documents': count,
            'content_types': content_types,
            'collection_name': COLLECTION_NAME
        }
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(COLLECTION_NAME)
            print(f"Deleted collection: {COLLECTION_NAME}")
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def search_by_type(self, query: str, content_type: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for documents of a specific type"""
        filter_metadata = {"type": content_type}
        return self.similarity_search(query, k, filter_metadata)
    
    def get_character_info(self, character_name: str) -> List[Dict[str, Any]]:
        """Get information about a specific character"""
        filter_metadata = {"type": "character"}
        results = self.similarity_search(character_name, k=5, filter_metadata=filter_metadata)
        
        # Also search in story content for character mentions
        story_results = self.search_by_type(character_name, "story", k=3)
        
        return results + story_results
    
    def get_word_meaning(self, word: str) -> List[Dict[str, Any]]:
        """Get meaning of a specific word"""
        filter_metadata = {"type": "word_meaning"}
        return self.similarity_search(word, k=3, filter_metadata=filter_metadata)
    
    def get_story_context(self, query: str) -> List[Dict[str, Any]]:
        """Get story context for a query"""
        return self.search_by_type(query, "story", k=5)
