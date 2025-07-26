"""
Main RAG System for Bengali Literature
Combines retrieval, generation, and memory management
"""
import os
import re
from typing import List, Dict, Any, Optional, Tuple
import openai
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from vector_store import BengaliVectorStore
from memory_manager import MemoryManager
from text_preprocessor import BengaliTextPreprocessor
from config import OPENAI_API_KEY, LLM_MODEL, MAX_CHUNKS_TO_RETRIEVE

class BengaliRAGSystem:
    def __init__(self):
        # Initialize components
        self.vector_store = BengaliVectorStore()
        self.memory_manager = MemoryManager()
        self.preprocessor = BengaliTextPreprocessor()
        
        # Initialize OpenAI
        openai.api_key = OPENAI_API_KEY
        self.llm = ChatOpenAI(
            model_name=LLM_MODEL,
            temperature=0.7,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Language detection patterns
        self.bengali_pattern = re.compile(r'[\u0980-\u09FF]')
        self.english_pattern = re.compile(r'[a-zA-Z]')
    
    def detect_language(self, text: str) -> str:
        """Detect if text is primarily Bengali or English"""
        bengali_chars = len(self.bengali_pattern.findall(text))
        english_chars = len(self.english_pattern.findall(text))
        
        if bengali_chars > english_chars:
            return 'bn'
        elif english_chars > 0:
            return 'en'
        else:
            return 'bn'  # Default to Bengali
    
    def translate_query_if_needed(self, query: str, target_language: str = 'bn') -> str:
        """Translate query if needed for better retrieval"""
        query_lang = self.detect_language(query)
        
        if query_lang != target_language:
            # Simple translation mapping for common terms
            translation_map = {
                'character': 'চরিত্র',
                'story': 'গল্প',
                'plot': 'কাহিনী',
                'author': 'লেখক',
                'meaning': 'অর্থ',
                'question': 'প্রশ্ন',
                'answer': 'উত্তর',
                'Anupam': 'অনুপম',
                'Kalyani': 'কল্যাণী',
                'uncle': 'মামা',
                'marriage': 'বিয়ে',
                'wedding': 'বিয়ে',
                'Rabindranath': 'রবীন্দ্রনাথ',
                'Tagore': 'ঠাকুর'
            }
            
            translated_query = query
            for eng, ben in translation_map.items():
                translated_query = re.sub(r'\b' + eng + r'\b', ben, translated_query, flags=re.IGNORECASE)
            
            return translated_query
        
        return query
    
    def retrieve_relevant_chunks(self, query: str, k: int = MAX_CHUNKS_TO_RETRIEVE) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for a query"""
        # Clean and preprocess query
        cleaned_query = self.preprocessor.clean_text(query)
        
        # Translate query for better retrieval if needed
        translated_query = self.translate_query_if_needed(cleaned_query, 'bn')
        
        # Get relevant chunks
        chunks = self.vector_store.similarity_search(translated_query, k=k)
        
        # If no good results with translated query, try original
        if not chunks or (chunks and chunks[0].get('distance', 0) > 0.8):
            chunks = self.vector_store.similarity_search(cleaned_query, k=k)
        
        return chunks
    
    def create_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Create context string from retrieved chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            content = chunk['content']
            metadata = chunk.get('metadata', {})
            chunk_type = metadata.get('type', 'unknown')
            
            # Format based on chunk type
            if chunk_type == 'story':
                context_parts.append(f"গল্পের অংশ {i}: {content}")
            elif chunk_type == 'character':
                context_parts.append(f"চরিত্র তথ্য {i}: {content}")
            elif chunk_type == 'mcq':
                context_parts.append(f"প্রশ্ন ও উত্তর {i}: {content}")
            elif chunk_type == 'word_meaning':
                context_parts.append(f"শব্দার্থ {i}: {content}")
            else:
                context_parts.append(f"তথ্য {i}: {content}")
        
        return "\n\n".join(context_parts)
    
    def create_system_prompt(self, query_language: str) -> str:
        """Create system prompt based on query language"""
        if query_language == 'en':
            return """You are a helpful assistant specialized in Bengali literature, specifically Rabindranath Tagore's short story "অপরিচিতা" (Aparichita/The Stranger). 

Your role:
- Answer questions about the story, characters, plot, themes, and literary analysis
- Provide accurate information based on the given context
- Respond in English when the user asks in English
- Respond in Bengali when the user asks in Bengali
- Be educational and helpful for students
- Include relevant quotes or examples when appropriate
- If you don't know something from the context, say so clearly

Context will be provided from the story and related educational materials."""
        else:
            return """আপনি বাংলা সাহিত্যের একজন সহায়ক সহকারী, বিশেষত রবীন্দ্রনাথ ঠাকুরের "অপরিচিতা" গল্পের বিশেষজ্ঞ।

আপনার ভূমিকা:
- গল্প, চরিত্র, কাহিনী, বিষয়বস্তু এবং সাহিত্য বিশ্লেষণ সম্পর্কে প্রশ্নের উত্তর দিন
- প্রদত্ত প্রসঙ্গের ভিত্তিতে সঠিক তথ্য প্রদান করুন
- ব্যবহারকারী ইংরেজিতে জিজ্ঞাসা করলে ইংরেজিতে উত্তর দিন
- ব্যবহারকারী বাংলায় জিজ্ঞাসা করলে বাংলায় উত্তর দিন
- শিক্ষার্থীদের জন্য শিক্ষামূলক এবং সহায়ক হন
- প্রাসঙ্গিক উদ্ধৃতি বা উদাহরণ অন্তর্ভুক্ত করুন
- প্রসঙ্গ থেকে কিছু না জানলে স্পষ্টভাবে বলুন

গল্প এবং সংশ্লিষ্ট শিক্ষামূলক উপকরণ থেকে প্রসঙ্গ প্রদান করা হবে।"""
    
    def generate_response(self, query: str, context: str, conversation_context: str = "") -> str:
        """Generate response using LLM"""
        query_language = self.detect_language(query)
        system_prompt = self.create_system_prompt(query_language)
        
        # Create the prompt
        if conversation_context:
            user_prompt = f"""Previous conversation context:
{conversation_context}

Current question: {query}

Relevant information from documents:
{context}

Please provide a comprehensive answer based on the given context."""
        else:
            user_prompt = f"""Question: {query}

Relevant information from documents:
{context}

Please provide a comprehensive answer based on the given context."""
        
        # Generate response
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm(messages)
            return response.content
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            if query_language == 'en':
                return f"I apologize, but I encountered an error while generating the response. Please try again. Error: {error_msg}"
            else:
                return f"দুঃখিত, উত্তর তৈরি করতে একটি ত্রুটি হয়েছে। অনুগ্রহ করে আবার চেষ্টা করুন। ত্রুটি: {error_msg}"
    
    def process_query(self, query: str, use_conversation_context: bool = True) -> Dict[str, Any]:
        """Process a complete query through the RAG pipeline"""
        # Get conversation context if requested
        conversation_context = ""
        if use_conversation_context:
            context_data = self.memory_manager.get_context_for_query(query)
            conversation_context = context_data.get('conversation_context', '')
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve_relevant_chunks(query)
        
        # Create context from chunks
        document_context = self.create_context_from_chunks(retrieved_chunks)
        
        # Generate response
        response = self.generate_response(query, document_context, conversation_context)
        
        # Detect language for memory
        query_language = self.detect_language(query)
        
        # Add to memory
        self.memory_manager.add_interaction(query, response, retrieved_chunks, query_language)
        
        return {
            'query': query,
            'response': response,
            'retrieved_chunks': retrieved_chunks,
            'language': query_language,
            'context_used': document_context,
            'conversation_context': conversation_context
        }
    
    def get_character_info(self, character_name: str) -> Dict[str, Any]:
        """Get specific character information"""
        chunks = self.vector_store.get_character_info(character_name)
        context = self.create_context_from_chunks(chunks)
        
        query_language = self.detect_language(character_name)
        if query_language == 'en':
            query = f"Tell me about the character {character_name}"
        else:
            query = f"{character_name} চরিত্র সম্পর্কে বলুন"
        
        response = self.generate_response(query, context)
        
        return {
            'character': character_name,
            'response': response,
            'retrieved_chunks': chunks
        }
    
    def get_word_meaning(self, word: str) -> Dict[str, Any]:
        """Get word meaning"""
        chunks = self.vector_store.get_word_meaning(word)
        context = self.create_context_from_chunks(chunks)
        
        query_language = self.detect_language(word)
        if query_language == 'en':
            query = f"What is the meaning of {word}?"
        else:
            query = f"{word} শব্দের অর্থ কী?"
        
        response = self.generate_response(query, context)
        
        return {
            'word': word,
            'response': response,
            'retrieved_chunks': chunks
        }
    
    def initialize_knowledge_base(self, json_file_path: str):
        """Initialize the knowledge base from JSON file"""
        print("Initializing knowledge base...")
        count = self.vector_store.process_and_store_documents(json_file_path)
        print(f"Knowledge base initialized with {count} document chunks")
        return count
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        vector_stats = self.vector_store.get_collection_stats()
        memory_stats = self.memory_manager.get_memory_stats()
        
        return {
            'vector_store': vector_stats,
            'memory': memory_stats,
            'system_status': 'active'
        }
