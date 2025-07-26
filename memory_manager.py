"""
Memory Management Module for Bengali RAG Application
Handles short-term (conversation) and long-term (document) memory
"""
import json
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import deque
from config import SHORT_TERM_MEMORY_SIZE

class ConversationMemory:
    """Manages short-term conversation memory"""
    
    def __init__(self, max_size: int = SHORT_TERM_MEMORY_SIZE):
        self.max_size = max_size
        self.conversations = deque(maxlen=max_size)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_conversation(self, user_query: str, assistant_response: str, 
                        retrieved_chunks: List[Dict[str, Any]] = None,
                        language: str = "bn"):
        """Add a conversation turn to memory"""
        conversation = {
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'assistant_response': assistant_response,
            'language': language,
            'retrieved_chunks': retrieved_chunks or [],
            'session_id': self.session_id
        }
        self.conversations.append(conversation)
    
    def get_recent_conversations(self, n: int = 3) -> List[Dict[str, Any]]:
        """Get the n most recent conversations"""
        return list(self.conversations)[-n:] if n <= len(self.conversations) else list(self.conversations)
    
    def get_conversation_context(self, n: int = 3) -> str:
        """Get formatted conversation context for the LLM"""
        recent = self.get_recent_conversations(n)
        context_parts = []
        
        for conv in recent:
            context_parts.append(f"User: {conv['user_query']}")
            context_parts.append(f"Assistant: {conv['assistant_response']}")
        
        return "\n".join(context_parts)
    
    def search_conversation_history(self, query: str) -> List[Dict[str, Any]]:
        """Search through conversation history for relevant past discussions"""
        relevant_conversations = []
        query_lower = query.lower()
        
        for conv in self.conversations:
            # Simple keyword matching - can be enhanced with semantic search
            if (query_lower in conv['user_query'].lower() or 
                query_lower in conv['assistant_response'].lower()):
                relevant_conversations.append(conv)
        
        return relevant_conversations
    
    def save_session(self, filepath: str = None):
        """Save conversation session to file"""
        if not filepath:
            filepath = f"conversation_session_{self.session_id}.json"
        
        session_data = {
            'session_id': self.session_id,
            'conversations': list(self.conversations),
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
    
    def load_session(self, filepath: str):
        """Load conversation session from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        self.session_id = session_data['session_id']
        self.conversations = deque(session_data['conversations'], maxlen=self.max_size)
    
    def clear_memory(self):
        """Clear all conversation memory"""
        self.conversations.clear()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

class DocumentMemory:
    """Manages long-term document memory and retrieval patterns"""
    
    def __init__(self):
        self.query_patterns = {}  # Track common query patterns
        self.document_access_frequency = {}  # Track which documents are accessed most
        self.user_preferences = {}  # Track user language and topic preferences
    
    def record_query_pattern(self, query: str, language: str, retrieved_docs: List[Dict[str, Any]]):
        """Record query patterns for optimization"""
        query_key = query.lower().strip()
        
        if query_key not in self.query_patterns:
            self.query_patterns[query_key] = {
                'count': 0,
                'languages': {},
                'common_doc_types': {},
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat()
            }
        
        pattern = self.query_patterns[query_key]
        pattern['count'] += 1
        pattern['last_seen'] = datetime.now().isoformat()
        pattern['languages'][language] = pattern['languages'].get(language, 0) + 1
        
        # Track document types retrieved for this query
        for doc in retrieved_docs:
            doc_type = doc.get('metadata', {}).get('type', 'unknown')
            pattern['common_doc_types'][doc_type] = pattern['common_doc_types'].get(doc_type, 0) + 1
    
    def record_document_access(self, doc_id: str, doc_type: str):
        """Record document access frequency"""
        if doc_id not in self.document_access_frequency:
            self.document_access_frequency[doc_id] = {
                'count': 0,
                'type': doc_type,
                'first_accessed': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat()
            }
        
        self.document_access_frequency[doc_id]['count'] += 1
        self.document_access_frequency[doc_id]['last_accessed'] = datetime.now().isoformat()
    
    def get_popular_documents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently accessed documents"""
        sorted_docs = sorted(
            self.document_access_frequency.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        return sorted_docs[:limit]
    
    def get_query_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get query suggestions based on historical patterns"""
        partial_lower = partial_query.lower()
        suggestions = []
        
        for query, pattern in self.query_patterns.items():
            if partial_lower in query and pattern['count'] > 1:
                suggestions.append(query)
        
        # Sort by frequency
        suggestions.sort(key=lambda q: self.query_patterns[q]['count'], reverse=True)
        return suggestions[:limit]
    
    def update_user_preferences(self, language: str, topic_type: str):
        """Update user preferences based on usage"""
        if 'language_preference' not in self.user_preferences:
            self.user_preferences['language_preference'] = {}
        if 'topic_preference' not in self.user_preferences:
            self.user_preferences['topic_preference'] = {}
        
        lang_pref = self.user_preferences['language_preference']
        topic_pref = self.user_preferences['topic_preference']
        
        lang_pref[language] = lang_pref.get(language, 0) + 1
        topic_pref[topic_type] = topic_pref.get(topic_type, 0) + 1
    
    def get_preferred_language(self) -> str:
        """Get user's preferred language based on usage"""
        if not self.user_preferences.get('language_preference'):
            return 'bn'  # Default to Bengali
        
        lang_prefs = self.user_preferences['language_preference']
        return max(lang_prefs.items(), key=lambda x: x[1])[0]
    
    def save_memory_data(self, filepath: str = "document_memory.pkl"):
        """Save document memory data"""
        memory_data = {
            'query_patterns': self.query_patterns,
            'document_access_frequency': self.document_access_frequency,
            'user_preferences': self.user_preferences,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(memory_data, f)
    
    def load_memory_data(self, filepath: str = "document_memory.pkl"):
        """Load document memory data"""
        try:
            with open(filepath, 'rb') as f:
                memory_data = pickle.load(f)
            
            self.query_patterns = memory_data.get('query_patterns', {})
            self.document_access_frequency = memory_data.get('document_access_frequency', {})
            self.user_preferences = memory_data.get('user_preferences', {})
            
            print(f"Loaded memory data with {len(self.query_patterns)} query patterns")
        except FileNotFoundError:
            print("No existing memory data found, starting fresh")
        except Exception as e:
            print(f"Error loading memory data: {e}")

class MemoryManager:
    """Combined memory manager for both short-term and long-term memory"""
    
    def __init__(self):
        self.conversation_memory = ConversationMemory()
        self.document_memory = DocumentMemory()
        
        # Load existing document memory
        self.document_memory.load_memory_data()
    
    def add_interaction(self, user_query: str, assistant_response: str,
                       retrieved_chunks: List[Dict[str, Any]], language: str = "bn"):
        """Add a complete interaction to both memory systems"""
        # Add to conversation memory
        self.conversation_memory.add_conversation(
            user_query, assistant_response, retrieved_chunks, language
        )
        
        # Record patterns in document memory
        self.document_memory.record_query_pattern(user_query, language, retrieved_chunks)
        
        # Record document access
        for chunk in retrieved_chunks:
            doc_id = chunk.get('id', 'unknown')
            doc_type = chunk.get('metadata', {}).get('type', 'unknown')
            self.document_memory.record_document_access(doc_id, doc_type)
        
        # Update user preferences
        topic_type = self._infer_topic_type(retrieved_chunks)
        self.document_memory.update_user_preferences(language, topic_type)
    
    def _infer_topic_type(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Infer the main topic type from retrieved chunks"""
        type_counts = {}
        for chunk in retrieved_chunks:
            doc_type = chunk.get('metadata', {}).get('type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        if type_counts:
            return max(type_counts.items(), key=lambda x: x[1])[0]
        return 'unknown'
    
    def get_context_for_query(self, query: str) -> Dict[str, Any]:
        """Get comprehensive context for a query"""
        return {
            'recent_conversations': self.conversation_memory.get_recent_conversations(3),
            'conversation_context': self.conversation_memory.get_conversation_context(3),
            'similar_past_queries': self.conversation_memory.search_conversation_history(query),
            'query_suggestions': self.document_memory.get_query_suggestions(query),
            'preferred_language': self.document_memory.get_preferred_language()
        }
    
    def save_all_memory(self):
        """Save both conversation and document memory"""
        self.conversation_memory.save_session()
        self.document_memory.save_memory_data()
        print("All memory data saved successfully")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        return {
            'conversation_count': len(self.conversation_memory.conversations),
            'query_patterns_count': len(self.document_memory.query_patterns),
            'document_access_records': len(self.document_memory.document_access_frequency),
            'session_id': self.conversation_memory.session_id,
            'preferred_language': self.document_memory.get_preferred_language(),
            'popular_documents': self.document_memory.get_popular_documents(5)
        }
