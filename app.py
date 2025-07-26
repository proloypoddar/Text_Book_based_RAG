"""
Streamlit Web Interface for Bengali RAG Application
"""
import streamlit as st
import json
import os
from datetime import datetime
from rag_system import BengaliRAGSystem
from config import APP_TITLE, APP_DESCRIPTION

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_system():
    """Initialize the RAG system"""
    if not st.session_state.initialized:
        with st.spinner("Initializing Bengali RAG System..."):
            try:
                st.session_state.rag_system = BengaliRAGSystem()
                
                # Check if knowledge base exists
                if os.path.exists("organized_content.json"):
                    st.session_state.rag_system.initialize_knowledge_base("organized_content.json")
                    st.session_state.initialized = True
                    st.success("✅ System initialized successfully!")
                else:
                    st.error("❌ Knowledge base file 'organized_content.json' not found!")
                    st.info("Please ensure the organized_content.json file is in the same directory.")
                    return False
            except Exception as e:
                st.error(f"❌ Error initializing system: {str(e)}")
                return False
    return True

def main():
    """Main application"""
    st.title(APP_TITLE)
    st.markdown(APP_DESCRIPTION)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Controls")
        
        # Initialize button
        if st.button("🚀 Initialize System", type="primary"):
            st.session_state.initialized = False
            initialize_system()
        
        # System status
        if st.session_state.initialized:
            st.success("✅ System Ready")
            
            # System stats
            if st.button("📊 Show System Stats"):
                stats = st.session_state.rag_system.get_system_stats()
                st.json(stats)
        else:
            st.warning("⚠️ System Not Initialized")
        
        st.divider()
        
        # Language selection
        st.header("🌐 Language Options")
        query_language = st.selectbox(
            "Select Query Language:",
            ["Auto-detect", "Bengali (বাংলা)", "English"],
            index=0
        )
        
        # Quick actions
        st.header("⚡ Quick Actions")
        
        if st.button("🧹 Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("💾 Save Session"):
            if st.session_state.rag_system:
                st.session_state.rag_system.memory_manager.save_all_memory()
                st.success("Session saved!")
    
    # Main content area
    if not st.session_state.initialized:
        st.info("👆 Please click 'Initialize System' in the sidebar to get started.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🔍 Search", "📖 Browse", "❓ Help"])
    
    with tab1:
        chat_interface()
    
    with tab2:
        search_interface()
    
    with tab3:
        browse_interface()
    
    with tab4:
        help_interface()

def chat_interface():
    """Chat interface"""
    st.header("💬 Ask Questions About অপরিচিতা")
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**👤 You ({chat['timestamp']}):**")
            st.markdown(chat['query'])
            
            st.markdown("**🤖 Assistant:**")
            st.markdown(chat['response'])
            
            # Show retrieved chunks in expander
            if chat.get('retrieved_chunks'):
                with st.expander(f"📄 Source Documents ({len(chat['retrieved_chunks'])} chunks)"):
                    for j, chunk in enumerate(chat['retrieved_chunks'], 1):
                        st.markdown(f"**Chunk {j}:**")
                        st.text(chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'])
                        st.caption(f"Type: {chunk.get('metadata', {}).get('type', 'unknown')}")
            
            st.divider()
    
    # Query input
    query = st.text_area(
        "Ask your question in Bengali or English:",
        placeholder="Example: অনুপমের চরিত্র সম্পর্কে বলুন / Tell me about Anupam's character",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("🚀 Ask", type="primary", disabled=not query.strip()):
            if query.strip():
                process_query(query.strip())
    
    with col2:
        use_context = st.checkbox("Use conversation context", value=True)

def search_interface():
    """Search interface for specific queries"""
    st.header("🔍 Advanced Search")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Character Information")
        character_name = st.selectbox(
            "Select Character:",
            ["অনুপম", "কল্যাণী", "মামা", "শম্ভুনাথ সেন", "হরিশ", "বিনুদাদা"]
        )
        
        if st.button("Get Character Info"):
            if st.session_state.rag_system:
                result = st.session_state.rag_system.get_character_info(character_name)
                st.markdown("**Character Information:**")
                st.markdown(result['response'])
    
    with col2:
        st.subheader("📚 Word Meanings")
        word = st.text_input("Enter word to get meaning:")
        
        if st.button("Get Meaning") and word:
            if st.session_state.rag_system:
                result = st.session_state.rag_system.get_word_meaning(word)
                st.markdown("**Word Meaning:**")
                st.markdown(result['response'])
    
    st.divider()
    
    # Advanced search options
    st.subheader("🎯 Filtered Search")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        content_type = st.selectbox(
            "Content Type:",
            ["All", "Story", "Characters", "Questions", "Word Meanings"]
        )
    
    with col2:
        search_query = st.text_input("Search Query:")
    
    with col3:
        max_results = st.slider("Max Results:", 1, 10, 5)
    
    if st.button("🔍 Advanced Search") and search_query:
        if st.session_state.rag_system:
            # Map content type to filter
            type_map = {
                "Story": "story",
                "Characters": "character", 
                "Questions": "mcq",
                "Word Meanings": "word_meaning"
            }
            
            if content_type == "All":
                chunks = st.session_state.rag_system.vector_store.similarity_search(search_query, max_results)
            else:
                chunks = st.session_state.rag_system.vector_store.search_by_type(
                    search_query, type_map[content_type], max_results
                )
            
            st.markdown(f"**Found {len(chunks)} results:**")
            for i, chunk in enumerate(chunks, 1):
                with st.expander(f"Result {i} - {chunk.get('metadata', {}).get('type', 'unknown')}"):
                    st.markdown(chunk['content'])
                    st.caption(f"Similarity: {1 - chunk.get('distance', 0):.2f}")

def browse_interface():
    """Browse interface for exploring content"""
    st.header("📖 Browse Content")
    
    if not st.session_state.rag_system:
        st.warning("System not initialized")
        return
    
    # Get system stats
    stats = st.session_state.rag_system.get_system_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documents", stats['vector_store']['total_documents'])
    
    with col2:
        st.metric("Content Types", len(stats['vector_store']['content_types']))
    
    with col3:
        st.metric("Conversations", stats['memory']['conversation_count'])
    
    # Content type breakdown
    st.subheader("📊 Content Distribution")
    content_types = stats['vector_store']['content_types']
    
    for content_type, count in content_types.items():
        st.markdown(f"**{content_type.title()}:** {count} documents")
    
    # Memory stats
    st.subheader("🧠 Memory Statistics")
    memory_stats = stats['memory']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Query Patterns:** {memory_stats['query_patterns_count']}")
        st.markdown(f"**Preferred Language:** {memory_stats['preferred_language']}")
    
    with col2:
        st.markdown(f"**Document Access Records:** {memory_stats['document_access_records']}")
        st.markdown(f"**Session ID:** {memory_stats['session_id']}")

def help_interface():
    """Help and examples interface"""
    st.header("❓ Help & Examples")
    
    st.subheader("🎯 Sample Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Bengali Questions:**")
        bengali_examples = [
            "অনুপমের চরিত্র সম্পর্কে বলুন",
            "কল্যাণী কেন বিয়ে করতে চায়নি?",
            "গল্পের মূল বিষয়বস্তু কী?",
            "শম্ভুনাথ সেনের চরিত্র বিশ্লেষণ করুন",
            "অপরিচিতা গল্পের শিক্ষা কী?"
        ]
        
        for example in bengali_examples:
            if st.button(f"📝 {example}", key=f"bn_{example}"):
                process_query(example)
    
    with col2:
        st.markdown("**English Questions:**")
        english_examples = [
            "Tell me about Anupam's character",
            "Why didn't Kalyani want to get married?",
            "What is the main theme of the story?",
            "Analyze Shambhunath Sen's character",
            "What is the moral lesson of Aparichita?"
        ]
        
        for example in english_examples:
            if st.button(f"📝 {example}", key=f"en_{example}"):
                process_query(example)
    
    st.subheader("💡 Tips for Better Results")
    
    tips = [
        "🎯 **Be specific**: Ask about particular characters, events, or themes",
        "🌐 **Use either language**: The system understands both Bengali and English",
        "📚 **Ask about context**: Request explanations of cultural or historical references",
        "🔍 **Try different phrasings**: If you don't get the answer you want, rephrase your question",
        "💬 **Use conversation context**: Enable context to have follow-up conversations"
    ]
    
    for tip in tips:
        st.markdown(tip)

def process_query(query: str):
    """Process a user query"""
    if not st.session_state.rag_system:
        st.error("System not initialized")
        return
    
    with st.spinner("Processing your question..."):
        try:
            result = st.session_state.rag_system.process_query(query)
            
            # Add to chat history
            chat_entry = {
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'query': query,
                'response': result['response'],
                'retrieved_chunks': result['retrieved_chunks'],
                'language': result['language']
            }
            
            st.session_state.chat_history.append(chat_entry)
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    # Auto-initialize on startup
    if not st.session_state.initialized:
        initialize_system()
    
    main()
