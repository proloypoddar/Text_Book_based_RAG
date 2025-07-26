"""
Command Line Interface for Bengali RAG Application
"""
import os
import sys
from rag_system import BengaliRAGSystem

class BengaliRAGCLI:
    def __init__(self):
        self.rag_system = None
        self.initialized = False
    
    def initialize(self):
        """Initialize the RAG system"""
        print("🚀 Initializing Bengali RAG System...")
        try:
            self.rag_system = BengaliRAGSystem()
            
            if os.path.exists("organized_content.json"):
                print("📚 Loading knowledge base...")
                count = self.rag_system.initialize_knowledge_base("organized_content.json")
                print(f"✅ System initialized with {count} document chunks")
                self.initialized = True
            else:
                print("❌ Error: organized_content.json not found!")
                print("Please ensure the knowledge base file is in the current directory.")
                return False
        except Exception as e:
            print(f"❌ Error initializing system: {str(e)}")
            return False
        
        return True
    
    def show_help(self):
        """Show help information"""
        help_text = """
🔥 Bengali RAG System - Command Line Interface

Available Commands:
  help, h          - Show this help message
  stats, s         - Show system statistics
  character <name> - Get character information
  meaning <word>   - Get word meaning
  clear, c         - Clear conversation history
  save             - Save session data
  quit, q, exit    - Exit the application

Sample Questions (Bengali):
  অনুপমের চরিত্র সম্পর্কে বলুন
  কল্যাণী কেন বিয়ে করতে চায়নি?
  গল্পের মূল বিষয়বস্তু কী?

Sample Questions (English):
  Tell me about Anupam's character
  Why didn't Kalyani want to get married?
  What is the main theme of the story?

Tips:
  - Ask questions in Bengali or English
  - Be specific about characters, events, or themes
  - Use conversation context for follow-up questions
        """
        print(help_text)
    
    def show_stats(self):
        """Show system statistics"""
        if not self.initialized:
            print("❌ System not initialized")
            return
        
        stats = self.rag_system.get_system_stats()
        
        print("\n📊 System Statistics:")
        print(f"📚 Total Documents: {stats['vector_store']['total_documents']}")
        print(f"🗂️  Content Types: {len(stats['vector_store']['content_types'])}")
        print(f"💬 Conversations: {stats['memory']['conversation_count']}")
        print(f"🧠 Query Patterns: {stats['memory']['query_patterns_count']}")
        print(f"🌐 Preferred Language: {stats['memory']['preferred_language']}")
        
        print("\n📋 Content Distribution:")
        for content_type, count in stats['vector_store']['content_types'].items():
            print(f"  {content_type}: {count}")
    
    def get_character_info(self, character_name):
        """Get character information"""
        if not self.initialized:
            print("❌ System not initialized")
            return
        
        print(f"🔍 Getting information about: {character_name}")
        result = self.rag_system.get_character_info(character_name)
        print(f"\n👤 Character Information:")
        print(result['response'])
    
    def get_word_meaning(self, word):
        """Get word meaning"""
        if not self.initialized:
            print("❌ System not initialized")
            return
        
        print(f"🔍 Getting meaning of: {word}")
        result = self.rag_system.get_word_meaning(word)
        print(f"\n📚 Word Meaning:")
        print(result['response'])
    
    def process_query(self, query):
        """Process a user query"""
        if not self.initialized:
            print("❌ System not initialized")
            return
        
        print("🤔 Processing your question...")
        try:
            result = self.rag_system.process_query(query)
            
            print(f"\n🤖 Answer:")
            print(result['response'])
            
            print(f"\n📄 Sources used: {len(result['retrieved_chunks'])} document chunks")
            print(f"🌐 Detected language: {result['language']}")
            
        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
    
    def clear_history(self):
        """Clear conversation history"""
        if not self.initialized:
            print("❌ System not initialized")
            return
        
        self.rag_system.memory_manager.conversation_memory.clear_memory()
        print("🧹 Conversation history cleared")
    
    def save_session(self):
        """Save session data"""
        if not self.initialized:
            print("❌ System not initialized")
            return
        
        self.rag_system.memory_manager.save_all_memory()
        print("💾 Session data saved")
    
    def run(self):
        """Main CLI loop"""
        print("📚 Bengali Literature RAG System")
        print("Ask questions about 'অপরিচিতা' by Rabindranath Tagore")
        print("Type 'help' for commands or 'quit' to exit\n")
        
        # Initialize system
        if not self.initialize():
            return
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'q', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() in ['help', 'h']:
                    self.show_help()
                
                elif user_input.lower() in ['stats', 's']:
                    self.show_stats()
                
                elif user_input.lower().startswith('character '):
                    character_name = user_input[10:].strip()
                    if character_name:
                        self.get_character_info(character_name)
                    else:
                        print("❌ Please specify a character name")
                
                elif user_input.lower().startswith('meaning '):
                    word = user_input[8:].strip()
                    if word:
                        self.get_word_meaning(word)
                    else:
                        print("❌ Please specify a word")
                
                elif user_input.lower() in ['clear', 'c']:
                    self.clear_history()
                
                elif user_input.lower() == 'save':
                    self.save_session()
                
                else:
                    # Process as a regular query
                    self.process_query(user_input)
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {str(e)}")

def main():
    """Main function"""
    cli = BengaliRAGCLI()
    cli.run()

if __name__ == "__main__":
    main()
