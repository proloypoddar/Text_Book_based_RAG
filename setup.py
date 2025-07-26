"""
Setup script for Bengali RAG Application
"""
import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating .env file...")
        with open(env_file, 'w') as f:
            f.write("# OpenAI API Configuration\n")
            f.write("OPENAI_API_KEY=your-openai-api-key-here\n")
            f.write("\n# Optional: Custom model settings\n")
            f.write("# LLM_MODEL=gpt-3.5-turbo\n")
            f.write("# EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2\n")
        print("✅ .env file created. Please add your OpenAI API key.")
    else:
        print("✅ .env file already exists")

def check_knowledge_base():
    """Check if knowledge base file exists"""
    kb_file = Path("organized_content.json")
    if not kb_file.exists():
        print("⚠️  Warning: organized_content.json not found")
        print("Please ensure the knowledge base file is in the project directory")
        return False
    print("✅ Knowledge base file found")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ["vector_db", "logs", "sessions"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    required_modules = [
        "langchain",
        "openai", 
        "chromadb",
        "sentence_transformers",
        "streamlit",
        "pandas",
        "numpy"
    ]
    
    failed_imports = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("✅ All imports successful")
    return True

def run_basic_test():
    """Run a basic system test"""
    print("🔬 Running basic system test...")
    try:
        from config import OPENAI_API_KEY, EMBEDDING_MODEL
        from text_preprocessor import BengaliTextPreprocessor
        
        # Test text preprocessor
        preprocessor = BengaliTextPreprocessor()
        test_text = "এটি একটি পরীক্ষা।"
        cleaned = preprocessor.clean_text(test_text)
        print(f"  ✅ Text preprocessing: '{test_text}' → '{cleaned}'")
        
        # Check API key
        if OPENAI_API_KEY == "your-openai-api-key-here":
            print("  ⚠️  OpenAI API key not configured")
        else:
            print("  ✅ OpenAI API key configured")
        
        print("✅ Basic system test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic system test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Bengali RAG System Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create .env file
    create_env_file()
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create directories
    create_directories()
    
    # Test imports
    if not test_imports():
        return False
    
    # Check knowledge base
    kb_exists = check_knowledge_base()
    
    # Run basic test
    if not run_basic_test():
        return False
    
    print("\n" + "=" * 40)
    print("🎉 Setup completed successfully!")
    
    if not kb_exists:
        print("\n⚠️  Next steps:")
        print("1. Add your knowledge base file (organized_content.json)")
        print("2. Configure your OpenAI API key in .env file")
    else:
        print("\n🚀 Ready to run!")
        print("Web interface: streamlit run app.py")
        print("CLI interface: python cli_app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
