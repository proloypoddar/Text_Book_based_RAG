# Bengali Literature RAG System 📚

A comprehensive Retrieval-Augmented Generation (RAG) application for Bengali literature, specifically designed for Rabindranath Tagore's short story "অপরিচিতা" (Aparichita/The Stranger). This system supports bilingual queries in Bengali and English with advanced memory management and document retrieval.

## 🌟 Features

- **Bilingual Support**: Ask questions in Bengali (বাংলা) or English
- **Advanced Text Processing**: Specialized Bengali text preprocessing and normalization
- **Smart Memory Management**: 
  - Short-term: Recent conversation context
  - Long-term: Document corpus in vector database
- **Multiple Interfaces**: Web UI (Streamlit) and Command Line Interface
- **Comprehensive Knowledge Base**: Story text, character analysis, Q&A, word meanings
- **Vector Search**: Semantic similarity search using multilingual embeddings
- **Context-Aware Responses**: Uses conversation history for better answers

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Language        │───▶│  Text           │
│ (Bengali/English)│    │  Detection       │    │  Preprocessing  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Response      │◀───│  LLM Generation  │◀───│  Vector Search  │
│   Generation    │    │  (GPT-3.5)       │    │  (ChromaDB)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        ▲                       │
         ▼                        │                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Memory        │    │  Context         │    │  Document       │
│   Management    │    │  Assembly        │    │  Retrieval      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- OpenAI API Key
- Git

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd bengali-rag-system
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Environment Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key-here
```

### Step 5: Prepare Knowledge Base

Ensure `organized_content.json` is in the project root directory. This file contains the preprocessed Bengali literature content.

## 🚀 Usage

### Web Interface (Recommended)

```bash
streamlit run app.py
```

Access the application at `http://localhost:8501`

### Command Line Interface

```bash
python cli_app.py
```

## 📋 Used Tools, Libraries & Packages

### Core RAG Framework
- **LangChain** (0.1.0): RAG pipeline orchestration
- **OpenAI** (1.6.1): Language model for response generation
- **ChromaDB** (0.4.22): Vector database for document storage
- **Sentence Transformers** (2.2.2): Multilingual embeddings

### Bengali Text Processing
- **indic-nlp-library** (0.81): Indic language processing
- **bnlp-toolkit** (3.6.0): Bengali NLP utilities
- **polyglot** (16.7.4): Multilingual text processing

### Data Processing & Utilities
- **pandas** (2.1.4): Data manipulation
- **numpy** (1.24.3): Numerical operations
- **regex** (2023.12.25): Advanced pattern matching
- **unidecode** (1.3.7): Unicode normalization

### Web Interface
- **Streamlit** (1.29.0): Interactive web application
- **Gradio** (4.12.0): Alternative UI framework

### Additional Tools
- **python-dotenv** (1.0.0): Environment variable management
- **tiktoken** (0.5.2): Token counting for OpenAI models

## 💬 Sample Queries and Expected Outputs

### Bengali Queries

#### Query 1: Character Analysis
**Input (Bengali):**
```
অনুপমের চরিত্র সম্পর্কে বলুন
```

**Expected Output:**
```
অনুপম হলো "অপরিচিতা" গল্পের প্রধান পুরুষ চরিত্র এবং কথক। তার বয়স ২৭ বছর এবং তিনি এমএ পাস। 

চরিত্রের বৈশিষ্ট্য:
- দুর্বল ব্যক্তিত্বের অধিকারী
- মায়ের আদেশ মানতে বাধ্য
- সিদ্ধান্ত নিতে অক্ষম
- ভালোমানুষ কিন্তু নিষ্ক্রিয়

পণ্ডিতমশাই তাকে শিমুল ফুল ও মাকাল ফলের সাথে তুলনা করে বিদ্রূপ করতেন। তিনি তামাক খান না এবং অন্তঃপুরের শাসনে চলতে প্রস্তুত।
```

#### Query 2: Plot Question
**Input (Bengali):**
```
কল্যাণী কেন বিয়ে করতে চায়নি?
```

**Expected Output:**
```
কল্যাণী বিয়ে করতে চায়নি কারণ:

১. আত্মমর্যাদাবোধ: তার পিতা শম্ভুনাথ সেনের আত্মসম্মানে আঘাত লেগেছিল
২. যৌতুক প্রথার বিরোধিতা: বিয়ের আগে গহনা পরীক্ষা করার অপমান
৩. দেশসেবার ব্রত: "আমরা দেশের কাজ করি" - এই আদর্শে বিশ্বাসী
৪. স্বাধীনচেতা মনোভাব: নিজের জীবনের সিদ্ধান্ত নিজেই নিতে চান

তিনি দৃঢ়ভাবে বলেছিলেন "আমার বিয়ে হইবে না" এবং "আমি কাহাকেও বিয়ে করিব না"।
```

### English Queries

#### Query 1: Theme Analysis
**Input (English):**
```
What is the main theme of Aparichita?
```

**Expected Output:**
```
The main themes of "Aparichita" (The Stranger) include:

1. **Opposition to Dowry System**: The story critiques the inhuman dowry practice through the humiliation of jewelry inspection before marriage.

2. **Women's Empowerment**: Kalyani represents a strong, independent woman who refuses to accept social injustice and chooses her own path.

3. **Dignity and Self-Respect**: Shambhunath Sen's refusal to give his daughter in marriage after the insult shows the importance of maintaining dignity.

4. **Social Reform**: The story advocates for progressive thinking and resistance against harmful social customs.

5. **Individual Agency**: Both father and daughter exercise their right to make independent decisions against societal pressure.

The story was groundbreaking as it showed the first instance of combined resistance by both men and women against the dowry system in Rabindranath's works.
```

#### Query 2: Character Comparison
**Input (English):**
```
Compare Anupam and Kalyani's characters
```

**Expected Output:**
```
Anupam and Kalyani represent contrasting character types:

**Anupam:**
- Weak-willed and passive
- Dependent on others (especially his uncle)
- Unable to make independent decisions
- Represents traditional, submissive masculinity
- 27 years old, MA graduate, but lacks confidence
- Good-natured but ineffective

**Kalyani:**
- Strong-willed and independent
- Makes her own life decisions
- Confident and articulate
- Represents progressive, empowered femininity
- 16-17 years old but more mature than Anupam
- Dedicated to social service and national work

**Key Contrast:**
While Anupam accepts social norms passively, Kalyani actively challenges them. She embodies the "New Woman" ideal that Rabindranath advocated - educated, independent, and socially conscious.
```

## 🔧 Configuration Options

### Model Configuration
```python
# config.py
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "gpt-3.5-turbo"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CHUNKS_TO_RETRIEVE = 5
```

### Memory Settings
```python
SHORT_TERM_MEMORY_SIZE = 10  # Recent conversations
LONG_TERM_MEMORY_ENABLED = True
```

## 🧠 Memory Management

### Short-Term Memory
- Stores recent conversation turns
- Provides context for follow-up questions
- Configurable memory size (default: 10 conversations)

### Long-Term Memory
- Document corpus stored in vector database
- Query pattern analysis
- User preference tracking
- Document access frequency monitoring

## 📊 System Statistics

Access system statistics through:
- Web UI: Browse tab → System Statistics
- CLI: Type `stats` command
- Programmatically: `rag_system.get_system_stats()`

## 🔍 Advanced Features

### Filtered Search
Search by content type:
- Story sections
- Character information
- Questions and answers
- Word meanings

### Character-Specific Queries
```python
result = rag_system.get_character_info("অনুপম")
```

### Word Meaning Lookup
```python
result = rag_system.get_word_meaning("অপরিচিতা")
```

## 🐛 Troubleshooting

### Common Issues

1. **"System not initialized" error**
   - Ensure `organized_content.json` exists
   - Check OpenAI API key in `.env` file

2. **Import errors**
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

3. **Memory issues**
   - Reduce `CHUNK_SIZE` in config.py
   - Clear conversation history regularly

4. **Slow responses**
   - Check internet connection
   - Verify OpenAI API key validity
   - Reduce `MAX_CHUNKS_TO_RETRIEVE`

## 📝 Development

### Project Structure
```
bengali-rag-system/
├── app.py                 # Streamlit web interface
├── cli_app.py            # Command line interface
├── rag_system.py         # Main RAG system
├── vector_store.py       # Vector database management
├── memory_manager.py     # Memory management
├── text_preprocessor.py  # Bengali text processing
├── config.py            # Configuration settings
├── requirements.txt     # Dependencies
├── organized_content.json # Knowledge base
└── README.md           # This file
```

### Adding New Features

1. **Custom Preprocessing**: Modify `text_preprocessor.py`
2. **New Query Types**: Extend `rag_system.py`
3. **UI Enhancements**: Update `app.py`
4. **Memory Features**: Enhance `memory_manager.py`

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration options

---

**Built with ❤️ for Bengali Literature Education**
