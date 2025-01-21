# DocuMind-Streamlit

A specialized document search and analysis system designed for handling Russian, Turkish, and English technical documentation. The system uses advanced NLP techniques and AI to provide intelligent search capabilities and document analysis.

## Features

- Multi-language support (Russian, Turkish, English)
- PDF and TXT file support
- Semantic search using ChromaDB
- AI-powered document analysis using GPT-4
- Interactive chat interface for document queries
- Section-based document parsing
- Responsive web interface using Streamlit

## Algorithm

The system works in the following steps:

1. **Document Processing**
   - Reads PDF/TXT files
   - Detects document language
   - Splits documents into meaningful sections using:
     - Section markers (Chapter, Section, etc.)
     - Natural paragraph breaks
     - Maintains document structure and context

2. **Vector Database**
   - Uses ChromaDB for document storage
   - Creates embeddings using SentenceTransformer
   - Stores document sections with metadata
   - Enables semantic search capabilities

3. **Search System**
   - Combines vector-based and text-based search
   - Ranks results by relevance
   - Preserves document context in results
   - Returns top 5 most relevant matches

4. **AI Analysis**
   - Uses GPT-4 for document analysis
   - Provides intelligent responses to queries
   - Maintains technical accuracy
   - Supports cross-document analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/BTankut/DocuMind-Streamlit.git
cd DocuMind-Streamlit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API key:
   - Sign up at [OpenRouter](https://openrouter.ai/)
   - Get your API key from the dashboard
   - Create a `.env` file in the project root
   - Copy contents from `.env.example` to `.env`
   - Replace `your_api_key_here` with your actual API key

> ⚠️ **Security Note**: 
> - Never commit your `.env` file or share your API key
> - The `.env` file is listed in `.gitignore` to prevent accidental commits
> - If you accidentally commit sensitive information, change your API key immediately
> - Each user must use their own API key

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Upload documents:
   - Use the sidebar to upload PDF/TXT files
   - Supports Russian, Turkish, and English documents

3. Search documents:
   - Use the search tab for keyword-based search
   - Use the AI chat tab for intelligent queries

## Technical Details

- **Frontend**: Streamlit
- **Backend**: Python
- **Vector Database**: ChromaDB
- **Embedding Model**: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- **Language Detection**: langdetect
- **PDF Processing**: PyPDF2
- **AI Model**: GPT-4 via OpenRouter API

## Recent Updates

- Improved document sectioning algorithm
- Enhanced chat interface with fixed prompt area
- Better handling of technical documentation structure
- Optimized vector search results
- Added support for section-based document parsing

## License

MIT License
