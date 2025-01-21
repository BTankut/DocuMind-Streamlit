import os
import streamlit as st
from typing import List, Dict
from langdetect import detect
from more_itertools import chunked
from pathlib import Path
import re
from datetime import datetime
import time
import PyPDF2
import io
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv
import requests
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# .env dosyasını yükle
load_dotenv()

# Dil çevirileri
TRANSLATIONS = {
    "tr": {
        "title": "📚 Doküman Arama Sistemi",
        "subtitle": "by BarisTankut",
        "description": "PDF ve TXT formatındaki Rusça, Türkçe ve İngilizce dokümanları yükleyin ve arama yapın.",
        "upload_title": "📝 Doküman Yükleme",
        "upload_label": "Dokümanları yükleyin (PDF/TXT) - Rusça, Türkçe, İngilizce",
        "search_tab": "🔍 Doküman Arama",
        "ai_tab": "🤖 Yapay Zeka Sohbet",
        "search_input": "🔍 Arama yapmak için bir kelime veya cümle girin:",
        "ai_input": "💭 Dokümanlar hakkında bir soru sorun:",
        "no_results": "⚠️ Sonuç bulunamadı.",
        "results_found": "✨ {} sonuç bulundu!",
        "result_title": "📄 Sonuç {} - {} (Parça {})",
        "doc_stats": "📊 Doküman boyutu: {} | {} karakter",
        "upload_first": "⚠️ Önce doküman yüklemelisiniz!",
        "ai_thinking": "🤖 Yapay zeka düşünüyor...",
        "error": "Üzgünüm, bir hata oluştu: {}"
    },
    "en": {
        "title": "📚 Document Search System",
        "subtitle": "by BarisTankut",
        "description": "Upload and search documents in Russian, Turkish and English (PDF and TXT format).",
        "upload_title": "📝 Document Upload",
        "upload_label": "Upload documents (PDF/TXT) - Russian, Turkish, English",
        "search_tab": "🔍 Document Search",
        "ai_tab": "🤖 AI Chat",
        "search_input": "🔍 Enter a word or phrase to search:",
        "ai_input": "💭 Ask a question about the documents:",
        "no_results": "⚠️ No results found.",
        "results_found": "✨ {} results found!",
        "result_title": "📄 Result {} - {} (Chunk {})",
        "doc_stats": "📊 Document size: {} | {} characters",
        "upload_first": "⚠️ Please upload documents first!",
        "ai_thinking": "🤖 AI is thinking...",
        "error": "Sorry, an error occurred: {}"
    },
    "ru": {
        "title": "📚 Система поиска документов",
        "subtitle": "by BarisTankut",
        "description": "Загрузите и выполните поиск по документам на русском, турецком и английском языках (формат PDF и TXT).",
        "upload_title": "📝 Загрузка документов",
        "upload_label": "Загрузите документы (PDF/TXT) - русский, турецкий, английский",
        "search_tab": "🔍 Поиск документов",
        "ai_tab": "🤖 ИИ чат",
        "search_input": "🔍 Введите слово или фразу для поиска:",
        "ai_input": "💭 Задайте вопрос о документах:",
        "no_results": "⚠️ Результаты не найдены.",
        "results_found": "✨ Найдено {} результатов!",
        "result_title": "📄 Результат {} - {} (Часть {})",
        "doc_stats": "📊 Размер документа: {} | {} символов",
        "upload_first": "⚠️ Сначала загрузите документы!",
        "ai_thinking": "🤖 ИИ думает...",
        "error": "Извините, произошла ошибка: {}"
    }
}

class DocumentSearchSystem:
    def __init__(self):
        self.documents: List[Dict] = []
        self.doc_dir = "documents"
        self.model = None
        self.embeddings = {}
        
        # Doküman dizinini oluştur
        os.makedirs(self.doc_dir, exist_ok=True)
        
        # ChromaDB istemcisini başlat
        try:
            self.chroma_client = PersistentClient(path="chroma_db")
        except Exception as e:
            st.error(f"ChromaDB başlatma hatası: {str(e)}")
            self.chroma_client = None
            return
        
        # Koleksiyonu oluştur veya var olanı al
        try:
            self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            )
            
            # Koleksiyonu sil ve yeniden oluştur
            try:
                self.chroma_client.delete_collection(name="multilingual_documents")
            except:
                pass
                
            self.collection = self.chroma_client.create_collection(
                name="multilingual_documents",
                embedding_function=self.sentence_transformer_ef
            )
        except Exception as e:
            st.error(f"Koleksiyon oluşturma hatası: {str(e)}")
            self.collection = None
        
        # Model yükleniyor mesajı
        if not st.session_state.get('model_loaded', False):
            with st.spinner('🤖 Yapay zeka modeli yükleniyor... (İlk açılışta biraz zaman alabilir)'):
                self.load_model()
                st.session_state.model_loaded = True
    
    def load_model(self):
        """Rusça dil modelini yükle"""
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Metin için vektör oluştur"""
        return self.model.encode(text, convert_to_tensor=True)
    
    def compute_similarity(self, query_embedding: torch.Tensor, text_embedding: torch.Tensor) -> float:
        """Benzerlik skorunu hesapla"""
        return torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), 
                                                   text_embedding.unsqueeze(0)).item()
    
    def extract_pdf_text(self, file) -> str:
        """PDF dosyasından metin çıkar"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"❌ PDF okuma hatası: {str(e)}")
            return ""
        
    def save_document(self, file) -> bool:
        try:
            # Dosya uzantısını kontrol et
            file_ext = Path(file.name).suffix.lower()
            
            if file_ext == '.pdf':
                content = self.extract_pdf_text(file)
                if not content:
                    return False
            elif file_ext == '.txt':
                content = file.read().decode('utf-8')
            else:
                st.warning(f"⚠️ {file.name} desteklenmeyen dosya formatı! Sadece .pdf ve .txt dosyaları kabul edilir.")
                return False
            
            # Dil kontrolü
            if not content:
                st.warning("⚠️ Bu belge metin içermiyor!")
                return False
                
            try:
                detected_lang = detect(content)
                if detected_lang not in ['ru', 'tr', 'en']:
                    st.warning(f"⚠️ Bu belge desteklenmeyen bir dilde ({detected_lang})! Sadece Rusça, Türkçe ve İngilizce belgeler kabul edilir.")
                    return False
            except:
                st.warning("⚠️ Belgenin dilini tespit edilemedi!")
                return False
            
            # Dokümanı anlamlı bölümlere ayır
            def split_by_sections(text):
                # Rusça, İngilizce ve Türkçe bölüm işaretleri
                section_patterns = [
                    r'Раздел \d+', r'Глава \d+', r'Том \d+',
                    r'Section \d+', r'Chapter \d+', r'Volume \d+',
                    r'Bölüm \d+', r'Kısım \d+'
                ]
                
                # Metni önce satırlara böl
                lines = text.split('\n')
                sections = []
                current_section = []
                
                for line in lines:
                    # Eğer satır bölüm başlangıcı ise
                    if any(re.match(pattern, line.strip()) for pattern in section_patterns):
                        if current_section:
                            section_text = '\n'.join(current_section).strip()
                            if section_text:
                                sections.append(section_text)
                            current_section = []
                    current_section.append(line)
                
                # Son bölümü ekle
                if current_section:
                    section_text = '\n'.join(current_section).strip()
                    if section_text:
                        sections.append(section_text)
                
                # Eğer bölüm bulunamadıysa, paragraf bazlı böl
                if not sections:
                    sections = [p.strip() for p in text.split('\n\n') if p.strip()]
                
                return sections
            
            # Dokümanı bölümlere ayır
            chunks = split_by_sections(content)
            
            try:
                # ChromaDB'ye ekle
                self.collection.add(
                    documents=chunks,
                    metadatas=[{"title": file.name} for _ in chunks],
                    ids=[f"{file.name}_{i}" for i in range(len(chunks))]
                )
            except Exception as e:
                st.error(f"❌ ChromaDB hatası: {str(e)}")
                st.error(f"Bölüm sayısı: {len(chunks)}")
                st.error(f"En büyük bölüm boyutu: {max(len(chunk) for chunk in chunks)} karakter")
                return False
            
            # Bellek içi dokümanlara da ekle
            document = {
                'title': file.name,
                'content': content,
                'chunks': chunks,
                'date_added': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.documents.append(document)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Dosya kaydetme hatası: {str(e)}")
            return False
    
    def search_documents(self, query: str, chunk_size: int = 500) -> List[Dict]:
        """Dokümanlarda arama yap"""
        try:
            # Vektör araması
            vector_results = self.collection.query(
                query_texts=[query],
                n_results=5
            )
            
            # Metin tabanlı arama için regex pattern
            pattern = re.compile(query, re.IGNORECASE)
            
            # Tüm dokümanları tara
            text_results = []
            for doc in self.documents:
                for chunk in doc['chunks']:
                    if pattern.search(chunk):
                        text_results.append({
                            'chunk': chunk,
                            'title': doc['title'],
                            'score': 1.0 if query.lower() in chunk.lower() else 0.8
                        })
            
            # Sonuçları birleştir
            combined_results = []
            
            # Vektör sonuçlarını ekle
            if vector_results['documents']:
                for doc, metadata, score in zip(
                    vector_results['documents'][0],
                    vector_results['metadatas'][0],
                    vector_results['distances'][0]
                ):
                    combined_results.append({
                        'chunk': doc,
                        'title': metadata['title'],
                        'score': 1 - score  # ChromaDB'de düşük mesafe = yüksek benzerlik
                    })
            
            # Metin sonuçlarını ekle
            combined_results.extend(text_results)
            
            # Tekrar eden sonuçları kaldır ve sırala
            seen = set()
            unique_results = []
            for result in combined_results:
                if result['chunk'] not in seen:
                    seen.add(result['chunk'])
                    unique_results.append(result)
            
            # Skora göre sırala
            unique_results.sort(key=lambda x: x['score'], reverse=True)
            
            return unique_results[:5]  # En iyi 5 sonucu döndür
            
        except Exception as e:
            st.error(f"❌ Arama hatası: {str(e)}")
            return []
    
    def highlight_text(self, text: str, query: str) -> str:
        """Metinde arama sorgusunu vurgula"""
        if not query:
            return text
            
        pattern = re.compile(f'({re.escape(query)})', re.IGNORECASE)
        return pattern.sub(r'**\1**', text)
            
    def ask_ai(self, question: str, context: str) -> str:
        """GPT-4'e soru sor"""
        try:
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                return "API anahtarı bulunamadı. Lütfen .env dosyasını kontrol edin."

            headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/BTankut/DocuMind-Streamlit",
                "X-Title": "DocuMind-Streamlit",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "openai/gpt-4",
                "messages": [
                    {
                        "role": "system",
                        "content": """Sen çok yetenekli bir Rusça doküman analiz asistanısın. Aşağıdaki özelliklere sahipsin:

1. Dil Yetenekleri:
   - Rusça metinleri mükemmel şekilde anlama ve analiz etme
   - Rusça-Türkçe çeviri yapabilme
   - Kullanıcının tercih ettiği dilde yanıt verme
   - Teknik ve akademik Rusça terminolojiye hakimiyet

2. Analiz Yetenekleri:
   - Dokümanların ana fikrini çıkarma
   - Önemli noktaları özetleme
   - Metindeki anahtar kavramları belirleme
   - Bağlamsal ilişkileri kurma
   - Karmaşık fikirleri basitleştirme

3. İletişim Tarzı:
   - Net ve anlaşılır ifadeler kullanma
   - Gerektiğinde detaylı açıklamalar yapma
   - Profesyonel ve saygılı bir ton kullanma
   - Kullanıcı sorularını doğru yorumlama

4. Özel Yetenekler:
   - Rusça dokümanlardan alıntı yapabilme
   - Teknik terimleri açıklayabilme
   - Metinler arası bağlantılar kurabilme
   - Gerektiğinde ek kaynaklara yönlendirme

Verilen bağlamı kullanarak soruları bu yetenekler çerçevesinde yanıtla. Her zaman doğru, güvenilir ve yapıcı bilgiler sun."""
                    },
                    {
                        "role": "user",
                        "content": f"""Bağlam:
{context}

Soru:
{question}

Lütfen yukarıdaki yeteneklerini kullanarak bu soruyu yanıtla."""
                    }
                ]
            }
            
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"API hatası: {response.status_code} - {response.text}"
            
        except Exception as e:
            return f"Üzgünüm, bir hata oluştu: {str(e)}"

def format_size(size_bytes: int) -> str:
    """Boyutu okunabilir formata çevir"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def main():
    # Session state'i başlat
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    # Session state başlatma
    if "lang" not in st.session_state:
        st.session_state.lang = "tr"
        
    # Sidebar ayarları
    with st.sidebar:
        # Dil seçimi
        lang = st.selectbox(
            "🌐 Dil Seçimi",
            ["Türkçe", "English", "Русский"],
            index=["tr", "en", "ru"].index(st.session_state.lang)
        )
        st.session_state.lang = {"Türkçe": "tr", "English": "en", "Русский": "ru"}[lang]
    
    # Çevirileri al
    t = TRANSLATIONS[st.session_state.lang]
    
    # Başlık ve alt başlık
    st.markdown(f"""
        <h1 style='font-size: 2.3em; margin-bottom: 0;'>{t["title"]}</h1>
        <p style='font-size: 0.8em; color: gray; margin-top: 0; margin-bottom: 20px;'>{t["subtitle"]}</p>
        """, unsafe_allow_html=True)
    st.write(t["description"])
    
    system = DocumentSearchSystem()
    
    # Sol sidebar
    with st.sidebar:
        st.header(t["upload_title"])
        uploaded_files = st.file_uploader(
            t["upload_label"],
            type=["txt", "pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for file in uploaded_files:
                system.save_document(file)
    
    # Ana içerik
    tab1, tab2 = st.tabs([t["search_tab"], t["ai_tab"]])
    
    # Arama sekmesi
    with tab1:
        query = st.text_input(t["search_input"])
        
        if query:
            results = system.search_documents(query)
            
            if not results:
                st.warning(t["no_results"])
            else:
                st.success(t["results_found"].format(len(results)))
                
                for i, result in enumerate(results, 1):
                    with st.expander(
                        t["result_title"].format(i, result['title'], 1)
                    ):
                        st.markdown(f"""
                        {result['chunk']}
                        
                        ---
                        {t["doc_stats"].format(format_size(len(result['chunk'].encode('utf-8'))), len(result['chunk']))}
                        """)
    
    # Yapay Zeka sekmesi
    with tab2:
        if not system.documents:
            st.warning(t["upload_first"])
        else:
            # Chat arayüzü
            st.markdown("""
                <style>
                /* Ana container */
                .main {
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 0;
                }

                /* Chat container */
                .chat-container {
                    display: flex;
                    flex-direction: column;
                    gap: 0;
                    padding-bottom: 120px;
                }

                /* Mesaj stilleri */
                .message {
                    display: flex;
                    padding: 20px 0;
                    margin: 0;
                    border-bottom: 1px solid rgba(0,0,0,0.1);
                }

                .message-content {
                    max-width: 800px;
                    margin: 0 auto;
                    width: 100%;
                    padding: 0 20px;
                }

                .user-message {
                    background: #f7f7f8;
                }

                .assistant-message {
                    background: #ffffff;
                }

                /* Input container */
                .input-container {
                    position: fixed;
                    bottom: 0;
                    left: 0;
                    right: 0;
                    background: linear-gradient(180deg, transparent 0%, #f7f7f8 50%);
                    padding: 2rem 1rem;
                }

                .input-box {
                    max-width: 800px;
                    margin: 0 auto;
                    background: white;
                    border: 1px solid #e5e5e5;
                    border-radius: 0.5rem;
                    box-shadow: 0 2px 6px rgba(0,0,0,.05);
                    padding: 0.5rem;
                }


                /* Chat input özelleştirmeleri */
                section[data-testid="stChatInput"] {
                    position: fixed !important;
                    bottom: 0 !important;
                    left: 0 !important;
                    right: 0 !important;
                    padding: 2rem 5rem !important;
                    background: linear-gradient(180deg, transparent 0%, var(--background-color) 50%) !important;
                    z-index: 1000 !important;
                }

                section[data-testid="stChatInput"] > div {
                    max-width: 48rem !important;
                    margin: 0 auto !important;
                    background: var(--background-color) !important;
                    border: 1px solid rgba(0,0,0,0.1) !important;
                    border-radius: 1rem !important;
                    box-shadow: 0 0 15px rgba(0,0,0,0.1) !important;
                    padding: 0.75rem !important;
                }

                section[data-testid="stChatInput"] textarea {
                    background: transparent !important;
                    border: none !important;
                    padding: 0.5rem !important;
                    resize: none !important;
                    height: 24px !important;
                    overflow-y: hidden !important;
                }

                [data-theme="dark"] section[data-testid="stChatInput"] {
                    background: linear-gradient(180deg, transparent 0%, #2d2d2d 50%) !important;
                }

                [data-theme="dark"] section[data-testid="stChatInput"] > div {
                    background: #40414f !important;
                    border-color: #565869 !important;
                }

                [data-theme="dark"] section[data-testid="stChatInput"] textarea {
                    color: #ffffff !important;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Chat container
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Mesajları göster
            for message in st.session_state.messages:
                message_class = "message " + ("user-message" if message["role"] == "user" else "assistant-message")
                st.markdown(f"""
                    <div class="{message_class}">
                        <div class="message-content">
                            {message["content"]}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Input alanı
            if question := st.chat_input(t["ai_input"]):
                # Tüm dokümanları birleştir
                all_docs = "\n---\n".join([
                    f"Document: {doc['title']}\nContent: {doc['content'][:1000]}"
                    for doc in system.documents
                ])
                
                # Mesajları ekle
                st.session_state.messages.append({"role": "user", "content": question})
                
                # AI yanıtı
                with st.spinner(t["ai_thinking"]):
                    answer = system.ask_ai(question, all_docs)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Sayfayı yenile
                st.rerun()

if __name__ == "__main__":
    main()
