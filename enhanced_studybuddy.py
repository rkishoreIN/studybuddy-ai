"""
Enhanced StudyBuddy AI - 3-Section Streamlit Application
Version: 3.7.0
Features: RAG with FAISS, Progress Tracking, Study Guide Generation, Answer Validation, Smart LLM Fallback, Temperature Control, GPT-4 Fallback, Current Events Detection
"""

import streamlit as st
import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import faiss
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
import PyPDF2
from io import BytesIO
from datetime import datetime, timedelta
import hashlib
import re

# Load environment variables
load_dotenv()

# Configuration for deployment
DATA_DIR = os.getenv('STUDYBUDDY_DATA_DIR', './studybuddy_data')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Version
VERSION = "3.7.0"

# Configure Streamlit page
st.set_page_config(
    page_title=f"StudyBuddy AI {VERSION}",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ProgressTracker:
    """Track learning progress and statistics"""
    
    def __init__(self):
        self.progress_file = "studybuddy_progress.json"
        self.load_progress()
    
    def load_progress(self):
        """Load progress from file"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    self.data = json.load(f)
            except:
                self.data = {
                    'total_questions': 0,
                    'correct_answers': 0,
                    'study_sessions': 0,
                    'total_cost': 0.0,
                    'topics_studied': {},
                    'learning_streak': 0,
                    'last_session': None,
                    'achievements': []
                }
        else:
            self.data = {
                'total_questions': 0,
                'correct_answers': 0,
                'study_sessions': 0,
                'total_cost': 0.0,
                'topics_studied': {},
                'learning_streak': 0,
                'last_session': None,
                'achievements': []
            }
    
    def save_progress(self):
        """Save progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def record_question(self, topic: str, cost: float):
        """Record a question asked"""
        self.data['total_questions'] += 1
        self.data['total_cost'] += cost
        
        if topic not in self.data['topics_studied']:
            self.data['topics_studied'][topic] = 0
        self.data['topics_studied'][topic] += 1
        
        # Update learning streak
        today = datetime.now().date()
        if self.data['last_session']:
            last_date = datetime.fromisoformat(self.data['last_session']).date()
            if (today - last_date).days == 1:
                self.data['learning_streak'] += 1
            elif (today - last_date).days > 1:
                self.data['learning_streak'] = 1
        else:
            self.data['learning_streak'] = 1
        
        self.data['last_session'] = today.isoformat()
        self.save_progress()
    
    def get_stats(self):
        """Get current statistics"""
        return {
            'total_questions': self.data['total_questions'],
            'accuracy': (self.data['correct_answers'] / max(self.data['total_questions'], 1)) * 100,
            'study_sessions': self.data['study_sessions'],
            'total_cost': self.data['total_cost'],
            'learning_streak': self.data['learning_streak'],
            'topics_count': len(self.data['topics_studied']),
            'topics': self.data['topics_studied']
        }

class EnhancedFAISSCollection:
    """Enhanced FAISS collection with better persistence and metadata"""
    
    def __init__(self, name: str, dimension: int = 1536, persist_directory: str = None):
        self.name = name
        self.dimension = dimension
        self.persist_directory = persist_directory or DATA_DIR
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.metadata = []
        self.ids = []
        self.embedding_cache = {}
        self.next_id = 0
        
        os.makedirs(persist_directory, exist_ok=True)
        self.load()
    
    def save(self):
        """Save collection with enhanced metadata"""
        if len(self.documents) > 0:
            # Save FAISS index
            index_path = os.path.join(self.persist_directory, f"{self.name}_index.faiss")
            faiss.write_index(self.index, index_path)
            
            # Save enhanced metadata
            metadata_path = os.path.join(self.persist_directory, f"{self.name}_metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.metadata,
                    'ids': self.ids,
                    'next_id': self.next_id,
                    'embedding_cache': self.embedding_cache,
                    'last_updated': datetime.now().isoformat(),
                    'version': VERSION
                }, f)
            
            return {
                'index_path': index_path,
                'metadata_path': metadata_path,
                'document_count': len(self.documents)
            }
        return None
    
    def load(self):
        """Load collection with version checking"""
        index_path = os.path.join(self.persist_directory, f"{self.name}_index.faiss")
        metadata_path = os.path.join(self.persist_directory, f"{self.name}_metadata.pkl")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                self.index = faiss.read_index(index_path)
                
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.metadata = data['metadata']
                    self.ids = data['ids']
                    self.next_id = data['next_id']
                    self.embedding_cache = data.get('embedding_cache', {})
                
                return True
            except Exception as e:
                logger.error(f"Error loading FAISS collection: {e}")
                return False
        return False
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """Add documents with enhanced processing"""
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        client = OpenAI(api_key=st.session_state.get('api_key'))
        
        for doc, meta in zip(documents, metadatas):
            # Generate embedding
            doc_hash = hashlib.md5(doc.encode()).hexdigest()
            if doc_hash in self.embedding_cache:
                embedding = self.embedding_cache[doc_hash]
            else:
                try:
                    response = client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=doc
                    )
                    embedding = response.data[0].embedding
                    self.embedding_cache[doc_hash] = embedding
                except Exception as e:
                    logger.error(f"Error generating embedding: {e}")
                    continue
            
            # Add to index
            embedding_array = np.array([embedding], dtype=np.float32)
            self.index.add(embedding_array)
            
            # Store metadata
            self.documents.append(doc)
            self.metadata.append(meta)
            self.ids.append(self.next_id)
            self.next_id += 1
        
        self.save()
    
    def search(self, query: str, n_results: int = 3) -> Dict:
        """Enhanced search with better result formatting"""
        if len(self.documents) == 0:
            return {
                'documents': [],
                'metadatas': [],
                'distances': [],
                'ids': []
            }
        
        client = OpenAI(api_key=st.session_state.get('api_key'))
        
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )
            query_embedding = np.array([response.data[0].embedding], dtype=np.float32)
            
            distances, indices = self.index.search(query_embedding, min(n_results, len(self.documents)))
            
            results = {
                'documents': [],
                'metadatas': [],
                'distances': [],
                'ids': []
            }
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.documents):
                    results['documents'].append(self.documents[idx])
                    results['metadatas'].append(self.metadata[idx])
                    results['distances'].append(float(dist))
                    results['ids'].append(self.ids[idx])
            
            return results
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return {'documents': [], 'metadatas': [], 'distances': [], 'ids': []}

class EnhancedRAGSystem:
    """Enhanced RAG system with better error handling and logging"""
    
    def __init__(self, collection: EnhancedFAISSCollection, progress_tracker: ProgressTracker):
        self.collection = collection
        self.progress_tracker = progress_tracker
        self.confidence_threshold = 0.3
        self.min_results = 2
        
        # API pricing
        self.embedding_cost = 0.0001
        self.chat_input_cost = 0.0015
        self.chat_output_cost = 0.002
    
    def count_tokens(self, text: str) -> int:
        """Improved token counting"""
        return int(len(text.split()) * 1.3)
    
    def calculate_confidence(self, search_results: Dict) -> Tuple[float, str]:
        """Enhanced confidence calculation"""
        if not search_results['documents']:
            return 0.0, 'llm_fallback'
        
        avg_distance = np.mean(search_results['distances'])
        avg_similarity = 1 / (1 + avg_distance)
        
        result_count = len(search_results['documents'])
        result_factor = min(result_count / self.min_results, 1.0)
        
        confidence = (avg_similarity * 0.7) + (result_factor * 0.3)
        
        if confidence >= self.confidence_threshold:
            return confidence, 'vector_only'
        elif confidence >= 0.15:
            return confidence, 'hybrid'
        else:
            return confidence, 'llm_fallback'
    
    def ask_question(self, question: str, topic: str = "general") -> Dict:
        """Enhanced question answering with answer validation for reliable fallback"""
        try:
            client = OpenAI(api_key=st.session_state.get('api_key'))
            
            total_cost = 0.0
            embedding_tokens = 0
            input_tokens = 0
            output_tokens = 0
            
            # Check if we have any documents
            has_documents = len(self.collection.documents) > 0
            
            if has_documents:
                # Search vector database
                search_results = self.collection.search(question, n_results=3)
                
                # Log search results
                log_entry = {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "activity": "Vector Search",
                    "details": f"Found {len(search_results['documents'])} documents",
                    "type": "info"
                }
                st.session_state.debug_logs.insert(0, log_entry)
                
                embedding_tokens = self.count_tokens(question)
                total_cost += (embedding_tokens / 1000) * self.embedding_cost
                
                # Build context
                context = "\n\n".join([
                    f"Source {i+1}: {doc}"
                    for i, doc in enumerate(search_results['documents'])
                ]) if search_results['documents'] else ""
                
                # CRITICAL: Validate if materials can answer the question
                if context:
                    validation_prompt = f"""You are evaluating whether study materials can answer a question.

Study Materials:
{context}

Question: {question}

Can these materials answer this question? Reply with ONLY "YES" or "NO" and nothing else."""

                    validation_response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a validator. Reply only YES or NO."},
                            {"role": "user", "content": validation_prompt}
                        ],
                        temperature=0,
                        max_tokens=5
                    )
                    
                    can_answer = validation_response.choices[0].message.content.strip().upper()
                    
                    validation_tokens = self.count_tokens(validation_prompt) + 5
                    total_cost += (validation_tokens / 1000) * self.chat_input_cost
                    
                    log_entry = {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "activity": "Answer Validation",
                        "details": f"Materials can answer: {can_answer}",
                        "type": "info"
                    }
                    st.session_state.debug_logs.insert(0, log_entry)
                    
                    if can_answer == "NO":
                        strategy = 'llm_fallback'
                        confidence = 0.0
                        log_entry = {
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "activity": "Validation Fallback",
                            "details": "Materials insufficient, using LLM general knowledge",
                            "type": "warning"
                        }
                        st.session_state.debug_logs.insert(0, log_entry)
                    else:
                        # Calculate confidence
                        confidence, strategy = self.calculate_confidence(search_results)
                        
                        # Additional distance check
                        if len(search_results['distances']) > 0:
                            min_distance = min(search_results['distances'])
                            if min_distance > 0.8:  # Stricter threshold
                                strategy = 'hybrid'
                else:
                    strategy = 'llm_fallback'
                confidence = 0.0
                
            else:
                # No documents available
                search_results = {'documents': [], 'metadatas': [], 'distances': [], 'ids': []}
                context = ""
                confidence = 0.0
                strategy = 'llm_fallback'
                
                log_entry = {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "activity": "No Materials",
                    "details": "No study materials, using LLM general knowledge",
                    "type": "warning"
                }
                st.session_state.debug_logs.insert(0, log_entry)
            
            # Generate answer based on strategy
            if strategy == 'llm_fallback':
                # Check if this is a question about current events that might need web search
                current_event_keywords = [
                    "president", "prime minister", "current", "latest", "recent", "2024", "2025",
                    "who is", "what is the current", "latest news", "recent events"
                ]
                
                is_current_event = any(keyword in question.lower() for keyword in current_event_keywords)
                
                if is_current_event:
                    # For current events, use a more direct approach
                    prompt = f"""You are a knowledgeable assistant. Answer this question with the most current information you have. If your knowledge might be outdated, please mention this.

Question: {question}

Answer:"""
                    system_message = "You are a knowledgeable assistant. Provide the most current information available. If you're unsure about recent changes, mention that your knowledge may not be completely up-to-date."
                else:
                    # For general questions, use standard approach
                    prompt = f"Answer this question using your knowledge: {question}"
                    system_message = "You are a knowledgeable assistant. Provide accurate, direct answers."
                
                model_to_use = "gpt-4"
                
            elif strategy == 'hybrid':
                # Hybrid mode
                prompt = f"""Use the study materials and your general knowledge to answer.

Study Materials:
{context}

Question: {question}

Provide a comprehensive answer:"""
                system_message = "You are a study assistant. Combine materials with general knowledge."
                model_to_use = "gpt-3.5-turbo"
                
            else:  # vector_only
                # Materials only
                prompt = f"""Answer based on the study materials.

Study Materials:
{context}

Question: {question}

Answer:"""
                system_message = "You are a study assistant. Answer from provided materials."
                model_to_use = "gpt-3.5-turbo"
            
            input_tokens = self.count_tokens(prompt)
            total_cost += (input_tokens / 1000) * self.chat_input_cost
            
            # Log LLM request
            log_entry = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "activity": "LLM Request",
                "details": f"Strategy: {strategy}, Model: {model_to_use}",
                "type": "info"
            }
            st.session_state.debug_logs.insert(0, log_entry)
            
            # Make LLM call
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=st.session_state.get('temperature', 0.7)
            )
            
            answer = response.choices[0].message.content
            output_tokens = self.count_tokens(answer)
            total_cost += (output_tokens / 1000) * self.chat_output_cost
            
            # Log success
            log_entry = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "activity": "Answer Generated",
                "details": f"Strategy: {strategy}",
                "type": "success"
            }
            st.session_state.debug_logs.insert(0, log_entry)
            
            # Record progress
            self.progress_tracker.record_question(topic, total_cost)
            
            return {
                'answer': answer,
                'confidence': confidence,
                'strategy': strategy,
                'sources': search_results.get('documents', []),
                'materials_used': len(search_results.get('documents', [])) > 0 and strategy != 'llm_fallback',
                'fallback_used': strategy == 'llm_fallback',
                'cost': {
                    'total': round(total_cost, 6),
                    'embedding_tokens': embedding_tokens,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'embedding_cost': round((embedding_tokens / 1000) * self.embedding_cost, 6),
                    'input_cost': round((input_tokens / 1000) * self.chat_input_cost, 6),
                    'output_cost': round((output_tokens / 1000) * self.chat_output_cost, 6)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in ask_question: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'confidence': 0.0,
                'strategy': 'error',
                'sources': [],
                'materials_used': False,
                'fallback_used': True,
                'cost': {'total': 0.0}
            }

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF with error handling"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Enhanced text chunking with better boundaries"""
    # Split by sentences first
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if current_length + len(sentence) > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def generate_study_guide(collection: EnhancedFAISSCollection, guide_type: str = "comprehensive") -> Dict:
    """Generate study guide based on collection content with LLM fallback"""
    client = OpenAI(api_key=st.session_state.get('api_key'))
    
    # Check if we have sufficient materials
    has_materials = len(collection.documents) > 0
    materials_used = False
    fallback_used = False
    
    if has_materials:
        sample_size = min(15, len(collection.documents))
        sample_docs = collection.documents[:sample_size]
        context = "\n\n".join(sample_docs)
        
        # Log material usage
        log_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "activity": "Study Guide Generation",
            "details": f"Using {len(collection.documents)} documents from study materials",
            "type": "info"
        }
        st.session_state.debug_logs.insert(0, log_entry)
        materials_used = True
    else:
        # Log fallback usage
        log_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "activity": "Study Guide Generation",
            "details": "No study materials available, using LLM general knowledge",
            "type": "warning"
        }
        st.session_state.debug_logs.insert(0, log_entry)
        fallback_used = True
    
    if guide_type == "comprehensive":
        if has_materials:
            prompt = f"""Based on the study materials, create a comprehensive study guide.

Study Materials:
{context}

Generate:
1. A concise summary (2-3 paragraphs) of the main topics covered
2. 5 key concepts or terms with brief definitions
3. 8 practice questions that test understanding

Format your response EXACTLY as follows:

SUMMARY:
[Your 2-3 paragraph summary here]

KEY CONCEPTS:
1. [Concept]: [Definition]
2. [Concept]: [Definition]
3. [Concept]: [Definition]
4. [Concept]: [Definition]
5. [Concept]: [Definition]

QUESTIONS:
1. [Question]?
2. [Question]?
3. [Question]?
4. [Question]?
5. [Question]?
6. [Question]?
7. [Question]?
8. [Question]?"""
        else:
            prompt = f"""Create a comprehensive study guide on general academic topics.

Generate:
1. A concise summary (2-3 paragraphs) covering common study topics
2. 5 key concepts or terms with brief definitions
3. 8 practice questions that test general knowledge

Format your response EXACTLY as follows:

SUMMARY:
[Your 2-3 paragraph summary here]

KEY CONCEPTS:
1. [Concept]: [Definition]
2. [Concept]: [Definition]
3. [Concept]: [Definition]
4. [Concept]: [Definition]
5. [Concept]: [Definition]

QUESTIONS:
1. [Question]?
2. [Question]?
3. [Question]?
4. [Question]?
5. [Question]?
6. [Question]?
7. [Question]?
8. [Question]?"""
    else:
        if has_materials:
            prompt = f"""Based on the study materials, generate 8 practice questions.

Study Materials:
{context}

Generate 8 questions that:
1. Can be definitively answered from the materials above
2. Cover different topics/concepts in the materials
3. Range from simple recall to deeper understanding
4. Are clear and specific

Format: Return only the questions, one per line, numbered 1-8."""
        else:
            prompt = f"""Generate 8 practice questions on general academic topics.

Generate 8 questions that:
1. Cover different academic subjects
2. Range from basic to intermediate level
3. Are clear and specific
4. Test general knowledge

Format: Return only the questions, one per line, numbered 1-8."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert study guide creator."},
                {"role": "user", "content": prompt}
            ],
            temperature=st.session_state.get('temperature', 0.7)
        )
        
        content = response.choices[0].message.content
        
        # Log successful generation
        current_temp = st.session_state.get('temperature', 0.7)
        log_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "activity": "Study Guide Generated",
            "details": f"Successfully generated {guide_type} study guide, temperature: {current_temp}",
            "type": "success"
        }
        st.session_state.debug_logs.insert(0, log_entry)
        
        if guide_type == "comprehensive":
            summary = ""
            key_concepts = []
            questions = []
            
            if "SUMMARY:" in content:
                summary_section = content.split("SUMMARY:")[1].split("KEY CONCEPTS:")[0].strip()
                summary = summary_section
            
            if "KEY CONCEPTS:" in content:
                concepts_section = content.split("KEY CONCEPTS:")[1].split("QUESTIONS:")[0].strip()
                concept_lines = [line.strip() for line in concepts_section.split('\n') if line.strip() and any(c.isalpha() for c in line)]
                key_concepts = concept_lines[:5]
            
            if "QUESTIONS:" in content:
                questions_section = content.split("QUESTIONS:")[1].strip()
                question_lines = [line.strip() for line in questions_section.split('\n') if line.strip() and '?' in line]
                for q in question_lines:
                    q = q.lstrip('0123456789.)')
                    q = q.strip()
                    if q:
                        questions.append(q)
            
            return {
                'summary': summary,
                'key_concepts': key_concepts[:5],
                'questions': questions[:8],
                'materials_used': materials_used,
                'fallback_used': fallback_used,
                'source': 'study_materials' if materials_used else 'llm_general'
            }
        else:
            questions = [q.strip() for q in content.split('\n') if q.strip() and any(c.isalpha() for c in q)]
            cleaned_questions = []
            for q in questions:
                q = q.lstrip('0123456789.)')
                q = q.strip()
                if q and '?' in q:
                    cleaned_questions.append(q)
            
            return {
                'questions': cleaned_questions[:8],
                'materials_used': materials_used,
                'fallback_used': fallback_used,
                'source': 'study_materials' if materials_used else 'llm_general'
            }
    
    except Exception as e:
        logger.error(f"Error generating study guide: {e}")
        # Log error
        log_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "activity": "Study Guide Generation Error",
            "details": f"Error: {str(e)}",
            "type": "error"
        }
        st.session_state.debug_logs.insert(0, log_entry)
        return {'questions': [], 'summary': '', 'key_concepts': [], 'materials_used': False, 'fallback_used': True, 'source': 'error'}

# Initialize session state
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'progress_tracker' not in st.session_state:
    st.session_state.progress_tracker = ProgressTracker()
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv('OPENAI_API_KEY', '')
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'debug_logs' not in st.session_state:
    st.session_state.debug_logs = []
if 'study_guide' not in st.session_state:
    st.session_state.study_guide = None

# Initialize collection
if st.session_state.collection is None and st.session_state.api_key:
    st.session_state.collection = EnhancedFAISSCollection("study_materials")
    st.session_state.rag_system = EnhancedRAGSystem(st.session_state.collection, st.session_state.progress_tracker)

# Main layout
st.title(f"🎓 StudyBuddy AI {VERSION}")
st.markdown("### Your Intelligent Study Assistant with RAG")

# Create three main columns
left_col, middle_col, right_col = st.columns([2, 5, 3])

# LEFT COLUMN - STATUS
with left_col:
    st.header("📊 Status")
    
    # API Key Status
    if st.session_state.api_key:
        st.success("✅ API Key Configured")
    else:
        st.error("❌ API Key Missing")
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            st.session_state.api_key = api_key
            st.rerun()
    
    # Current Settings
    st.subheader("⚙️ Current Settings")
    st.metric("Version", VERSION)
    st.metric("Temperature", f"{st.session_state.get('temperature', 0.7):.1f}")
    
    st.divider()
    
    # Progress Statistics
    stats = st.session_state.progress_tracker.get_stats()
    
    st.subheader("📈 Learning Progress")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions Asked", stats['total_questions'])
        st.metric("Learning Streak", f"{stats['learning_streak']} days")
    with col2:
        st.metric("Study Sessions", stats['study_sessions'])
        st.metric("Topics Studied", stats['topics_count'])
    
    st.metric("Total Cost", f"${stats['total_cost']:.4f}")
    
    # Collection Status
    if st.session_state.collection:
        st.subheader("📚 Materials Status")
        st.metric("Documents Loaded", len(st.session_state.collection.documents))
        
        if len(st.session_state.collection.documents) > 0:
            # Show topics
            topics = {}
            for meta in st.session_state.collection.metadata:
                topic = meta.get('topic', 'general')
                topics[topic] = topics.get(topic, 0) + 1
            
            st.write("**Topics:**")
            for topic, count in topics.items():
                st.write(f"• {topic}: {count} chunks")
    
    st.divider()
    
    # Temperature Control
    st.subheader("🌡️ AI Settings")
    
    # Temperature slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Controls randomness in AI responses. Lower = more focused, Higher = more creative"
    )
    
    # Store temperature in session state
    st.session_state.temperature = temperature
    
    # Temperature explanation
    with st.expander("ℹ️ What is Temperature?", expanded=False):
        st.markdown("""
        **Temperature** controls how "creative" or "focused" the AI responses are:
        
        - **0.0 - 0.3**: Very focused, deterministic answers
        - **0.4 - 0.7**: Balanced, good for most tasks (recommended)
        - **0.8 - 1.2**: More creative and varied responses
        - **1.3 - 2.0**: Very creative, may be less accurate
        
        **For StudyBuddy**: Use 0.7 for balanced answers, or lower (0.3-0.5) for more focused study responses.
        """)
    
    st.divider()
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    
    if st.button("🔄 Refresh Status", use_container_width=True):
        st.rerun()
    
    if st.button("📊 View Progress", use_container_width=True):
        st.session_state.show_progress = True
    
    if st.button("🗑️ Clear Materials", use_container_width=True):
        if st.session_state.collection:
            st.session_state.collection.clear()
            st.success("Materials cleared!")
            st.rerun()

# MIDDLE COLUMN - MAIN INTERFACE
with middle_col:
    st.header("💬 Study Interface")
    
    # File Upload Section
    with st.expander("📁 Upload Study Materials", expanded=False):
        category = st.selectbox(
            "Select Category",
            ["General", "History", "Science", "Math", "Technology", "Literature", "Business"],
            help="Categorize your study materials"
        )
        
        uploaded_files = st.file_uploader(
            "Upload files (PDF, TXT)",
            accept_multiple_files=True,
            type=['pdf', 'txt']
        )
        
        if uploaded_files and st.session_state.api_key:
            if st.button("Process Materials"):
                with st.spinner("Processing files..."):
                    # Log file processing start
                    log_entry = {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "activity": "File Processing Started",
                        "details": f"Processing {len(uploaded_files)} files in category: {category}",
                        "type": "info"
                    }
                    st.session_state.debug_logs.insert(0, log_entry)
                    
                    if st.session_state.collection is None:
                        st.session_state.collection = EnhancedFAISSCollection("study_materials")
                        st.session_state.rag_system = EnhancedRAGSystem(st.session_state.collection, st.session_state.progress_tracker)
                        
                        # Log collection initialization
                        log_entry = {
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "activity": "Database Initialized",
                            "details": "Created new FAISS collection for study materials",
                            "type": "success"
                        }
                        st.session_state.debug_logs.insert(0, log_entry)
                    
                    all_chunks = []
                    all_metadata = []
                    
                    for file in uploaded_files:
                        # Log individual file processing
                        log_entry = {
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "activity": "File Processing",
                            "details": f"Processing file: {file.name}",
                            "type": "info"
                        }
                        st.session_state.debug_logs.insert(0, log_entry)
                        
                        if file.name.endswith('.pdf'):
                            text = extract_text_from_pdf(file)
                            if not text:
                                log_entry = {
                                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                                    "activity": "PDF Processing Warning",
                                    "details": f"Could not extract text from {file.name}",
                                    "type": "warning"
                                }
                                st.session_state.debug_logs.insert(0, log_entry)
                        else:
                            text = file.read().decode('utf-8')
                        
                        chunks = chunk_text(text)
                        all_chunks.extend(chunks)
                        all_metadata.extend([{
                            "source": file.name,
                            "topic": category.lower(),
                            "upload_date": datetime.now().isoformat()
                        }] * len(chunks))
                        
                        # Log chunking results
                        log_entry = {
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "activity": "Text Chunking",
                            "details": f"Created {len(chunks)} chunks from {file.name}",
                            "type": "info"
                        }
                        st.session_state.debug_logs.insert(0, log_entry)
                    
                    # Log embedding generation
                    log_entry = {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "activity": "Embedding Generation",
                        "details": f"Generating embeddings for {len(all_chunks)} chunks",
                        "type": "info"
                    }
                    st.session_state.debug_logs.insert(0, log_entry)
                    
                    st.session_state.collection.add_documents(all_chunks, all_metadata)
                    st.session_state.rag_system = EnhancedRAGSystem(st.session_state.collection, st.session_state.progress_tracker)
                    
                    # Log successful processing
                    log_entry = {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "activity": "Materials Processed",
                        "details": f"Successfully processed {len(all_chunks)} chunks from {len(uploaded_files)} files",
                        "type": "success"
                    }
                    st.session_state.debug_logs.insert(0, log_entry)
                    
                    st.success(f"✅ Processed {len(all_chunks)} chunks from {len(uploaded_files)} files")
    
    # Study Guide Generation
    with st.expander("📚 Generate Study Guide", expanded=False):
        guide_type = st.radio(
            "Guide Type",
            ["questions_only", "comprehensive"],
            format_func=lambda x: "Questions Only" if x == "questions_only" else "Full Study Guide"
        )
        
        if st.button("Generate Study Guide"):
            if st.session_state.collection and len(st.session_state.collection.documents) > 0:
                with st.spinner("Generating study guide..."):
                    guide = generate_study_guide(st.session_state.collection, guide_type)
                    st.session_state.study_guide = guide
                    st.success("Study guide generated!")
            else:
                st.warning("Please upload some materials first.")
    
    # Display Study Guide
    if st.session_state.study_guide:
        st.subheader("📖 Generated Study Guide")
        
        guide = st.session_state.study_guide
        
        # Show source information
        if guide.get('source') == 'study_materials':
            st.success("✅ **Generated from your study materials**")
        elif guide.get('source') == 'llm_general':
            st.warning("⚠️ **Generated using LLM general knowledge** - No study materials available")
        elif guide.get('source') == 'error':
            st.error("❌ **Error generating study guide**")
        
        if guide.get('summary'):
            st.write("**Summary:**")
            st.write(guide['summary'])
        
        if guide.get('key_concepts'):
            st.write("**Key Concepts:**")
            for concept in guide['key_concepts']:
                st.write(f"• {concept}")
        
        if guide.get('questions'):
            st.write("**Practice Questions:**")
            for i, question in enumerate(guide['questions'], 1):
                if st.button(f"{i}. {question}", key=f"guide_q_{i}", use_container_width=True):
                    st.session_state.auto_question = question
                    st.rerun()
    
    # Question Answering Interface
    st.subheader("❓ Ask Questions")
    
    question = st.text_input(
        "Ask a question about your study materials:",
        value=st.session_state.get('auto_question', ''),
        key="question_input"
    )
    
    if st.session_state.get('auto_question'):
        st.session_state.auto_question = ""
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("Ask", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear History", use_container_width=True):
            st.session_state.qa_history = []
            st.rerun()
    
    # Process Question
    if ask_button and question:
        if not st.session_state.collection:
            st.session_state.collection = EnhancedFAISSCollection("study_materials")
            st.session_state.rag_system = EnhancedRAGSystem(st.session_state.collection, st.session_state.progress_tracker)
        
        with st.spinner("Thinking..."):
            result = st.session_state.rag_system.ask_question(question, "general")
            st.session_state.qa_history.insert(0, (question, result))
            
            # Log the interaction
            log_entry = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "question": question,
                "strategy": result['strategy'],
                "confidence": result['confidence'],
                "cost": result['cost']['total']
            }
            st.session_state.debug_logs.insert(0, log_entry)
    
    # Display Q&A History
    if st.session_state.qa_history:
        st.subheader("💬 Conversation History")
        
        for i, (q, r) in enumerate(st.session_state.qa_history):
            with st.expander(f"Q: {q}", expanded=(i==0)):
                # Strategy indicator with enhanced messaging
                if r['strategy'] == 'vector_only':
                    st.success("✅ **Answer from YOUR MATERIALS** - High confidence from study materials")
                elif r['strategy'] == 'hybrid':
                    st.warning("⚡ **HYBRID Answer** - Combining your materials with AI knowledge")
                elif r['strategy'] == 'llm_fallback':
                    st.info("🤖 **Answer from AI KNOWLEDGE** - No study materials available, using general AI knowledge")
                    st.caption("⚠️ Note: AI knowledge may not include the most recent information. For current events, consider checking recent sources.")
                else:
                    st.error("❌ **Error** - Unable to generate answer")
                
                st.write(f"**Answer:** {r['answer']}")
                
                # Enhanced metrics with fallback information
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Confidence", f"{r['confidence']:.1%}")
                with col2:
                    strategy_display = r['strategy'].replace('_', ' ').title()
                    if r.get('fallback_used'):
                        strategy_display += " (Fallback)"
                    st.metric("Strategy", strategy_display)
                with col3:
                    sources_count = len(r['sources']) if r['sources'] else 0
                    st.metric("Sources", sources_count)
                with col4:
                    st.metric("Cost", f"${r['cost']['total']:.4f}")
                
                # Show fallback information
                if r.get('fallback_used'):
                    st.info("ℹ️ **Note**: This answer was generated using AI general knowledge because no relevant study materials were found.")
                elif r.get('materials_used'):
                    st.success("✅ **Note**: This answer was generated using your uploaded study materials.")
                
                # Sources
                if r['sources']:
                    with st.expander("📚 Sources Used"):
                        for j, source in enumerate(r['sources'], 1):
                            st.write(f"**Source {j}:**")
                            st.text(source[:300] + "..." if len(source) > 300 else source)

# RIGHT COLUMN - COMPREHENSIVE LOGGING
with right_col:
    st.header("🔧 System Logs")
    
    # Comprehensive Activity Logs
    if st.session_state.debug_logs:
        st.subheader("📝 All Activities")
        
        # Filter options
        log_filter = st.selectbox(
            "Filter Logs",
            ["All", "Info", "Success", "Warning", "Error"],
            help="Filter logs by type"
        )
        
        # Display filtered logs
        filtered_logs = st.session_state.debug_logs
        if log_filter != "All":
            filtered_logs = [log for log in st.session_state.debug_logs if log.get('type', 'info') == log_filter.lower()]
        
        for log in filtered_logs[:15]:  # Show more logs
            with st.container():
                # Color code based on log type
                if log.get('type') == 'error':
                    st.error(f"🚨 {log['timestamp']} - {log.get('activity', 'Unknown')}")
                elif log.get('type') == 'warning':
                    st.warning(f"⚠️ {log['timestamp']} - {log.get('activity', 'Unknown')}")
                elif log.get('type') == 'success':
                    st.success(f"✅ {log['timestamp']} - {log.get('activity', 'Unknown')}")
                else:
                    st.info(f"ℹ️ {log['timestamp']} - {log.get('activity', 'Unknown')}")
                
                # Show details
                if log.get('details'):
                    st.caption(f"Details: {log['details']}")
                
                # Show additional info for Q&A logs
                if 'confidence' in log and 'cost' in log:
                    st.caption(f"Confidence: {log['confidence']:.1%} | Cost: ${log['cost']:.4f}")
                
                # Show fallback information in logs
                if log.get('activity') == 'LLM Fallback':
                    st.caption("⚠️ Using AI general knowledge (no study materials)")
                elif log.get('activity') == 'Vector Search' and log.get('details', '').startswith('Found 0'):
                    st.caption("ℹ️ No relevant materials found, will use LLM fallback")
                
                st.divider()
    else:
        st.info("No activity logs yet. Start using the application to see logs here.")
    
    # System Status
    st.subheader("⚙️ System Status")
    
    if st.session_state.api_key:
        st.success("API: Connected")
    else:
        st.error("API: Not Connected")
    
    if st.session_state.collection:
        st.success(f"Database: {len(st.session_state.collection.documents)} documents")
    else:
        st.warning("Database: Not initialized")
    
    # Session Statistics
    st.subheader("📊 Session Stats")
    
    if st.session_state.debug_logs:
        total_logs = len(st.session_state.debug_logs)
        error_count = len([log for log in st.session_state.debug_logs if log.get('type') == 'error'])
        warning_count = len([log for log in st.session_state.debug_logs if log.get('type') == 'warning'])
        success_count = len([log for log in st.session_state.debug_logs if log.get('type') == 'success'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Activities", total_logs)
            st.metric("Errors", error_count)
        with col2:
            st.metric("Warnings", warning_count)
            st.metric("Success", success_count)
    
    # Clear logs button
    if st.button("Clear All Logs", use_container_width=True):
        st.session_state.debug_logs = []
        st.rerun()

# Footer
st.divider()
st.caption(f"StudyBuddy AI {VERSION} - Built with RAG, FAISS, and Streamlit")