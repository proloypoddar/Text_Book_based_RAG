"""
Bengali Text Preprocessing Module
Handles cleaning, normalization, and preparation of Bengali text for RAG
"""
import re
import json
import unicodedata
from typing import List, Dict, Any
import pandas as pd
from config import BENGALI_STOPWORDS

class BengaliTextPreprocessor:
    def __init__(self):
        self.bengali_stopwords = set(BENGALI_STOPWORDS)
        self.punctuation_pattern = re.compile(r'[।,;:!?()[\]{}"\'-]')
        self.whitespace_pattern = re.compile(r'\s+')
        
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters for consistent processing"""
        # Normalize to NFC form
        text = unicodedata.normalize('NFC', text)
        
        # Replace common Unicode variations
        replacements = {
            'ৎ': 'ত্',  # Normalize khanda ta
            'ং': 'ং',   # Normalize anusvara
            'ঃ': 'ঃ',   # Normalize visarga
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize Bengali text"""
        if not text:
            return ""
            
        # Normalize Unicode
        text = self.normalize_unicode(text)
        
        # Remove extra whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """Remove Bengali stopwords"""
        words = text.split()
        filtered_words = [word for word in words if word not in self.bengali_stopwords]
        return ' '.join(filtered_words)
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from Bengali text"""
        # Split by Bengali sentence terminators
        sentences = re.split(r'[।!?]', text)
        
        # Clean and filter empty sentences
        sentences = [self.clean_text(sent) for sent in sentences if sent.strip()]
        
        return sentences
    
    def preprocess_json_content(self, json_file_path: str) -> Dict[str, Any]:
        """Preprocess the organized JSON content"""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_data = {}
        
        # Process story text
        if 'organized_sections' in data and 'story_text' in data['organized_sections']:
            processed_story = []
            for section in data['organized_sections']['story_text']:
                processed_section = {
                    'section': section['section'],
                    'title': self.clean_text(section['title']),
                    'content': self.clean_text(section['content']),
                    'sentences': self.extract_sentences(section['content'])
                }
                processed_story.append(processed_section)
            processed_data['story_text'] = processed_story
        
        # Process MCQ questions
        if 'organized_sections' in data and 'mcq_questions' in data['organized_sections']:
            processed_mcq = {}
            for category, questions in data['organized_sections']['mcq_questions'].items():
                processed_questions = []
                for q in questions:
                    processed_q = {
                        'question_number': q['question_number'],
                        'question': self.clean_text(q['question']),
                        'options': {k: self.clean_text(v) for k, v in q['options'].items()},
                        'correct_answer': q['correct_answer'],
                        'explanation': self.clean_text(q.get('explanation', ''))
                    }
                    if 'source' in q:
                        processed_q['source'] = q['source']
                    processed_questions.append(processed_q)
                processed_mcq[category] = processed_questions
            processed_data['mcq_questions'] = processed_mcq
        
        # Process creative questions
        if 'organized_sections' in data and 'creative_questions' in data['organized_sections']:
            processed_creative = []
            for cq in data['organized_sections']['creative_questions']:
                processed_cq = {
                    'question_number': cq['question_number'],
                    'context': self.clean_text(cq['context']),
                    'questions': {k: self.clean_text(v) for k, v in cq['questions'].items()},
                    'answers': {k: self.clean_text(v) for k, v in cq.get('answers', {}).items()}
                }
                processed_creative.append(processed_cq)
            processed_data['creative_questions'] = processed_creative
        
        # Process word meanings
        if 'organized_sections' in data and 'word_meanings' in data['organized_sections']:
            processed_meanings = {}
            for section, meanings in data['organized_sections']['word_meanings'].items():
                processed_section = {}
                for word, meaning in meanings.items():
                    processed_section[self.clean_text(word)] = self.clean_text(meaning)
                processed_meanings[section] = processed_section
            processed_data['word_meanings'] = processed_meanings
        
        # Process author info
        if 'organized_sections' in data and 'author_info' in data['organized_sections']:
            author_info = data['organized_sections']['author_info']
            processed_author = {}
            for key, value in author_info.items():
                if isinstance(value, dict):
                    processed_author[key] = {k: self.clean_text(str(v)) for k, v in value.items()}
                elif isinstance(value, str):
                    processed_author[key] = self.clean_text(value)
                else:
                    processed_author[key] = value
            processed_data['author_info'] = processed_author
        
        # Process character details
        if 'organized_sections' in data and 'characters_detailed' in data['organized_sections']:
            characters = data['organized_sections']['characters_detailed']
            processed_characters = {}
            for category, chars in characters.items():
                processed_category = {}
                for char_name, char_info in chars.items():
                    processed_char = {}
                    for key, value in char_info.items():
                        if isinstance(value, str):
                            processed_char[key] = self.clean_text(value)
                        else:
                            processed_char[key] = value
                    processed_category[char_name] = processed_char
                processed_characters[category] = processed_category
            processed_data['characters_detailed'] = processed_characters
        
        return processed_data
    
    def create_searchable_chunks(self, processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create searchable text chunks from processed data"""
        chunks = []
        
        # Story text chunks
        if 'story_text' in processed_data:
            for section in processed_data['story_text']:
                chunk = {
                    'type': 'story',
                    'section': section['section'],
                    'title': section['title'],
                    'content': section['content'],
                    'metadata': {
                        'section_number': section['section'],
                        'content_type': 'narrative'
                    }
                }
                chunks.append(chunk)
        
        # MCQ chunks
        if 'mcq_questions' in processed_data:
            for category, questions in processed_data['mcq_questions'].items():
                for q in questions:
                    chunk = {
                        'type': 'mcq',
                        'question': q['question'],
                        'options': q['options'],
                        'answer': q['correct_answer'],
                        'explanation': q['explanation'],
                        'content': f"প্রশ্ন: {q['question']} উত্তর: {q['explanation']}",
                        'metadata': {
                            'category': category,
                            'question_number': q['question_number'],
                            'content_type': 'question_answer'
                        }
                    }
                    chunks.append(chunk)
        
        # Creative question chunks
        if 'creative_questions' in processed_data:
            for cq in processed_data['creative_questions']:
                chunk = {
                    'type': 'creative',
                    'context': cq['context'],
                    'questions': cq['questions'],
                    'answers': cq['answers'],
                    'content': f"প্রসঙ্গ: {cq['context']} প্রশ্ন ও উত্তর: {' '.join(cq['answers'].values())}",
                    'metadata': {
                        'question_number': cq['question_number'],
                        'content_type': 'creative_question'
                    }
                }
                chunks.append(chunk)
        
        # Word meaning chunks
        if 'word_meanings' in processed_data:
            for section, meanings in processed_data['word_meanings'].items():
                for word, meaning in meanings.items():
                    chunk = {
                        'type': 'word_meaning',
                        'word': word,
                        'meaning': meaning,
                        'content': f"শব্দ: {word} অর্থ: {meaning}",
                        'metadata': {
                            'section': section,
                            'content_type': 'vocabulary'
                        }
                    }
                    chunks.append(chunk)
        
        # Character chunks
        if 'characters_detailed' in processed_data:
            for category, characters in processed_data['characters_detailed'].items():
                for char_name, char_info in characters.items():
                    content_parts = [f"চরিত্র: {char_name}"]
                    for key, value in char_info.items():
                        content_parts.append(f"{key}: {value}")
                    
                    chunk = {
                        'type': 'character',
                        'character_name': char_name,
                        'character_info': char_info,
                        'content': ' '.join(content_parts),
                        'metadata': {
                            'category': category,
                            'content_type': 'character_description'
                        }
                    }
                    chunks.append(chunk)
        
        return chunks
