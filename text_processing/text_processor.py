import nltk
nltk.download('punkt')

import fitz
import langdetect
from underthesea import word_tokenize, pos_tag
from nltk.tokenize import word_tokenize as nltk_tokenize
from nltk.tokenize import sent_tokenize
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tiền xử lý văn bản
class TextProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        # Convert to lowercase
        text = text.lower()
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove special characters and numbers
        text = re.sub(r'[^a-z\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def detect_language(text: str) -> str:
        try:
            logger.info(langdetect.detect(text))
            return langdetect.detect(text)
        except:
            return 'vi'

    @staticmethod
    def preprocess_vietnamese(text: str) -> str:
        """Tiền Xử lý tiếng việt """
        try:
            # Tokenize Vietnamese text
            tokens = word_tokenize(text)

            # Join tokens back together
            processed_text = ' '.join(tokens)

            # Remove special characters and normalize spacing
            processed_text = re.sub(r'[^\w\s]', ' ', processed_text)
            processed_text = re.sub(r'\s+', ' ', processed_text).strip().lower()

            return processed_text
        except Exception as e:
            logger.error(f"Error in Vietnamese preprocessing: {str(e)}")
            # Fallback to basic preprocessing if advanced processing fails
            return TextProcessor.basic_preprocess(text)

    @staticmethod
    def preprocess_english(text: str) -> str:
        try:
            # Clean text from special characters and normalize spacing
            text = TextProcessor.clean_text(text)

            # Tokenize into sentences for better context preservation
            sentences = sent_tokenize(text)
            processed_sentences = []

            for sentence in sentences:
                # Tokenize words
                tokens = nltk_tokenize(sentence)

                # Keep the tokens as they are
                processed_sentences.append(' '.join(tokens))

            return ' '.join(processed_sentences)
        except Exception as e:
            logger.error(f"Error in English preprocessing: {str(e)}")
            return TextProcessor.basic_preprocess(text)

    @staticmethod
    def basic_preprocess(text: str) -> str:
        """Basic preprocessing as fallback"""
        text = re.sub(r'[^\w\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip().lower()

    @staticmethod
    def preprocess_text(text: str, language: str) -> str:
        if language == 'vi':
            return TextProcessor.preprocess_vietnamese(text)
        elif language == 'en':
            return TextProcessor.preprocess_english(text)
        return TextProcessor.basic_preprocess(text)

    @staticmethod
    def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
        text = ""
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
                for page in pdf:
                    text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise ValueError(f"Error processing PDF: {str(e)}")

    @staticmethod
    def extract_text_from_pdf(pdf_file):
        text = ""
        try:
            with fitz.open(stream=pdf_file.read(), filetype="pdf") as pdf:
                for page in pdf:
                    text += page.get_text()
        except Exception as e:
            raise ValueError(f"Error processing PDF: {str(e)}")
        return text
