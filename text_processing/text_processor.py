import io
import nltk
nltk.download('punkt')
import PyPDF2
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
            tokens = word_tokenize(text)
            processed_text = ' '.join(tokens)
            processed_text = re.sub(r'[^\w\s]', ' ', processed_text)
            processed_text = re.sub(r'\s+', ' ', processed_text).strip().lower()

            return processed_text
        except Exception as e:
            logger.error(f"Error in Vietnamese preprocessing: {str(e)}")
            return TextProcessor.basic_preprocess(text)

    @staticmethod
    def preprocess_english(text: str) -> str:
        try:
            text = TextProcessor.clean_text(text)
            sentences = sent_tokenize(text)
            processed_sentences = []

            for sentence in sentences:
                tokens = nltk_tokenize(sentence)
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
    def extract_text_from_pdf_bytes (pdf_bytes: bytes) -> str:
        text = ""
        try:
            # Create a PDF reader object from bytes
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))

            # Extract text from each page
            for page in pdf_reader.pages:
                text += page.extract_text()

            return text
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise ValueError(f"Error processing PDF: {str(e)}")

    @staticmethod
    def extract_text_from_pdf (pdf_file):
        text = ""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise ValueError(f"Error processing PDF: {str(e)}")
