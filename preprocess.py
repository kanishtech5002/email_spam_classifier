import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

# Initialize NLTK PorterStemmer (rule-based stemmer, requires no external corpora)
stemmer = PorterStemmer()

# Store built-in scikit-learn English stopwords in a set for O(1) lookup
stop_words = set(ENGLISH_STOP_WORDS)

def preprocess_text(text):
    """
    Preprocesses raw text using a structured NLP pipeline:
    1. Converts text to lowercase.
    2. Removes all punctuation marks.
    3. Removes numeric digits.
    4. Tokenizes text into individual words.
    5. Filters out English stopwords using Scikit-Learn's built-in ENGLISH_STOP_WORDS.
    6. Stems words to their root forms using PorterStemmer.
    7. Cleans up any extra whitespaces.
    
    Parameters:
        text (str): The raw email text to be processed.
        
    Returns:
        str: The fully preprocessed and normalized text.
    """
    # Handle null or non-string inputs gracefully
    if not isinstance(text, str):
        return ""
    
    # 1. Convert text to lowercase
    text = text.lower()
    
    # 2. Remove punctuation (replaced with space to avoid merging words together, e.g. "spam/ham" -> "spam ham")
    punctuation_mapping = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(punctuation_mapping)
    
    # 3. Remove numbers (digits 0-9)
    text = re.sub(r'\d+', ' ', text)
    
    # 4. Split text into individual words by whitespace
    words = text.split()
    
    # 5. Remove stopwords and 6. Apply Porter stemming
    processed_words = []
    for word in words:
        word = word.strip()
        if word and word not in stop_words:
            # Apply stemming
            stemmed = stemmer.stem(word)
            processed_words.append(stemmed)
            
    # 7. Join words back with a single space, removing any multiple consecutive spaces
    cleaned_text = ' '.join(processed_words)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text
