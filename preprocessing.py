import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()

    words = word_tokenize(text)

    words = [
        word for word in words
        if word not in stop_words
        and word not in string.punctuation
    ]

    return " ".join(words)