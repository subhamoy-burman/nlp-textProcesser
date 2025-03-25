import nltk
import string
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('stopwords')
nltk.download('wordnet') # For lemmatization later
nltk.download('averaged_perceptron_tagger')  # For POS tagging
nltk.download('maxent_ne_chunker')  # For NER
nltk.download('words') #Also for NER and other purposes


def preprocess_text(filePath):
    with open(filePath, 'r', encoding='utf-8') as file:
        text = file.read()
        print("\n Printing the whole text: \n \n",text)
        sentences = nltk.sent_tokenize(text)
        print("\n Printing sentences after tokenization: \n \n",sentences)

        words = []
        for sent in sentences:
            words.extend(nltk.word_tokenize(sent))
        words = [w.lower() for w in words]
        print("\n Printing words after tokenization: \n \n",words)

        stopwords = nltk.corpus.stopwords.words('english')
        words = [w for w in words if w not in stopwords]
        print("\n Printing words after removing stopwords: \n \n",words)

        words = [w for w in words if w not in string.punctuation]
        print("\n Printing words after removing punctuation: \n \n",words)

        words = [w for w in words if w.isalpha()]
        print("\n Printing words after removing non-alphabetic characters: \n \n",words)

preprocess_text('simple.txt')