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
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import wordnet

def get_wordnet_pos(nltk_tag):
    """Map NLTK POS tags to WordNet POS tags."""
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        # Default to noun if mapping is unclear
        return wordnet.NOUN

def preprocess_text(filePath):
    with open(filePath, 'r', encoding='utf-8') as file:
        text = file.read()
        print("\n Printing the whole text: \n \n",text)
        sentences = nltk.sent_tokenize(text)
        print("\n Printing sentences after tokenization: \n \n",sentences)

        wordsInitial = []
        for sent in sentences:
            words.extend(nltk.word_tokenize(sent))
        wordsInitial = [w.lower() for w in words]
        print("\n Printing words after tokenization: \n \n",wordsInitial)

        stopwords = nltk.corpus.stopwords.words('english')
        words = [w for w in wordsInitial if w not in stopwords]
        print("\n Printing words after removing stopwords: \n \n",words)

        words = [w for w in words if w not in string.punctuation]
        print("\n Printing words after removing punctuation: \n \n",words)

        words = [w for w in words if w.isalpha()]
        print("\n Printing words after removing non-alphabetic characters: \n \n",words)

        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(w) for w in words]
        print("\n Printing words after stemming: \n \n",stemmed_words)

        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(w) for w in words]
        print("\n Printing words after lemmatization: \n \n",lemmatized_words)

        pos_tags = nltk.pos_tag(words)
        print("\n--- POS Tags ---")
        print(pos_tags)

        lemmatized_words_pos = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
        print("\n Printing words after POS-aware lemmatization: \n \n", lemmatized_words_pos)

preprocess_text('simple.txt')