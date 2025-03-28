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
nltk.download('maxent_ne_chunker_tab')

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
        print(f"\n--- Processing Text from: {filePath} ---")
        print("\n--- Original Text ---")
        print(text)

        # --- Step 1: Sentence Tokenization ---
        sentences = nltk.sent_tokenize(text)
        print("\n--- Sentences ---")
        print(sentences)

        # --- Steps for NER (Requires Original Case and POS Tags) ---
        print("\n--- Named Entity Recognition (NER) ---")
        all_named_entities = [] # To store entities from all sentences

        for sentence in sentences:
            # 1a. Word Tokenize (Keep Original Case)
            words_original_case = nltk.word_tokenize(sentence)

            # 1b. POS Tagging (On Original Case Words)
            pos_tags_original_case = nltk.pos_tag(words_original_case)
            # print(f"\nPOS Tags for NER in sentence: {pos_tags_original_case}") # Optional: view tags used by NER

            # 1c. NER Chunking (Input is POS tagged words)
            ner_tree = nltk.ne_chunk(pos_tags_original_case)
            print(f"\nNER Tree for sentence: '{sentence}'")
            print(ner_tree)

            # 1d. Extract Entities from the Tree
            print("\nExtracted Entities from sentence:")
            for chunk in ner_tree:
                if hasattr(chunk, 'label'): # Check if it's a tagged entity chunk
                    entity_name = ' '.join(c[0] for c in chunk)
                    entity_label = chunk.label()
                    print(f"  - {entity_name}: {entity_label}")
                    all_named_entities.append((entity_name, entity_label)) # Store entity

    print("\n--- All Extracted Named Entities (List) ---")
    print(all_named_entities)

    print("\n--- Text Cleaning & Normalization Steps ---")
    # Tokenize all words and lowercase
    words_all = nltk.word_tokenize(text)
    words_lower = [w.lower() for w in words_all]
    print("\nLowercase Tokens:", words_lower[:50], "...") # Show subset

    # Remove Stopwords
    stopwords_list = nltk.corpus.stopwords.words('english')
    words_no_stopwords = [w for w in words_lower if w not in stopwords_list]

    # Remove Punctuation & Non-Alphabetic
    words_alpha = [w for w in words_no_stopwords if w.isalpha()]
    print("\nCleaned Alpha Tokens:", words_alpha[:50], "...") # Show subset

    # POS Tagging (for Lemmatization) - on cleaned, lowercased words
    pos_tags_cleaned = nltk.pos_tag(words_alpha)

    # Stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(w) for w in words_alpha]
    print("\nStemmed Words:", stemmed_words[:50], "...") # Show subset

    # POS-aware Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words_pos = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags_cleaned]
    print("\nPOS-aware Lemmatized Words:", lemmatized_words_pos[:50], "...") # Show subset

preprocess_text('simple.txt')