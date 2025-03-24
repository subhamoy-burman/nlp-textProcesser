import nltk
import string

def preprocess_text(filePath):
    with open(filePath, 'r', encoding='utf-8') as file:
        text = file.read()
        print(text)

preprocess_text('simple.txt')