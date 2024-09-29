import re
import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

def preprocess(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)  
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    sentences = sent_tokenize(text)
    tokens = [word_tokenize(sentence) for sentence in sentences]
    return tokens

def extract_features(text):
    doc = nlp(text)
    features = {
        "pos_tags": [(token.text, token.pos_) for token in doc],
        "dependency_parse": [(token.text, token.dep_, token.head.text) for token in doc],
        "n_grams": [doc[i:i+2].text for i in range(len(doc)-1)]  
    }
    return features

def check_grammar(tokens):
    errors = []
    for sentence in tokens:
        doc = nlp(' '.join(sentence))
        for token in doc:
            if token.dep_ == 'nsubj' and token.head.lemma_ == 'go' and token.pos_ != 'NOUN':
                if token.head.text == 'go' and token.text == 'She':
                    errors.append((token.head.text, "should be 'goes'"))
            if token.dep_ == 'nsubj' and token.head.lemma_ == 'be':
                if token.head.text == 'is' and token.pos_ == 'NOUN' and token.text in ['they', 'we']:
                    errors.append((token.head.text, "should be 'are'"))
            if token.dep_ == 'dobj' and token.pos_ == 'NOUN':
                if token.text.lower() not in ['a', 'an', 'the']:
                    errors.append((token.text, "missing article before '{}'".format(token.text)))
    return errors

def correct_errors(errors):
    corrections = {}
    for error in errors:
        word, suggestion = error
        if suggestion == "should be 'goes'":
            corrections[word] = "Consider using 'goes' instead of '{}'".format(word)
        elif suggestion == "should be 'are'":
            corrections[word] = "Consider using 'are' instead of '{}'".format(word)
        elif "missing article" in suggestion:
            corrections[word] = "Consider adding 'a', 'an', or 'the' before '{}'".format(word)
    return corrections

if __name__ == "__main__":
    while True:
        text = input("Enter your text (or type 'exit' to quit): ")
        if text.lower() == 'exit':
            print("Exiting the program. Goodbye!")
            break

        if text:
            tokens = preprocess(text)
            features = extract_features(text)
            errors = check_grammar(tokens)
            corrections = correct_errors(errors)

            print("Errors Found:", errors)
            print("Suggested Corrections:", corrections)
        else:
            print("No text provided for analysis.")
