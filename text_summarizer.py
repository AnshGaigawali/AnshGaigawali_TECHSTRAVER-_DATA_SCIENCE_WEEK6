import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BartForConditionalGeneration, BartTokenizer

nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

def preprocess(text):
    doc = nlp(text.lower())
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def extractive_summary(documents, n_sentences=2):
    cleaned_docs = [preprocess(doc) for doc in documents]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_docs)

    sentence_scores = tfidf_matrix.sum(axis=1).A1
    ranked_sentences = [i for i in range(len(sentence_scores))]
    ranked_sentences.sort(key=lambda i: sentence_scores[i], reverse=True)
    return [documents[i] for i in ranked_sentences[:n_sentences]]

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

def abstractive_summary(text):
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    user_input = input("Enter the text you want to summarize: ")
    documents = [user_input]
    
    extractive = extractive_summary(documents, n_sentences=2)
    abstractive = abstractive_summary(documents[0])

    print("\nExtractive Summary:", extractive)
    print("Abstractive Summary:", abstractive)
