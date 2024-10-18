from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline, RobertaTokenizer, RobertaForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
import pdfplumber
from docx import Document
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import concurrent.futures
import torch
from gensim import corpora
from gensim.models import LdaModel
from keybert import KeyBERT  # Model-based keyword extraction

# Download stopwords if not already available
nltk.download('stopwords')

app = Flask(__name__)  # Corrected __name__
CORS(app)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load language models and assign them to GPU if available
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0 if torch.cuda.is_available() else -1)
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
topic_classifier = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5).to(device)  # Assuming 5 topics

# Initialize KeyBERT for keyword extraction
kw_model = KeyBERT(model="distilbert-base-nli-mean-tokens")

@app.route('/')
def home():
    return "Welcome to the Text Analysis API! Use the /analyze endpoint to analyze text."

# Function for model-based keyword extraction
def extract_keywords(text, n=10):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=n)
    return [keyword[0] for keyword in keywords]

# Function for LDA-based topic modeling (returning topics as sentences without commas)
def lda_topic_modeling(text, num_topics=5, num_words=5):
    # Preprocess text for topic modeling
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words and word not in string.punctuation]
    
    # Create dictionary and corpus
    dictionary = corpora.Dictionary([words])
    corpus = [dictionary.doc2bow([word]) for word in words]

    # Train LDA model
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    
    # Extract topics and format them into sentences without commas
    topics = lda_model.print_topics(num_topics=num_topics, num_words=num_words)
    topic_sentences = []
    for topic in topics:
        topic_words = [word.split('*')[1].replace('"', '').strip() for word in topic[1].split('+')]
        topic_sentence = "Topic {}: {}".format(topic[0], ' '.join(topic_words))  # Join words with spaces
        topic_sentences.append(topic_sentence)
    
    return topic_sentences

# Helper functions to extract text from different file formats
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text

# Helper function to split text into manageable chunks
def chunk_text(text, max_length=512):
    words = text.split()
    return [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]

# Preprocessing function for text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Removing Punctuation
    text = re.sub(r"\d+", "", text)  # Removing numbers
    text = " ".join(text.split())  # Multiple spaces are reduced to a single space
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Function to analyze each chunk in parallel
def analyze_chunk(chunk):
    summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False, truncation=True)[0]['summary_text']
    sentiment = sentiment_analyzer(chunk, truncation=True)[0]
    return summary, sentiment

# Endpoint to handle text analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    text = None

    # Check if a file is uploaded or text is provided
    if 'file' in request.files:
        file = request.files['file']
        filename = file.filename
        file_extension = os.path.splitext(filename)[1].lower()

        try:
            # Extract text based on the file type
            if file_extension == ".txt":
                text = file.read().decode('utf-8')
            elif file_extension == ".pdf":
                text = extract_text_from_pdf(file)
            elif file_extension == ".docx":
                text = extract_text_from_docx(file)
            else:
                return jsonify({"error": "Unsupported file format."}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Check for text input in the form
    if 'textInput' in request.form:
        text = request.form['textInput']

    # If no text was provided, return an error
    if text is None or not text.strip():
        return jsonify({"error": "No text or file uploaded."}), 400

    # Preprocess the extracted text
    text = preprocess_text(text)

    # Split the text into chunks to avoid size mismatch
    text_chunks = chunk_text(text, max_length=512)

    # Use parallel processing to handle each chunk
    summaries = []
    sentiments = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(analyze_chunk, text_chunks))

        for summary, sentiment in results:
            summaries.append(summary)
            sentiments.append(sentiment)

    # Run topic modeling on the entire text
    topics = lda_topic_modeling(text)

    # Combine results
    summarized_text = ' '.join(summaries)
    combined_sentiment = {
        "label": sentiments[0]['label'],
        "score": sum([s['score'] for s in sentiments]) / len(sentiments)
    }

    # Keyword extraction on full text
    keywords = extract_keywords(text)

    return jsonify({
        "summary": summarized_text,
        "sentiment": combined_sentiment,
        "keywords": keywords,
        "topics": topics  # Now returning topic sentences without commas
    })

if __name__ == '__main__':
    app.run(debug=True)
