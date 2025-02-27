import os
from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load the abstractive summarization model
abstractive_summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")

# Function for extractive summarization
def extractive_summarization(text, num_sentences=3):
    # Split the text into sentences
    sentences = text.split('. ')
    if len(sentences) < num_sentences:
        return text  # Return the full text if it's too short

    # Compute TF-IDF scores
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Compute cosine similarity between sentences
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Rank sentences based on their importance
    sentence_scores = similarity_matrix.sum(axis=1)
    ranked_sentences = [sentence for _, sentence in sorted(zip(sentence_scores, sentences), reverse=True)]

    # Select the top sentences
    summary = '. '.join(ranked_sentences[:num_sentences])
    return summary

@app.route('/')
def home():
    # Render the main UI
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    # Get the input text and summarization type from the frontend
    data = request.json
    text = data.get('text', '')
    summary_type = data.get('type', 'abstractive')  # Default to abstractive

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        if summary_type == 'abstractive':
            # Generate abstractive summary
            summary = abstractive_summarizer(text, max_length=100, min_length=30, do_sample=False)
            summary = summary[0]['summary_text']
        elif summary_type == 'extractive':
            # Generate extractive summary
            summary = extractive_summarization(text)
        else:
            return jsonify({"error": "Invalid summary type"}), 400

        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Bind to the PORT environment variable provided by Render (or default to 5000 for local development)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)