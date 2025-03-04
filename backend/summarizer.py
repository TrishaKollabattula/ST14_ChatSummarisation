from transformers import pipeline

# Load the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_chat(chat_text):
    summary = summarizer(chat_text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']
