#plt.switch_backend('Agg')

from flask import Flask, render_template, request, send_file, jsonify,redirect,url_for
import pandas as pd
import re
import sqlite3
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid threading issues
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import io
import base64
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('punkt_tab')

app = Flask(__name__)

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()


# Preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Sentiment analysis function
def get_sentiment(text):
    sentiment_score = sia.polarity_scores(text)['compound']
    if sentiment_score > 0.05:
        return 'Positive', '😊'
    elif sentiment_score < -0.05:
        return 'Negative', '😡'
    else:
        return 'Neutral', '😐'

# Load and preprocess initial data
def load_initial_data():
    csv_folder = 'csv'
    csv_files = [
        os.path.join(csv_folder, 'eBay.csv'),
        os.path.join(csv_folder, 'walmart-products.csv'),
        os.path.join(csv_folder, 'amazon-products.csv'),
        os.path.join(csv_folder, 'shein-products.csv'),
        os.path.join(csv_folder, 'shopee-products.csv'),
        os.path.join(csv_folder, 'lazada-products.csv'),
        os.path.join(csv_folder, 'Zara - Products.csv')
    ]
    combined_df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
    review_columns = ['reviews', 'top_reviews', 'customer_reviews', 'item_reviews', 'most_relevant_reviews']
    df_reviews = combined_df[[col for col in review_columns if col in combined_df.columns]].copy()
    df_reviews['all_reviews'] = df_reviews.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    df_reviews['cleaned_reviews'] = df_reviews['all_reviews'].apply(preprocess_text)
    df_reviews['sentiment'], df_reviews['emoji'] = zip(*df_reviews['cleaned_reviews'].apply(get_sentiment))
    df_reviews['review_length'] = df_reviews['cleaned_reviews'].apply(lambda x: len(str(x).split()))
    df_reviews.to_csv('reviews_preprocessed.csv', index=False)
    return df_reviews

# Generate plot and return base64 string
def plot_to_base64():
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be a CSV'}), 400

    # Create testing_csv_files folder
    csv_folder = 'testing_csv_files'
    os.makedirs(csv_folder, exist_ok=True)

    # Generate filename with timestamp
    base_filename = os.path.splitext(file.filename)[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f"{base_filename}_{timestamp}.csv"
    csv_path = os.path.join(csv_folder, csv_filename)

    # Handle duplicate filenames
    counter = 0
    while os.path.exists(csv_path):
        counter += 1
        csv_filename = f"{base_filename}_{timestamp}_{counter}.csv"
        csv_path = os.path.join(csv_folder, csv_filename)

    # Save CSV
    file.seek(0)  # Reset file pointer
    file.save(csv_path)

    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        print("Uploaded CSV columns:", df.columns.tolist())

        review_columns = [
            'reviews', 'top_reviews', 'customer_reviews', 'item_reviews', 'most_relevant_reviews',
            'review_text', 'feedback', 'comment', 'description', 'text'
        ]
        available_columns = [col for col in review_columns if col in df.columns]

        if not available_columns:
            return jsonify({
                'error': 'No recognizable review columns found.',
                'available_columns': df.columns.tolist()
            }), 400

        df['all_reviews'] = df[available_columns].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
        df['cleaned_reviews'] = df['all_reviews'].apply(preprocess_text)
        df['sentiment'], df['emoji'] = zip(*df['cleaned_reviews'].apply(get_sentiment))
        df['review_length'] = df['cleaned_reviews'].apply(lambda x: len(str(x).split()))

        df_for_db = pd.DataFrame({
            'review_text': df['cleaned_reviews'],
            'sentiment': df['sentiment']
        })
        if 'timestamp' in df.columns:
            df_for_db['timestamp'] = df['timestamp']

        # Save to reviews.db
        conn = sqlite3.connect('reviews.db')
        df_for_db.to_sql('reviews', conn, if_exists='append', index=False)
        conn.close()

        return jsonify({'message': 'File uploaded and processed successfully'})

    except pd.errors.EmptyDataError:
        return jsonify({'error': 'CSV file is empty or malformed'}), 400
    except pd.errors.ParserError:
        return jsonify({'error': 'CSV file is invalid or corrupted'}), 400
    except Exception as e:
        return jsonify({'error': f'Error processing CSV: {str(e)}'}), 500
    
@app.route('/visual')
def visual():
    return render_template('visual.html')

@app.route('/analyze', methods=['GET'])
def analyze():
    df = pd.read_csv('reviews_preprocessed.csv')
    sentiment_counts = df['sentiment'].value_counts().to_dict()
    return jsonify(sentiment_counts)

@app.route('/visualizations', methods=['GET'])
def visualizations():
    df = pd.read_csv('reviews_preprocessed.csv')
    os.makedirs('static/images', exist_ok=True)

    # Sentiment Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='sentiment', palette='coolwarm', hue='sentiment', legend=False)
    plt.title("Sentiment Distribution of Reviews")
    sentiment_plot = plot_to_base64()
    plt.close()

    # Word Cloud
    text = ' '.join(df['cleaned_reviews'].fillna(''))  # Replace NaN with empty string
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Most Common Words in Reviews")
    wordcloud_img = plot_to_base64()
    plt.close()

    # Top 10 Frequent Words
    all_words = ' '.join(df['cleaned_reviews'].fillna('')).split()  # Replace NaN with empty string
    word_freq = Counter(all_words).most_common(10)
    words, counts = zip(*word_freq)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(counts), y=list(words), palette='viridis')
    plt.title("Top 10 Most Frequent Words")
    top_words_img = plot_to_base64()
    plt.close()

    # Review Length Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['review_length'].fillna(0), bins=30, kde=True, color='blue')  # Replace NaN with 0
    plt.title("Distribution of Review Lengths")
    review_length_img = plot_to_base64()
    plt.close()

    # Sentiment vs Review Length
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='sentiment', y='review_length', palette='coolwarm', hue='sentiment', legend=False)
    plt.title("Sentiment vs. Review Length")
    sentiment_vs_length_img = plot_to_base64()
    plt.close()

    # Top Bigrams
    def get_top_ngrams(corpus, ngram_range=(2, 2), n=10):
        vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
        X = vectorizer.fit_transform(corpus.fillna(''))  # Replace NaN with empty string
        ngram_counts = X.toarray().sum(axis=0)
        ngram_freq = dict(zip(vectorizer.get_feature_names_out(), ngram_counts))
        return sorted(ngram_freq.items(), key=lambda x: x[1], reverse=True)[:n]

    top_bigrams = get_top_ngrams(df['cleaned_reviews'], (2, 2))
    plt.figure(figsize=(10, 5))
    sns.barplot(x=[x[1] for x in top_bigrams], y=[x[0] for x in top_bigrams], palette='magma')
    plt.title("Top 10 Bigrams in Reviews")
    bigrams_img = plot_to_base64()
    plt.close()

    # Top Trigrams
    top_trigrams = get_top_ngrams(df['cleaned_reviews'], (3, 3))
    plt.figure(figsize=(10, 5))
    sns.barplot(x=[x[1] for x in top_trigrams], y=[x[0] for x in top_trigrams], palette='viridis')
    plt.title("Top 10 Trigrams in Reviews")
    trigrams_img = plot_to_base64()
    plt.close()

    # Positive and Negative Words
    positive_words, negative_words = [], []
    for review in df['cleaned_reviews'].fillna(''):  # Replace NaN with empty string
        words = word_tokenize(str(review))
        for word in words:
            score = sia.polarity_scores(word)['compound']
            if score > 0.2:
                positive_words.append(word)
            elif score < -0.2:
                negative_words.append(word)
    
    positive_counts = Counter(positive_words).most_common(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=[x[1] for x in positive_counts], y=[x[0] for x in positive_counts], palette='Blues')
    plt.title("Top 10 Positive Words")
    positive_words_img = plot_to_base64()
    plt.close()

    negative_counts = Counter(negative_words).most_common(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=[x[1] for x in negative_counts], y=[x[0] for x in negative_counts], palette='Reds')
    plt.title("Top 10 Negative Words")
    negative_words_img = plot_to_base64()
    plt.close()

    return jsonify({
        'sentiment_plot': sentiment_plot, 'wordcloud': wordcloud_img, 'top_words': top_words_img,
        'review_length': review_length_img, 'sentiment_vs_length': sentiment_vs_length_img,
        'bigrams': bigrams_img, 'trigrams': trigrams_img, 'positive_words': positive_words_img,
        'negative_words': negative_words_img
    })
def save_and_encode_plot(filename):
    # Save the image to static/images
    output_path = os.path.join('static/images', filename)
    plt.savefig(output_path, format='png', bbox_inches='tight')
    
    # Also convert it to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return encoded

sentiment_plot = save_and_encode_plot('sentiment_distribution.png')
wordcloud_img = save_and_encode_plot('wordcloud.png')
top_words_img = save_and_encode_plot('top_words.png')
review_length_img = save_and_encode_plot('review_length.png')
sentiment_vs_length_img = save_and_encode_plot('sentiment_vs_length.png')
bigrams_img = save_and_encode_plot('bigrams.png')
trigrams_img = save_and_encode_plot('trigrams.png')
positive_words_img = save_and_encode_plot('positive_words.png')
negative_words_img = save_and_encode_plot('negative_words.png')


sia = SentimentIntensityAnalyzer()
from collections import Counter

@app.route('/product', methods=['GET', 'POST'])
def product():
    
    conn = sqlite3.connect('reviews.db')
    c = conn.cursor()

    # 1. Get all reviews first for counts
    c.execute("SELECT sentiment FROM reviews")
    all_sentiments = [row[0] for row in c.fetchall()]
    sentiment_counts = Counter(all_sentiments)
    total_reviews = len(all_sentiments)

    counts = {
        "All": total_reviews,
        "Positive": sentiment_counts.get("Positive", 0),
        "Negative": sentiment_counts.get("Negative", 0),
        "Neutral": sentiment_counts.get("Neutral", 0)
    }

    # 2. Now filter reviews for display
    query = "SELECT id, review_text, sentiment FROM reviews WHERE 1=1"
    params = []

    if request.method == 'POST':
        sentiment = request.form.get('sentiment')
        keywords = request.form.get('keywords')
        if sentiment and sentiment != 'All':
            query += " AND sentiment = ?"
            params.append(sentiment)
        if keywords:
            query += " AND review_text LIKE ?"
            params.append(f'%{keywords}%')

    sort_by = request.form.get('sort_by', 'id')
    sort_order = request.form.get('sort_order', 'ASC')
    sort_by = 'id' if sort_by not in ['id', 'sentiment'] else sort_by
    sort_order = 'ASC' if sort_order not in ['ASC', 'DESC'] else sort_order
    query += f" ORDER BY {sort_by} {sort_order}"
    

    c.execute(query, params)
    reviews = c.fetchall()
    search_count = len(reviews)
    conn.close()

    return render_template('product.html', reviews=reviews, counts=counts, search_count=search_count)

import logging

logging.basicConfig(filename='sentimental.txt', level=logging.DEBUG)

@app.route('/analyze_sentence', methods=['POST'])
def analyze_sentence():
    sentence = request.form.get('sentence', '').strip()
    
    if not sentence:
        return jsonify({'error': 'No sentence provided'}), 400

    logging.info(f"Received sentence: {sentence}")

    sentiment_score = sia.polarity_scores(sentence)['compound']
    sentence_sentiment = 'Positive' if sentiment_score > 0.05 else 'Negative' if sentiment_score < -0.05 else 'Neutral'

    logging.info(f"Sentiment score: {sentiment_score}, Sentiment: {sentence_sentiment}")

    conn = sqlite3.connect('reviews.db')
    c = conn.cursor()
    c.execute("SELECT review_text, sentiment FROM reviews")
    reviews = c.fetchall()
    conn.close()

    logging.info(f"Fetched {len(reviews)} reviews from database")

    if not reviews:
        return jsonify({
            'sentence': sentence,
            'sentiment': sentence_sentiment,
            'matching_reviews': [],
            'message': 'No reviews available for comparison'
        })

    documents = [sentence] + [review[0] for review in reviews if review[0]]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    matching_reviews = []
    for i, (review_text, review_sentiment) in enumerate(reviews):
        if not review_text:
            continue
        if i < len(similarity_scores):
            similarity = similarity_scores[i]
            if sentence_sentiment == review_sentiment or similarity > 0.2:
                matching_reviews.append({
                    'review': review_text,
                    'sentiment': review_sentiment,
                    'similarity': round(similarity, 2)
                })

    logging.info(f"Matching reviews found: {len(matching_reviews)}")

    return jsonify({
        'sentence': sentence,
        'sentiment': sentence_sentiment,
        'matching_reviews': matching_reviews
    })
load_initial_data()
