import pickle 
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier as knn
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import pandas as pd
import numpy as np
import nltk
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app) 
# Load the model
model = pickle.load(open('similarity1.pkl', 'rb'))

# Load the attractions and preferences data
attractions = pd.read_csv(r"C:\Users\Bogere\OneDrive\Desktop\Tourism\tourism_attractions.csv")
preferences = pd.read_csv(r"C:\Users\Bogere\OneDrive\Desktop\Tourism\userA_preferences.csv")
attractions = attractions[['item_id', 'name', 'experience_tags']]

# Normalize the documents
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
def normalize_document(document):
    document = re.sub(r'[^a-zA-Z0-9\s]', '', document, re.I|re.A)
    document = document.lower()
    document = document.strip()
    tokens = nltk.word_tokenize(document)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    document = ' '.join(filtered_tokens)
    return document
normalize_corpus = np.vectorize(normalize_document)

norm_corpus_attractions = normalize_corpus(list(attractions['experience_tags']))
norm_corpus_preferences = normalize_corpus(list(preferences['preferences']))

# Compute the cosine similarity scores
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
tfidf_matrix_attractions = tfidf_vectorizer.fit_transform(norm_corpus_attractions)
tfidf_matrix_preferences = tfidf_vectorizer.transform(norm_corpus_preferences)
cosine_similarity = cosine_similarity(tfidf_matrix_attractions, tfidf_matrix_preferences)
df_cosinesimilarity = pd.DataFrame(cosine_similarity)
df_cosinesimilarity.index = df_cosinesimilarity.index +1
df_cosinesimilarity.index.name = 'item_id'
df_cosinesimilarity = df_cosinesimilarity.rename(columns={0: 'similarity_score'})

@app.route('/', methods=['GET'])
def index():
     if request.method == 'GET':
        # Get the selected checkboxes from the user input
        selected_preferences = request.form.getlist('title')

        # Normalize the selected preferences
        norm_selected_preferences = normalize_corpus(selected_preferences)

        # Compute the cosine similarity scores
        tfidf_matrix_selected_preferences = tfidf_vectorizer.transform(norm_selected_preferences)
        cosine_similarity_selected = cosine_similarity(tfidf_matrix_attractions, tfidf_matrix_selected_preferences)
        df_cosinesimilarity_selected = pd.DataFrame(cosine_similarity_selected)
        df_cosinesimilarity_selected.index = df_cosinesimilarity_selected.index +1
        df_cosinesimilarity_selected.index.name = 'item_id'
        df_cosinesimilarity_selected = df_cosinesimilarity_selected.rename(columns={0: 'similarity_score'})

        # Merge the attractions data with the similarity scores
        attractions_with_similarity_scores = pd.merge(attractions, df_cosinesimilarity_selected, on='item_id')

        # Sort the recommendations by similarity score in descending order
        recommendations = attractions_with_similarity_scores.sort_values(by='similarity_score', ascending=False)

        # Select the top N recommendations
        N = 5
        top_recommendations = recommendations['name'].tolist()[:N]

        # Return the recommendations as a JSON object
        return jsonify({'recommendations': top_recommendations})


if __name__ == '__main__':
    app.run(debug=True)