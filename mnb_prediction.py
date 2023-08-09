import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import joblib

# Load data from CSV file
data = pd.read_csv('google_play_store_apps_reviews_training.csv')

# Remove package_name column
data = data.drop('package_name', axis=1)

# Filter out records with polarity values other than 0 and 1
data = data[data['polarity'].isin([0, 1])]

# Vectorize text using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['review'])
y = data['polarity']


# Train Gaussian Naive Bayes model
gnb = MultinomialNB()
gnb.fit(X.toarray(), y)

# Save trained model to file
joblib.dump(gnb, 'gnb_model.joblib')

# Load trained model from file
gnb = joblib.load('gnb_model.joblib')

# Get user input for new text to predict sentiment for
input_text = input('Enter review text: ')

# Preprocess input text (lowercase, remove non-alphanumeric characters, tokenize)
input_text = input_text.lower().replace(r'[^a-zA-Z0-9\s]', '').split()

# Vectorize input text using CountVectorizer
input_vector = vectorizer.transform([' '.join(input_text)]).toarray()

# Predict sentiment for input text using trained model
predicted_sentiment = gnb.predict(input_vector)[0]

# Print predicted sentiment
if predicted_sentiment==0:
    print('Negative')
else:
    print('Positive')