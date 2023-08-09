import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
data = pd.read_csv('google_play_store_apps_reviews_training.csv')

data.head()
import sklearn
print(data.polarity.value_counts())


from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts = cv.fit_transform(data['review'])

def preprocess_data(data):
    # Remove package name as it's not relevant
    data = data.drop('package_name', axis=1)

    # Convert text to lowercase
    data['review'] = data['review'].str.strip().str.lower()
    return data



data = preprocess_data(data)

# Split into training and testing data
x = data['review']
y = data['polarity']
x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.25, random_state=42)


# Vectorize text reviews to numbers
vec = CountVectorizer(stop_words='english',ngram_range = (1,1))
x = vec.fit_transform(x).toarray()
x_test = vec.transform(x_test).toarray()


from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB

model = GaussianNB()
model.fit(x, y)

print("Accuracy  of Gaussian Naive bayes = ")
print(model.score(x_test, y_test))


# Save model
joblib.dump(model, 'model.pkl')
