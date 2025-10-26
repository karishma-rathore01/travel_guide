import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK resources are available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Load dataset
data = pd.read_csv("travel_destinations.csv")

# Preprocessing
stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(filtered)

# Apply preprocessing to descriptions
data["processed"] = data["description"].apply(preprocess)

# Vectorize descriptions
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data["processed"])

def recommend_destination(user_input):
    # Preprocess user input
    processed_input = preprocess(user_input)
    input_vec = vectorizer.transform([processed_input])
    
    # Calculate similarity
    similarity = cosine_similarity(input_vec, tfidf_matrix)
    
    # Get best match
    index = similarity.argmax()
    score = similarity[0][index]
    
    return data.iloc[index]["destination"], data.iloc[index]["description"], score

# Main loop
print("🌍 Welcome to the AI Travel Guide!")
print("Tell me what kind of destination you’re looking for (type 'exit' to quit).\n")

while True:
    user_query = input("Your travel preference: ")
    if user_query.lower() in ["exit", "quit"]:
        print("👋 Goodbye! Have a safe journey!")
        break
    
    place, desc, score = recommend_destination(user_query)
    print(f"\n✨ Suggested Destination: {place}")
    print(f"📌 Why: {desc}")
    print(f"🔍 Match Score: {round(score*100,2)}%\n")
