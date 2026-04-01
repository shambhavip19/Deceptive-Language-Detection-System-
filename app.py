import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Load models and vectorizer
nb_model = pickle.load(open("nb_model.pkl", "rb"))
lr_model = pickle.load(open("lr_model.pkl", "rb"))
svm_model = pickle.load(open("svm_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))

# Same cleaning function from before
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    filtered = [w for w in words if w not in stop_words]
    return ' '.join(filtered)

# App UI
st.title("🔍 Fake Review Detector")
st.write("Enter a product review below and see what all 3 models think!")

review = st.text_area("✍️ Enter your review here:", height=150)

if st.button("Detect!"):
    if review.strip() == "":
        st.warning("Please enter a review first!")
    else:
        cleaned = clean_text(review)
        vectorized = vectorizer.transform([cleaned])

        results = {
            "Naive Bayes": nb_model.predict(vectorized)[0],
            "Logistic Regression": lr_model.predict(vectorized)[0],
            "SVM": svm_model.predict(vectorized)[0]
        }

        st.subheader("📊 Results:")
        for model_name, prediction in results.items():
            if prediction == "CG":
                st.error(f"❌ {model_name}: FAKE Review")
            else:
                st.success(f"✅ {model_name}: GENUINE Review")
