## Deceptive Language Detection using NLP and Machine Learning
Classifying product reviews as deceptive or genuine using traditional NLP techniques and supervised learning.

---

### Dataset
Fake Reviews Dataset with 40,000 review records containing product review text and binary labels indicating whether a review is original or computer generated.

Source: Kaggle

Features used: review text after cleaning and vectorization using TF-IDF with 5000 features

Target column: label, where OR means Original and CG means Computer Generated

---

### Workflow
1. Load and explore the data, shape, label distribution, column names
2. Clean raw review text by lowercasing and removing punctuation and numbers
3. Remove stopwords to eliminate common words with no signal value
4. Vectorize cleaned text using TF-IDF to convert text into numerical features
5. Split data into 80% training and 20% testing sets
6. Train three models and evaluate each using accuracy, precision, recall and F1
7. Visualize confusion matrices for all three models to analyze errors
8. Export trained models and vectorizer, then build and deploy a Streamlit web app

---

### Models Used

Naive Bayes
- Baseline model using word probability independently per class
- Fast and simple but assumes all words are independent of each other
- Accuracy: 85.3%

Logistic Regression
- Learns weights for each word and considers them together during prediction
- More expressive than Naive Bayes and handles word relationships better
- Accuracy: 88.0%

SVM
- Finds the optimal decision boundary between deceptive and genuine reviews
- Focuses on hard to classify reviews near the boundary for better generalization
- Accuracy: 88.3%

---

### Results

| Model | Accuracy | F1 Score | Correct | Wrong |
|---|---|---|---|---|
| Naive Bayes | 85.3% | 0.85 | 6897 | 1190 |
| Logistic Regression | 88.0% | 0.88 | 7118 | 969 |
| SVM | 88.3% | 0.88 | 7144 | 943 |

SVM gave the best overall performance with the least misclassifications. Logistic Regression was a close second with the same F1 score. Naive Bayes performed well considering its simplicity but had the highest false alarm rate, flagging 683 genuine reviews as deceptive.

---

### Libraries

| Library | Purpose |
|---|---|
| pandas, numpy | Data handling and manipulation |
| re | Text cleaning using regular expressions |
| nltk | Stopword removal |
| scikit-learn | TF-IDF vectorization, model training, evaluation, confusion matrix |
| matplotlib | Confusion matrix visualization |
| pickle | Saving and loading trained models |
| streamlit | Building and deploying the web app |

---

### What I Learned
- How raw text is cleaned, vectorized and converted into something a model can learn from
- Why TF-IDF is effective at capturing meaningful words over common noise
- How three different models approach the same classification problem differently
- That model accuracy on test data does not always reflect real world behavior
- How to export trained models and build a working multi model web app using Streamlit
