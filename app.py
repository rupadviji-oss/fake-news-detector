import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import shap
from lime.lime_text import LimeTextExplainer

# Load dataset
df = pd.read_csv("data.csv")
X = df["text"]
y = df["subject"]

# Train model
vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)
model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# Streamlit UI
st.title("📰 Fake News Detector")
st.write("Paste any news text below and check if it's FAKE or REAL.")

user_input = st.text_area("Enter news text:")

if st.button("Check News"):
    if user_input.strip() != "":
        # Predict
        X_user = vectorizer.transform([user_input])
        prediction = model.predict(X_user)[0]
        st.subheader(f"Result: {prediction}")

        # SHAP explanation
        explainer = shap.LinearExplainer(model, X_vec, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_user)
        feature_names = vectorizer.get_feature_names_out()
        st.write("Top SHAP features influencing this prediction:")
        top_features = shap_values[0].argsort()[-5:]
        for i in top_features:
            st.write(f"{feature_names[i]} → {shap_values[0][i]}")

        # LIME explanation
        class_names = list(df["subject"].unique())
        lime_explainer = LimeTextExplainer(class_names=class_names)

        def predict_proba(texts):
            return model.predict_proba(vectorizer.transform(texts))

        exp = lime_explainer.explain_instance(user_input, predict_proba, num_features=5)
        st.write("LIME explanation:")
        st.write(exp.as_list())
    else:
        st.warning("Please enter some text.")
