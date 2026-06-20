import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import shap
from lime.lime_text import LimeTextExplainer

# 1. Load dataset
df = pd.read_csv("data/data.csv")

print("Columns in dataset:", df.columns)

# 2. Features and labels
X = df["text"]   # input features
y = df["subject"]  # FAKE/REAL labels

# 3. Convert text to features
vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)

# 5. Train classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. SHAP explainability (fixed handling)
# 7. SHAP explainability (fixed handling for ndarray)
explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test)

# Pick the first test sample
sample_index = 0
sample_shap_values = shap_values[sample_index].flatten()  # no .toarray()

print("\nTop SHAP features for first test sample:")
feature_names = np.array(vectorizer.get_feature_names_out())
top_features = np.argsort(sample_shap_values)[-5:]
for i in top_features:
    print(feature_names[i], sample_shap_values[i])


# 8. LIME explainability
class_names = list(df["subject"].unique())
lime_explainer = LimeTextExplainer(class_names=class_names)

def predict_proba(texts):
    return model.predict_proba(vectorizer.transform(texts))

exp = lime_explainer.explain_instance(X.iloc[0], predict_proba, num_features=5)
print("\nLIME explanation for first sample:")
print(exp.as_list())
