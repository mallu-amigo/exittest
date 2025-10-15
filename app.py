import os
import numpy as np
import streamlit as st

# Gracefully handle missing optional dependencies so the app shows a friendly message
try:
    import joblib
except Exception:
    joblib = None

# gensim imports (may be optional if you only use TF-IDF model)
try:
    from gensim.utils import simple_preprocess
    from gensim.models import Word2Vec
    gensim_available = True
except Exception:
    simple_preprocess = None
    Word2Vec = None
    gensim_available = False

st.set_page_config(page_title="Sentiment Predictor", layout="centered")

# Early check: joblib is required to load saved sklearn/gensim artifacts
if joblib is None:
    st.title("Product Review Sentiment Predictor")
    st.error("Missing dependency: `joblib` is not installed in this environment.")
    st.write("To fix this locally, run:")
    st.code("python -m pip install joblib", language="bash")
    st.write("If you are deploying (Streamlit Cloud / other), add `joblib` to requirements.txt and redeploy.")
    st.stop()

# FILES IN PROJECT ROOT (no models/ folder)
TFIDF_LR_FILENAME = "tfidf_lr.joblib"
W2V_FILENAME = "w2v.model"
RF_W2V_FILENAME = "rf_w2v.joblib"

st.title("Product Review Sentiment Predictor")
st.write("Enter a product review below and pick a model to predict sentiment (Positive / Negative).")

available_models = []
if os.path.exists(TFIDF_LR_FILENAME):
    available_models.append("TF-IDF + LogisticRegression")
if os.path.exists(W2V_FILENAME) and os.path.exists(RF_W2V_FILENAME) and gensim_available:
    available_models.append("Word2Vec(avg) + RandomForest")
elif os.path.exists(W2V_FILENAME) and os.path.exists(RF_W2V_FILENAME) and not gensim_available:
    available_models.append("Word2Vec(avg) + RandomForest (gensim missing)")

if not available_models:
    st.error(
        "No saved model files found in the project root. Please ensure the files "
        f"'{TFIDF_LR_FILENAME}', '{W2V_FILENAME}', and '{RF_W2V_FILENAME}' are present next to app.py."
    )
    st.stop()

model_choice = st.sidebar.selectbox("Choose model", available_models)

# Lazy load when needed
tfidf_pipeline = None
w2v_model = None
rf_w2v = None

if model_choice.startswith("TF-IDF") and os.path.exists(TFIDF_LR_FILENAME):
    try:
        data = joblib.load(TFIDF_LR_FILENAME)
        tfidf_pipeline = data  # expected dict with 'vectorizer' and 'clf'
    except Exception as e:
        st.error(f"Failed to load TF-IDF pipeline: {e}")
        st.stop()
elif model_choice.startswith("Word2Vec"):
    if not gensim_available:
        st.error("gensim is not available in this environment. Install gensim to use Word2Vec model.")
        st.stop()
    try:
        w2v_model = Word2Vec.load(W2V_FILENAME)
    except Exception as e:
        st.error(f"Failed to load Word2Vec model ({W2V_FILENAME}): {e}")
        st.stop()
    try:
        rf_w2v = joblib.load(RF_W2V_FILENAME)
    except Exception as e:
        st.error(f"Failed to load RandomForest artifact ({RF_W2V_FILENAME}): {e}")
        st.stop()

st.markdown("---")
user_text = st.text_area("Paste or type a product review here:", height=150)

if st.button("Predict"):
    text = (user_text or "").strip()
    if not text:
        st.warning("Please enter a review to predict.")
    else:
        if model_choice.startswith("TF-IDF") and tfidf_pipeline is not None:
            vectorizer = tfidf_pipeline.get("vectorizer")
            clf = tfidf_pipeline.get("clf")
            if vectorizer is None or clf is None:
                st.error("TF-IDF pipeline is missing expected components ('vectorizer'/'clf').")
            else:
                X = vectorizer.transform([text])
                proba = clf.predict_proba(X)[0, 1] if hasattr(clf, "predict_proba") else None
                pred = clf.predict(X)[0]
                label = "Positive" if int(pred) == 1 else "Negative"
                st.success(f"Predicted sentiment: {label}")
                if proba is not None:
                    st.write(f"Confidence (pos probability): {proba:.3f}")
        elif model_choice.startswith("Word2Vec") and w2v_model is not None and rf_w2v is not None:
            tokens = simple_preprocess(text, deacc=True)
            vector_size = rf_w2v.get("vector_size", getattr(w2v_model, "vector_size", None))
            vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv.key_to_index]
            if len(vecs) == 0:
                st.warning("No in-vocabulary tokens found for this input. Cannot compute averaged embedding.")
            else:
                avg_vec = np.mean(vecs, axis=0).reshape(1, -1)
                clf = rf_w2v.get("clf") if isinstance(rf_w2v, dict) else rf_w2v
                if clf is None:
                    st.error("RandomForest classifier not found in saved artifact.")
                else:
                    proba = clf.predict_proba(avg_vec)[0, 1] if hasattr(clf, "predict_proba") else None
                    pred = clf.predict(avg_vec)[0]
                    label = "Positive" if int(pred) == 1 else "Negative"
                    st.success(f"Predicted sentiment: {label}")
                    if proba is not None:
                        st.write(f"Confidence (pos probability): {proba:.3f}")
        else:
            st.error("Model artifacts not available or failed to load.")

st.markdown("---")
st.write(
    "Tip: If this app cannot load models, ensure the model files are in the project root and that "
    "`joblib` and `gensim` are listed in requirements.txt."
)