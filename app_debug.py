# name=app_debug.py
import os
import streamlit as st
import sys
st.set_page_config(page_title="App Debug", layout="centered")

st.title("App Debugging helper")

# Environment info
st.subheader("Python & packages")
st.write("Python:", sys.version.splitlines()[0])
try:
    import joblib
    st.write("joblib:", joblib.__version__)
except Exception as e:
    st.write("joblib: MISSING ->", e)

try:
    import gensim
    st.write("gensim:", gensim.__version__)
    from gensim.utils import simple_preprocess
    from gensim.models import Word2Vec
except Exception as e:
    st.write("gensim: MISSING or error ->", e)

try:
    import numpy as np
    st.write("numpy:", np.__version__)
except Exception as e:
    st.write("numpy: MISSING ->", e)

st.subheader("Files present")
files = os.listdir(".")
st.write(files)

# Check model files existence & sizes
for fname in ("tfidf_lr.joblib", "w2v.model", "rf_w2v.joblib"):
    if os.path.exists(fname):
        try:
            size = os.path.getsize(fname)
            st.write(f"{fname} — present — {size/1024/1024:.2f} MB")
        except Exception as e:
            st.write(f"{fname} — present but cannot stat: {e}")
    else:
        st.write(f"{fname} — NOT FOUND")

# Try quick model load when clicking button
if st.button("Try loading models now"):
    # Load w2v
    if os.path.exists("w2v.model"):
        try:
            st.write("Loading Word2Vec...")
            from gensim.models import Word2Vec
            w2v = Word2Vec.load("w2v.model")
            st.write("Word2Vec loaded OK, vector_size:", getattr(w2v, "vector_size", getattr(getattr(w2v, 'wv', None), 'vector_size', None)))
        except Exception as e:
            st.error("Failed to load w2v.model: " + str(e))
    else:
        st.warning("w2v.model not present")

    # Load RF
    if os.path.exists("rf_w2v.joblib"):
        try:
            st.write("Loading RandomForest artifact...")
            import joblib
            rf_data = joblib.load("rf_w2v.joblib")
            st.write("RF artifact loaded. Type:", type(rf_data))
            clf = rf_data.get("clf") if isinstance(rf_data, dict) else rf_data
            st.write("Classifier type:", type(clf))
        except Exception as e:
            st.error("Failed to load rf_w2v.joblib: " + str(e))
    else:
        st.warning("rf_w2v.joblib not present")