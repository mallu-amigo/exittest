import sys, importlib, traceback
import streamlit as st

st.title("Runtime debug for packages")

st.subheader("Python")
st.write(sys.executable)
st.write(sys.version)

st.subheader("Check imports and versions")
pkgs = ["gensim", "joblib", "numpy", "scikit_learn", "sklearn", "streamlit", "tornado"]
for pkg in ["gensim", "joblib", "numpy", "sklearn", "streamlit", "tornado"]:
    try:
        m = importlib.import_module(pkg)
        ver = getattr(m, "__version__", "unknown")
        st.success(f"{pkg} imported: {ver}")
    except Exception as e:
        st.error(f"{pkg} import FAILED: {e}")
        st.text(traceback.format_exc())

st.subheader("Pip installed packages (first 200 chars)")
try:
    import pkg_resources, json
    installed = sorted([f"{p.key}=={p.version}" for p in pkg_resources.working_set])
    st.write(installed[:50])
except Exception as e:
    st.write("pkg listing failed:", e)
