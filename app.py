"""
app.py - the Streamlit web page (UI part).
The Groq API key is read ONLY from Streamlit Secrets (GROQ_API_KEY).
"""
import streamlit as st

from agent import run_research

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="wide")
st.title("🔎 AI Research Agent")
st.caption("CrewAI + Groq (gpt-oss-120b) + DuckDuckGo Search")

# ----- Read the API key from Streamlit Secrets -----
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = None

if not api_key:
    st.error(
        "GROQ_API_KEY not found. In Streamlit Cloud go to "
        "App settings -> Secrets and add:  GROQ_API_KEY = \"gsk_...\""
    )
    st.stop()  # stop here so the rest of the page doesn't run

# ----- Sidebar -----
with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. You enter a topic\n"
        "2. The agent searches the web\n"
        "3. It writes a report + research gaps"
    )

# ----- Main area -----
topic = st.text_input(
    "Research topic",
    placeholder="e.g. Contactless heart rate monitoring using radar",
)

if st.button("Generate report", type="primary"):
    if not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        with st.spinner("Agent is searching and writing... this can take 1-2 minutes."):
            try:
                report = run_research(topic.strip(), api_key)
                st.session_state["report"] = report
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# Show the last report (stays visible even after the page reruns)
if "report" in st.session_state:
    st.divider()
    st.markdown(st.session_state["report"])
    st.download_button(
        "Download report (.md)",
        data=st.session_state["report"],
        file_name="research_report.md",
        mime="text/markdown",
    )
