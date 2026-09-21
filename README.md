# AI Research Agent (CrewAI + Groq + Streamlit)

Enter a research topic -> a single CrewAI agent searches DuckDuckGo -> you get a report and research gaps.

## Deploy on Streamlit Cloud
1. Upload these files to a GitHub repo (main branch).
2. share.streamlit.io -> Create app -> pick repo, branch `main`, main file `app.py`.
3. Advanced settings -> Python version 3.12, and paste in Secrets:
   `GROQ_API_KEY = "gsk_..."`
4. Deploy.
