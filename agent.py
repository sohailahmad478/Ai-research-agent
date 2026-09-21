"""
agent.py - the "brain" of the app (CrewAI part).

One agent + one task + one search tool:
  topic in  ->  agent searches DuckDuckGo  ->  report + research gaps out
"""
import os

# Turn off CrewAI telemetry (must be set BEFORE importing crewai)
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool
from ddgs import DDGS

# ---------------------------------------------------------------------------
# Settings (change these in one place)
# ---------------------------------------------------------------------------
GROQ_MODEL = "openai/gpt-oss-120b"          # the model name on Groq
GROQ_BASE_URL = "https://api.groq.com/openai/v1"  # Groq's OpenAI-compatible endpoint
MAX_SEARCH_RESULTS = 5                      # results per search (keep small: saves tokens)


# ---------------------------------------------------------------------------
# 1) The search tool (free, no API key needed)
# ---------------------------------------------------------------------------
@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """Search the web with DuckDuckGo. Input must be a short search query string.
    Returns a numbered list with title, URL and a short snippet for each result."""
    try:
        results = DDGS().text(query, max_results=MAX_SEARCH_RESULTS)
    except Exception as e:  # DuckDuckGo can rate-limit; don't crash the app
        return f"Search failed ({e}). Try again with a different query."

    if not results:
        return "No results found. Try a different query."

    lines = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "")
        url = r.get("href", "")
        snippet = (r.get("body", "") or "")[:300]
        lines.append(f"{i}. {title}\n   URL: {url}\n   Snippet: {snippet}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2) The LLM (Groq, through its OpenAI-compatible API)
# ---------------------------------------------------------------------------
def build_llm(groq_api_key: str) -> LLM:
    return LLM(
        # "openai/" tells CrewAI to use its built-in OpenAI-style client.
        # The rest ("openai/gpt-oss-120b") is the real model name on Groq.
        model=f"openai/{GROQ_MODEL}",
        base_url=GROQ_BASE_URL,
        api_key=groq_api_key,
        temperature=0.3,
        max_tokens=4096,
    )


# ---------------------------------------------------------------------------
# 3) Build the agent, the task, and run the crew
# ---------------------------------------------------------------------------
def run_research(topic: str, groq_api_key: str) -> str:
    llm = build_llm(groq_api_key)

    researcher = Agent(
        role="Senior Research Analyst",
        goal=(
            "Research a topic using web search, write a clear structured report, "
            "and identify genuine research gaps."
        ),
        backstory=(
            "You are an experienced academic researcher. You only state facts you "
            "found in search results, you never invent references, and you are "
            "good at spotting what has NOT been studied yet."
        ),
        tools=[duckduckgo_search],
        llm=llm,
        allow_delegation=False,  # single-agent app
        max_iter=8,              # max reasoning/search steps (protects your token limit)
        verbose=False,
    )

    task = Task(
        description=(
            "Research the topic: '{topic}'.\n\n"
            "Steps:\n"
            "1. Run 3 to 5 different DuckDuckGo searches (overview, recent work, "
            "methods, challenges, limitations/future work).\n"
            "2. Write the report ONLY from what the search results support.\n"
            "3. Finish with a research gap analysis.\n"
        ),
        expected_output=(
            "A Markdown report with these sections:\n"
            "# <Title>\n"
            "## 1. Introduction\n"
            "## 2. Background and Key Concepts\n"
            "## 3. Current Approaches and Recent Developments\n"
            "## 4. Challenges and Limitations\n"
            "## 5. Research Gaps (3 to 5 gaps; for each: what is missing, why it "
            "matters, and a possible research direction)\n"
            "## 6. Conclusion\n"
            "## References (only URLs that appeared in your search results)"
        ),
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff(inputs={"topic": topic})
    return result.raw  # the final report as text
