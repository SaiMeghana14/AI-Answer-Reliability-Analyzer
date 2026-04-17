import streamlit as st
import matplotlib.pyplot as plt
import requests
import re

from utils.generator_basic import generate_basic_answer
from utils.evaluator import compute_similarity, get_score
from utils.pdf_export import generate_pdf
from utils.advanced_eval import detect_hallucinations, explain_winner


# ------------------ CONFIG ------------------
st.set_page_config(page_title="AI Reliability Battle", layout="wide")

st.title("⚔️ AI Reliability Battle: Baseline vs Retrieval AI")


# ------------------ MULTI-SOURCE RETRIEVAL ------------------

def get_wikipedia_data(query):
    try:
        import wikipedia
        return wikipedia.summary(query, sentences=3), "Wikipedia"
    except:
        return "", "Wikipedia"


def get_duckduckgo_data(query):
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json"
        data = requests.get(url).json()
        return data.get("AbstractText") or data.get("RelatedTopics", [{}])[0].get("Text", "")
    except:
        return "", "DuckDuckGo"


def get_combined_data(query):
    wiki_data, wiki_src = get_wikipedia_data(query)
    ddg_data, ddg_src = get_duckduckgo_data(query)

    combined = f"{wiki_data} {ddg_data}".strip()

    sources = []
    if wiki_data:
        sources.append(("Wikipedia", f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"))
    if ddg_data:
        sources.append(("DuckDuckGo", "https://duckduckgo.com/?q=" + query))

    return combined if combined else "No reliable data found.", sources


# ------------------ RETRIEVAL AI ------------------

def generate_retrieval_answer(question):
    data, _ = get_combined_data(question)
    return f"Based on reliable sources: {data}"


# ------------------ WORD-LEVEL HIGHLIGHT ------------------

def highlight_words(answer, reference):
    answer_words = answer.split()
    ref_words = set(reference.lower().split())

    highlighted = ""

    for word in answer_words:
        clean_word = re.sub(r'[^\w]', '', word.lower())

        if clean_word in ref_words:
            highlighted += f"✅ {word} "
        else:
            highlighted += f"❌ {word} "

    return highlighted

# ------------------ GAUGE ------------------

def show_gauge(score, title):
    fig, ax = plt.subplots()
    ax.barh([0], [score])
    ax.set_xlim(0, 100)
    ax.set_title(title)
    ax.set_yticks([])
    st.pyplot(fig)


# ------------------ INPUT ------------------

question = st.text_input("🔍 Enter your question:")


# ------------------ MAIN ------------------

if st.button("Analyze") and question:

    # Generate answers
    basic_answer = generate_basic_answer(question)
    retrieval_answer = generate_retrieval_answer(question)

    # Get ground truth
    real_data, sources = get_combined_data(question)

    # Scores
    sim_basic = compute_similarity(basic_answer, real_data)
    sim_retrieval = compute_similarity(retrieval_answer, real_data)

    score_basic, conf_basic = get_score(sim_basic)
    score_retrieval, conf_retrieval = get_score(sim_retrieval)

    # Winner
    winner, reason = explain_winner(score_basic, score_retrieval,
                                    "Baseline AI", "Retrieval AI")

     # ------------------ DISPLAY ------------------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 Baseline AI (Ungrounded)")
        st.write(basic_answer)
        st.metric("Score", f"{score_basic}%")
        st.write(conf_basic)

    with col2:
        st.subheader("🌐 Retrieval AI (Knowledge-Grounded)")
        st.write(retrieval_answer)
        st.metric("Score", f"{score_retrieval}%")
        st.write(conf_retrieval)

    st.divider()

    # Ground truth
    st.subheader("📚 Ground Truth (Multi-Source)")
    st.write(real_data)

    st.write("🔗 Sources:")
    for name, link in sources:
        st.markdown(f"- [{name}]({link})")

    st.divider()

    # Winner
    st.subheader("🏆 Winner")
    st.success(winner)

    st.subheader("🧠 Why This Model Won")
    st.write(reason)

    st.divider()

    # Gauges
    st.subheader("🎯 Trust Score Gauges")

    col3, col4 = st.columns(2)

    with col3:
        show_gauge(score_basic, "Baseline AI Trust")

    with col4:
        show_gauge(score_retrieval, "Retrieval AI Trust")

    st.divider()

    # Word-level highlighting
    st.subheader("🔍 Word-Level Hallucination Detection")

    col5, col6 = st.columns(2)

    with col5:
        st.write("### 🤖 Baseline AI")
        st.write(highlight_words(basic_answer, real_data))

    with col6:
        st.write("### 🌐 Retrieval AI")
        st.write(highlight_words(retrieval_answer, real_data))
    st.divider()

    # PDF Export
    st.subheader("📄 Export Report")

    pdf_file = generate_pdf(
        question,
        basic_answer,
        retrieval_answer,
        score_basic,
        score_retrieval
    )

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📥 Download PDF Report",
            data=f,
            file_name="ai_reliability_report.pdf",
            mime="application/pdf"
        )
