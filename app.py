import streamlit as st
import matplotlib.pyplot as plt
import nltk
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

from utils.generator_basic import generate_basic_answer
from utils.retriever import get_wiki_data
from utils.highlighter import highlight_text
from utils.pdf_export import generate_pdf
from utils.evaluator import compute_similarity, get_score
from utils.advanced_eval import detect_hallucinations, explain_winner

# Download tokenizer
nltk.download('punkt')

# Safe OpenAI import
try:
    from utils.generator_openai import generate_openai_answer
    openai_available = True
except:
    openai_available = False


# ------------------ CONFIG ------------------
st.set_page_config(page_title="AI Reliability Battle", layout="wide")

st.title("⚔️ AI vs AI Reliability Battle Arena")


# ------------------ GAUGE ------------------
def show_gauge(score, title):
    fig, ax = plt.subplots()
    ax.barh([0], [score])
    ax.set_xlim(0, 100)
    ax.set_title(title)
    ax.set_yticks([])
    st.pyplot(fig)


# ------------------ HIGHLIGHT FUNCTION ------------------
def highlight_text(text, hallucinations):
    highlighted = ""
    for item in hallucinations:
        sentence = item["sentence"]
        if item["hallucinated"]:
            highlighted += f"❌ {sentence}\n\n"
        else:
            highlighted += f"✅ {sentence}\n\n"
    return highlighted


# ------------------ PDF EXPORT ------------------
def generate_pdf(question, basic_ans, openai_ans, score_basic, score_openai):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_file.name)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph(f"<b>Question:</b> {question}", styles["Normal"]))
    content.append(Paragraph(f"<b>Basic AI Answer:</b> {basic_ans}", styles["Normal"]))
    content.append(Paragraph(f"<b>OpenAI Answer:</b> {openai_ans}", styles["Normal"]))
    content.append(Paragraph(f"<b>Basic Score:</b> {score_basic}", styles["Normal"]))
    content.append(Paragraph(f"<b>OpenAI Score:</b> {score_openai}", styles["Normal"]))

    doc.build(content)
    return temp_file.name


# ------------------ INPUT ------------------
question = st.text_input("🔍 Enter your question:")


# ------------------ CACHE ------------------
@st.cache_data
def get_openai_cached(q):
    return generate_openai_answer(q)


# ------------------ MAIN ------------------
if st.button("Analyze") and question:

    with st.spinner("Analyzing..."):

        basic_answer = generate_basic_answer(question)

        if openai_available:
            try:
                openai_answer = get_openai_cached(question)
                if "rate limit" in openai_answer.lower():
                    openai_answer = basic_answer + " (Fallback)"
            except:
                openai_answer = basic_answer + " (Fallback)"
        else:
            openai_answer = basic_answer + " (OpenAI not available)"

        real_data = get_wiki_data(question)

        # Scores
        sim_basic = compute_similarity(basic_answer, real_data)
        sim_openai = compute_similarity(openai_answer, real_data)

        score_basic, conf_basic = get_score(sim_basic)
        score_openai, conf_openai = get_score(sim_openai)

        # Hallucination detection
        hall_basic = detect_hallucinations(basic_answer, real_data)
        hall_openai = detect_hallucinations(openai_answer, real_data)

    # ------------------ DISPLAY ------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 Basic AI")
        st.write(basic_answer)
        st.metric("Score", f"{score_basic}%")
        st.write(conf_basic)

    with col2:
        st.subheader("🧠 OpenAI")
        st.write(openai_answer)
        st.metric("Score", f"{score_openai}%")
        st.write(conf_openai)

    st.divider()

    st.subheader("📚 Ground Truth")
    st.write(real_data)

    st.divider()

    # Winner
    winner, reason = explain_winner(score_basic, score_openai)

    st.subheader("🏆 Winner")
    st.success(f"{winner}")

    st.subheader("🧠 Why This Model Won")
    st.write(reason)

    st.divider()

    # Gauge
    st.subheader("🎯 Trust Score Gauges")

    col3, col4 = st.columns(2)
    with col3:
        show_gauge(score_basic, "Basic AI Trust")

    with col4:
        show_gauge(score_openai, "OpenAI Trust")

    st.divider()

    # Sentence analysis
    st.subheader("🔍 Sentence-Level Hallucination Detection")

    col5, col6 = st.columns(2)

    with col5:
        st.write("### 🤖 Basic AI")
        st.text(highlight_text(basic_answer, hall_basic))

    with col6:
        st.write("### 🧠 OpenAI")
        st.text(highlight_text(openai_answer, hall_openai))

    st.divider()

    # PDF Export
    st.subheader("📄 Export Report")

    pdf_file = generate_pdf(
        question,
        basic_answer,
        openai_answer,
        score_basic,
        score_openai
    )

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📥 Download PDF Report",
            data=f,
            file_name="ai_reliability_report.pdf",
            mime="application/pdf"
        )
