import streamlit as st
import matplotlib.pyplot as plt
import re

from utils.generator_basic import generate_basic_answer
from utils.pdf_export import generate_pdf
from utils.advanced_scoring import reliability_score
from utils.multi_retrieval import retrieve_all, source_agreement
from utils.claim_checker import check_claims
from utils.adversarial_tests import detect_false_premise
from utils.benchmark import benchmark_questions


# ---------------- CONFIG ----------------

st.set_page_config(
    page_title="TrustEval",
    layout="wide"
)

st.title("🧠 TrustEval: Reliability Analysis of AI Answers")


# ---------------- HELPERS ----------------

def combine_sources_text(sources):
    text = " ".join(
        v for v in sources.values()
        if v and len(v.strip()) > 0
    )

    if not text:
        text = "No reliable data found."

    return text


def source_links(query, sources):

    links=[]

    if sources["wiki"]:
        links.append(
            ("Wikipedia",
             f"https://en.wikipedia.org/wiki/{query.replace(' ','_')}")
        )

    if sources["duck"]:
        links.append(
            ("DuckDuckGo",
             f"https://duckduckgo.com/?q={query}")
        )

    return links


def generate_retrieval_answer(question):
    sources = retrieve_all(question)
    text = combine_sources_text(sources)

    return f"Based on reliable sources: {text}"


def highlight_words(answer, reference):

    answer_words=answer.split()
    ref_words=set(reference.lower().split())

    out=""

    for w in answer_words:

        clean=re.sub(r"[^\w]","",w.lower())

        if clean in ref_words:
            out += f"✅ {w} "
        else:
            out += f"❌ {w} "

    return out


def show_gauge(score,title):

    fig,ax=plt.subplots()

    ax.barh([0],[score])

    ax.set_xlim(0,100)

    ax.set_title(title)

    ax.set_yticks([])

    st.pyplot(fig)


# ---------------- INPUT ----------------

question = st.text_input(
    "🔍 Ask a question:"
)


# ---------------- ADVERSARIAL CHECK ----------------

if question and detect_false_premise(question):

    st.error(
      "⚠ False premise detected. This question may be invalid or misleading."
    )


# ---------------- ANALYZE ----------------

if st.button("Analyze") and question:

    # ---------------- ANSWERS ----------------

    basic_answer = generate_basic_answer(question)

    retrieval_answer = generate_retrieval_answer(question)

    # ---------------- RETRIEVAL ----------------

    sources = retrieve_all(question)

    reference_text = combine_sources_text(sources)

    confidence = source_agreement(sources)

    # ---------------- RELIABILITY SCORING ----------------

    score_basic = reliability_score(
        basic_answer,
        reference_text
    )

    score_retrieval = reliability_score(
        retrieval_answer,
        reference_text
    )

    # ---------------- WINNER ----------------

    if score_retrieval > score_basic:
        winner="Retrieval AI"
        reason="Higher factual overlap and stronger source grounding."

    else:
        winner="Baseline AI"
        reason="Unexpectedly stronger alignment on this query."

    # ---------------- DISPLAY ----------------

    col1,col2 = st.columns(2)

    with col1:

        st.subheader(
            "🤖 Baseline AI"
        )

        st.write(
            basic_answer
        )

        st.metric(
            "Reliability Score",
            f"{score_basic:.2f}%"
        )

    with col2:

        st.subheader(
            "🌐 Retrieval AI"
        )

        st.write(
            retrieval_answer
        )

        st.metric(
            "Reliability Score",
            f"{score_retrieval:.2f}%"
        )

    st.divider()

    # ---------------- SOURCE AGREEMENT ----------------

    st.subheader(
        "📚 Source Agreement"
    )

    st.write(
        f"Confidence: **{confidence}**"
    )

    st.write(
        "Sources:"
    )

    for name,link in source_links(
        question,
        sources
    ):
        st.markdown(
          f"- [{name}]({link})"
        )

    st.divider()

    # ---------------- WINNER EXPLANATION ----------------

    st.subheader(
        "🏆 Winner"
    )

    st.success(
        winner
    )

    st.write(
        reason
    )

    st.divider()

    # ---------------- TRUST GAUGES ----------------

    st.subheader(
        "🎯 Trust Score Gauges"
    )

    c3,c4=st.columns(2)

    with c3:
        show_gauge(
            score_basic,
            "Baseline Trust"
        )

    with c4:
        show_gauge(
            score_retrieval,
            "Retrieval Trust"
        )

    st.divider()

    # ---------------- WORD HIGHLIGHT ----------------

    st.subheader(
        "🔍 Word-Level Hallucination Detection"
    )

    c5,c6=st.columns(2)

    with c5:

        st.write(
            highlight_words(
                basic_answer,
                reference_text
            )
        )

    with c6:

        st.write(
            highlight_words(
                retrieval_answer,
                reference_text
            )
        )

    st.divider()

    # ---------------- CLAIM CHECKING ----------------

    st.subheader(
        "🧠 Claim-by-Claim Verification"
    )

    claims = check_claims(
        retrieval_answer,
        reference_text
    )

    for claim,label in claims:

        st.write(
            f"- {claim} → {label}"
        )

    st.divider()

    # ---------------- PDF REPORT ----------------

    st.subheader(
        "📄 Export Report"
    )

    pdf_file = generate_pdf(
        question,
        basic_answer,
        retrieval_answer,
        round(score_basic,2),
        round(score_retrieval,2)
    )

    with open(
        pdf_file,
        "rb"
    ) as f:

        st.download_button(
            label="📥 Download PDF Report",
            data=f,
            file_name="TrustEval_Report.pdf",
            mime="application/pdf"
        )


# ---------------- BENCHMARK MODE ----------------

st.divider()

st.subheader(
    "📊 Benchmark Mode"
)

if st.button(
    "Run Benchmark"
):

    tests=benchmark_questions()

    for q,truth in tests:

        st.write(
            f"Q: {q}"
        )

        st.write(
            f"Expected: {truth}"
        )

        st.write("---")
