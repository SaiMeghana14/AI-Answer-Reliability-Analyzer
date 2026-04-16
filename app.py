# app.py

import streamlit as st
import matplotlib.pyplot as plt

from utils.generator_basic import generate_basic_answer
from utils.retriever import get_wiki_data
from utils.evaluator import compute_similarity, get_score

# Try importing OpenAI generator safely
try:
    from utils.generator_openai import generate_openai_answer
    openai_available = True
except:
    openai_available = False


# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AI Reliability Battle", layout="wide")

st.title("⚔️ AI vs AI Reliability Battle Arena")
st.write("Compare AI-generated answers using real-world data (Wikipedia).")


# ------------------ INPUT ------------------
question = st.text_input("🔍 Enter your question:")


# ------------------ CACHE (PREVENT RATE LIMIT) ------------------
@st.cache_data
def get_openai_cached(question):
    return generate_openai_answer(question)


# ------------------ MAIN LOGIC ------------------
if st.button("Analyze") and question:

    with st.spinner("Analyzing responses..."):

        # ✅ Basic AI (always works)
        basic_answer = generate_basic_answer(question)

        # ✅ OpenAI (safe handling)
        if openai_available:
            try:
                openai_answer = get_openai_cached(question)

                # Fallback if rate limit message appears
                if "rate limit" in openai_answer.lower():
                    openai_answer = basic_answer + " (Fallback due to API limit)"

            except Exception:
                openai_answer = basic_answer + " (Fallback: OpenAI failed)"
        else:
            openai_answer = basic_answer + " (OpenAI not available)"

        # ✅ Retrieve ground truth
        real_data = get_wiki_data(question)

        # ✅ Evaluate both
        sim_basic = compute_similarity(basic_answer, real_data)
        sim_openai = compute_similarity(openai_answer, real_data)

        score_basic, conf_basic = get_score(sim_basic)
        score_openai, conf_openai = get_score(sim_openai)

    # ------------------ DISPLAY ------------------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 Basic AI")
        st.write(basic_answer)
        st.metric("Score", f"{score_basic}%")
        st.write(conf_basic)

    with col2:
        st.subheader("🧠 Advanced AI (OpenAI)")
        st.write(openai_answer)
        st.metric("Score", f"{score_openai}%")
        st.write(conf_openai)

    st.divider()

    st.subheader("📚 Ground Truth (Wikipedia)")
    st.write(real_data)

    st.divider()

    # ------------------ WINNER ------------------
    if score_openai > score_basic:
        st.success("🏆 Advanced AI (OpenAI) is more reliable!")
    elif score_basic > score_openai:
        st.success("🏆 Basic AI wins (unexpected!)")
    else:
        st.info("🤝 It's a tie!")

    # ------------------ GRAPH ------------------
    st.subheader("📊 Reliability Comparison")

    models = ["Basic AI", "OpenAI"]
    scores = [score_basic, score_openai]

    fig, ax = plt.subplots()
    ax.bar(models, scores)
    ax.set_ylabel("Accuracy Score")
    ax.set_title("AI Reliability Comparison")

    st.pyplot(fig)

    st.divider()

    # ------------------ FINAL VERDICT ------------------
    st.subheader("🧠 Final Verdict")

    if score_openai > 75 or score_basic > 75:
        st.success("High reliability detected in at least one model.")
    elif score_openai > 50 or score_basic > 50:
        st.warning("Moderate reliability. Some gaps exist.")
    else:
        st.error("Low reliability. Possible hallucinations detected.")
