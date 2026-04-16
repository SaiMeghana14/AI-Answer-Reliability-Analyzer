import streamlit as st
import matplotlib.pyplot as plt

from utils.generator_basic import generate_basic_answer
from utils.generator_openai import generate_openai_answer
from utils.retriever import get_wiki_data
from utils.evaluator import compute_similarity, get_score

st.set_page_config(page_title="AI Battle Arena", layout="wide")

st.title("⚔️ AI vs AI Reliability Battle Arena")
st.write("Compare AI models based on factual accuracy using real-world data.")

question = st.text_input("🔍 Enter your question:")

if st.button("Analyze") and question:

    # Generate answers
    basic_answer = generate_basic_answer(question)
    openai_answer = generate_openai_answer(question)

    # Retrieve truth data
    real_data = get_wiki_data(question)

    # Evaluate both
    sim_basic = compute_similarity(basic_answer, real_data)
    sim_openai = compute_similarity(openai_answer, real_data)

    score_basic, conf_basic = get_score(sim_basic)
    score_openai, conf_openai = get_score(sim_openai)

    # Layout
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

    # 🏆 Winner Logic
    if score_openai > score_basic:
        st.success("🏆 OpenAI model is more reliable!")
    elif score_basic > score_openai:
        st.success("🏆 Basic model wins (unexpected!)")
    else:
        st.info("🤝 It's a tie!")

    # 📊 GRAPH (IMPORTANT)
    st.subheader("📊 Reliability Comparison")

    models = ["Basic AI", "OpenAI"]
    scores = [score_basic, score_openai]

    fig, ax = plt.subplots()
    ax.bar(models, scores)
    ax.set_ylabel("Accuracy Score")
    ax.set_title("AI Reliability Comparison")

    st.pyplot(fig)
