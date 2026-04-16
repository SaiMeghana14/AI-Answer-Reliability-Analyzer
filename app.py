import streamlit as st
from utils.generator import generate_answer
from utils.retriever import get_wiki_data
from utils.evaluator import compute_similarity, get_score
from utils.highlighter import find_missing_info, find_overlap

st.set_page_config(page_title="AI Reliability Analyzer", layout="wide")

st.title("🧠 AI Answer Reliability Analyzer")
st.write("Evaluate AI-generated answers using real-world data (Wikipedia).")

question = st.text_input("🔍 Enter your question:")

if st.button("Analyze") and question:

    # Step 1: Generate AI Answer
    ai_answer = generate_answer(question)

    # Step 2: Retrieve Real Data
    real_data = get_wiki_data(question)

    # Step 3: Evaluate
    similarity = compute_similarity(ai_answer, real_data)
    score, confidence = get_score(similarity)

    # Step 4: Highlights
    missing = find_missing_info(ai_answer, real_data)
    overlap = find_overlap(ai_answer, real_data)

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 AI Answer")
        st.write(ai_answer)

    with col2:
        st.subheader("📚 Retrieved Data (Wikipedia)")
        st.write(real_data)

    st.divider()

    # Score Section
    st.subheader("📊 Evaluation Results")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("Accuracy Score", f"{score}%")

    with col4:
        st.write(f"### {confidence}")

    st.divider()

    # Insights
    st.subheader("🔎 Analysis Insights")

    st.write("**✅ Overlapping Concepts:**")
    st.write(overlap)

    st.write("**⚠️ Missing Concepts:**")
    st.write(missing)

    st.divider()

    # Final Verdict
    if score > 75:
        st.success("AI answer is highly reliable.")
    elif score > 50:
        st.warning("AI answer is partially reliable.")
    else:
        st.error("AI answer may be unreliable or hallucinated.")
