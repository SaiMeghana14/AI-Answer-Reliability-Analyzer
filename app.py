import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from utils.generator_basic import generate_basic_answer
from utils.pdf_export import generate_pdf

from utils.retrieval_engine import RetrievalEngine
from utils.source_agreement import SourceAgreement

from utils.reliability_engine import ReliabilityEngine

from utils.claim_engine import ClaimEngine
from utils.highlight_engine import HighlightEngine

from utils.adversarial_tests import detect_false_premise

from utils.benchmark_runner import BenchmarkRunner

from utils.charts import radar

st.set_page_config(
    page_title="TrustEval",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>

/* Sidebar */
section[data-testid="stSidebar"]{
    background: #F8FAFC;
    border-right: 1px solid #E5E7EB;
}

/* Metric Cards */
[data-testid="metric-container"]{
    background: white;
    border: 1px solid #E5E7EB;
    padding: 10px;
    border-radius: 10px;
}

/* Divider */
hr{
    margin-top:0.6rem;
    margin-bottom:0.6rem;
}

/* Reduce top padding */
.block-container{
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("## 🧠 TrustEval")
    st.caption("AI Answer Reliability Framework")

    st.divider()

    st.markdown("### 🚀 Version")
    st.success("v2.0")

    st.divider()

    st.markdown("### 📊 Evaluation Modules")

    st.markdown("🟢 Semantic Similarity")
    st.markdown("🟢 Citation Support")
    st.markdown("🟢 Entity Consistency")
    st.markdown("🟢 Claim Verification")
    st.markdown("🟢 Source Agreement")
    st.markdown("🟢 Contradiction Detection")

    st.divider()

    st.markdown("### 📈 Framework")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Modules", "6")

    with c2:
        st.metric("Status", "Ready")

    st.divider()

    st.markdown("### 🛠 Powered By")

    st.markdown("""
- Streamlit
- Scikit-Learn
- Wikipedia API
- Matplotlib
""")

    st.divider()

    st.caption("🧪 Research Prototype")
    
# ==========================================================
# HOME
# ==========================================================

st.title("🧠 TrustEval")

st.subheader("AI Answer Reliability Evaluation Framework")

st.write(
    "Evaluate AI responses by comparing baseline answers with "
    "retrieval-grounded evidence and reliability metrics."
)

st.divider()

c1, c2 = st.columns(2)

with c1:
    st.success("✔ Semantic Similarity")
    st.success("✔ Citation Support")
    st.success("✔ Entity Consistency")

with c2:
    st.success("✔ Claim Verification")
    st.success("✔ Source Agreement")
    st.success("✔ Contradiction Detection")

st.divider()

m1, m2, m3 = st.columns(3)

m1.metric("Evaluation Metrics", "6")
m2.metric("Framework Version", "2.0")
m3.metric("Status", "Ready")

st.markdown("## 🔍 Ask a Question")

question = st.text_input(
    "Enter your question",
    placeholder="Example: What is Artificial Intelligence?",
    label_visibility="collapsed"
)

analyze = st.button(
    "🔍 Analyze",
    use_container_width=True
)

if question:

    if detect_false_premise(question):

        st.error(
            "⚠️ False Premise Detected\n\n"
            "The question appears to contain an invalid assumption."
        )
        
    if analyze and question:

        engine = RetrievalEngine()
        sources = engine.retrieve(question)
    
        reference_text = SourceAgreement.merged_text(
            sources
        )
    
        confidence = SourceAgreement.confidence(
            sources
        )
        
        basic_answer = generate_basic_answer(
            question
        )
    
        retrieval_answer = f"""
        After comparing multiple reliable sources,
        {reference_text[:1200]}
        """
        
        basic_eval = ReliabilityEngine.evaluate(
        basic_answer,
        reference_text
        )

        retrieval_eval = ReliabilityEngine.evaluate(
            retrieval_answer,
            reference_text
        )
    
        score_basic = basic_eval["final"]
        score_retrieval = retrieval_eval["final"]
    
        if score_retrieval > score_basic:
            winner = "🌐 Retrieval AI"
            reason = (
                "Retrieval AI is better grounded in "
                "external evidence."
            )
        
        else:
    
            winner = "🤖 Baseline AI"
            reason = (
                "Baseline answer aligned more closely "
                "with retrieved evidence."
            )
            
        st.divider()
    
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Overall Reliability",
            f"{retrieval_eval['final']}%"
        )
    
        m2.metric(
            "Claims",
            len(
                ClaimEngine.extract_claims(
                    retrieval_answer
                )
            )
        )
    
        m3.metric(
            "Evidence Sources",
            len(
                [
                    s for s in sources.values()
                    if s["text"]
                ]
            )
        )
    
        m4.metric(
            "Confidence",
            confidence
        )
        
        st.divider()
    
        tab1, tab2 = st.tabs(
            [
                "🤖 Baseline AI",
                "🌐 Retrieval AI"
            ]
        )
    
        with tab1:
    
            st.subheader(
                "Baseline Response"
            )
    
            st.write(
                basic_answer
            )
    
        with tab2:
    
            st.subheader(
                "Retrieval-Based Response"
            )
    
            st.write(
                retrieval_answer
            )
            
        st.divider()
    
        st.subheader(
            "📊 Reliability Breakdown"
        )
    
        comparison = pd.DataFrame({
    
            "Metric":[
    
                "Semantic",
    
                "Citation",
    
                "Entity",
    
                "Contradiction",
    
                "Overall"
    
            ],
    
            "Baseline":[
    
                basic_eval["semantic"],
    
                basic_eval["citation"],
    
                basic_eval["entity"],
    
                100-basic_eval["contradiction"],
    
                basic_eval["final"]
    
            ],
    
            "Retrieval":[
    
                retrieval_eval["semantic"],
    
                retrieval_eval["citation"],
    
                retrieval_eval["entity"],
    
                100-retrieval_eval["contradiction"],
    
                retrieval_eval["final"]
    
            ]
    
        })
    
        st.dataframe(
            comparison,
            use_container_width=True
        )
        st.subheader(
            "📈 Metric Comparison"
        )
    
        chart = pd.DataFrame({
    
            "Metric":[
    
                "Semantic",
    
                "Citation",
    
                "Entity",
    
                "Contradiction"
    
            ],
    
            "Baseline":[
    
                basic_eval["semantic"],
    
                basic_eval["citation"],
    
                basic_eval["entity"],
    
                100-basic_eval["contradiction"]
    
            ],
    
            "Retrieval":[
    
                retrieval_eval["semantic"],
    
                retrieval_eval["citation"],
    
                retrieval_eval["entity"],
    
                100-retrieval_eval["contradiction"]
    
            ]
    
        })
    
        st.bar_chart(
            chart.set_index("Metric")
        )
        st.subheader(
            "🕸 Reliability Radar"
        )
    
        scores = {
    
            "Semantic":
            retrieval_eval["semantic"],
    
            "Citation":
            retrieval_eval["citation"],
    
            "Entity":
            retrieval_eval["entity"],
    
            "Contradiction":
            100-retrieval_eval["contradiction"]
    
        }
    
        st.pyplot(
            radar(scores)
        )
        st.divider()
    
        st.subheader(
            "📚 Evidence Sources"
        )

        for name, info in sources.items():
    
            if info["text"]:
    
                with st.expander(
                    f"📄 {name}"
                ):
    
                    st.write(
                        info["text"]
                    )
    
                    if info["url"]:
    
                        st.link_button(
    
                            "Open Source",
    
                            info["url"]
    
                        )
                        
        st.subheader("🤝 Source Agreement")
    
        if confidence == "High":
    
            st.success(
                "🟢 High agreement between sources."
            )
    
        elif confidence == "Medium":
    
            st.warning(
                "🟡 Moderate agreement."
            )
    
        elif confidence == "Low":
    
            st.warning(
                "🟠 Limited supporting evidence."
            )
    
        else:
    
            st.error(
                "🔴 Insufficient evidence."
            )
            
        st.divider()
    
        st.subheader(
            "⚙ Evaluation Pipeline"
        )
    
        st.info("""
    
    Question
    
    ↓
    
    Baseline AI
    
    ↓
    
    Retrieval Engine
    
    ↓
    
    Evidence Collection
    
    ↓
    
    Semantic Similarity
    
    ↓
    
    Entity Matching
    
    ↓
    
    Claim Verification
    
    ↓
    
    Reliability Score
    
    """)
        st.divider()
    
        st.subheader(
            "🧠 Claim Verification"
        )
    
        claim_results = ClaimEngine.analyze(
            retrieval_answer,
            reference_text
        )
    
        supported = sum(
            c["label"] == "Supported"
            for c in claim_results
        )
    
        partial = sum(
            c["label"] == "Partially Supported"
            for c in claim_results
        )
    
        unsupported = sum(
            c["label"] == "Unsupported"
            for c in claim_results
        )
    
        c1, c2, c3 = st.columns(3)
    
        c1.metric(
            "✅ Supported",
            supported
        )
    
        c2.metric(
            "⚠ Partial",
            partial
        )
    
        c3.metric(
            "❌ Unsupported",
            unsupported
        )
    
        st.divider()
    
        for claim in claim_results:
    
            text = (
                f"**Claim:** {claim['claim']}\n\n"
                f"Confidence: {claim['confidence']} "
                f"({claim['score']}%)"
            )
    
            if claim["label"] == "Supported":
    
                st.success(text)
    
            elif claim["label"] == "Partially Supported":
    
                st.warning(text)
    
            else:
    
                st.error(text)
                
        st.divider()
    
        st.subheader(
            "🔍 Evidence Highlighting"
        )
    
        highlighted = HighlightEngine.highlight(
            retrieval_answer,
            reference_text
        )
    
        st.markdown(
            highlighted,
            unsafe_allow_html=True
        )
        st.divider()
    
        st.subheader(
            "📋 Reliability Analysis"
        )
    
        if retrieval_eval["semantic"] >= 80:
    
            st.success(
                "High semantic similarity with retrieved evidence."
            )
    
        else:
    
            st.warning(
                "Semantic similarity is lower than expected."
            )
    
        if retrieval_eval["citation"] >= 70:
    
            st.success(
                "Good citation support."
            )
    
        else:
    
            st.warning(
                "Citation support is limited."
            )
    
        if retrieval_eval["entity"] >= 70:
    
            st.success(
                "Entity consistency is strong."
            )
    
        else:
    
            st.warning(
                "Entity consistency could be improved."
            )
    
        if retrieval_eval["contradiction"] > 0:
    
            st.error(
                "Potential factual contradiction detected."
            )
    
        else:
    
            st.success(
                "No contradiction detected."
            )
        st.divider()
    
        st.subheader(
            "🏆 Final Comparison"
        )
    
        st.success(winner)
    
        st.write(reason)
    
        improvement = round(
            score_retrieval - score_basic,
            2
        )
    
        if improvement > 0:
    
            st.info(
                f"Retrieval improved reliability by "
                f"{improvement}%."
            )
    
        elif improvement < 0:
    
            st.warning(
                f"Baseline outperformed retrieval by "
                f"{abs(improvement)}%."
            )
    
        else:
    
            st.info(
                "Both approaches achieved similar reliability."
            )
        st.divider()
    
        st.subheader(
            "🎯 Final Assessment"
        )
    
        score = retrieval_eval["final"]
    
        if score >= 85:
    
            st.success("""
    
    ### 🟢 Highly Reliable
    
    This answer is well supported by
    multiple evidence sources with
    minimal contradictions.
    
    """)
    
        elif score >= 70:
    
            st.warning("""
    
    ### 🟡 Moderately Reliable
    
    Most claims appear supported,
    but important statements should
    still be verified.
    
    """)
    
        elif score >= 50:
    
            st.warning("""
    
    ### 🟠 Needs Verification
    
    Some evidence supports the answer,
    but confidence is moderate.
    
    """)
    
        else:
    
            st.error("""
    
    ### 🔴 Low Reliability
    
    Evidence is weak or contradictory.
    Do not rely on this answer without
    independent verification.
    
    """)
        st.divider()
    
        st.subheader(
            "📄 Export Report"
        )
    
        pdf_file = generate_pdf(
    
            question,
    
            basic_answer,
    
            retrieval_answer,
    
            round(score_basic, 2),
    
            round(score_retrieval, 2)
    
        )
    
        with open(pdf_file, "rb") as pdf:
    
            st.download_button(
    
                label="📥 Download PDF Report",
    
                data=pdf,
    
                file_name="TrustEval_Report.pdf",
    
                mime="application/pdf"
    
        )
# ==========================================================
# BENCHMARK SUITE
# ==========================================================

st.divider()

st.header("📊 Benchmark Suite")

st.write(
    """
Evaluate TrustEval on a benchmark dataset and compare
Baseline AI with Retrieval AI.
"""
)

if st.button(
    "🚀 Run Benchmark Suite",
    use_container_width=True
):

    runner = BenchmarkRunner()

    results = runner.run(
        "data/benchmark_questions.csv"
    )

    st.success(
        "Benchmark completed successfully."
    )

    st.dataframe(
        results,
        use_container_width=True
    )
    st.divider()

    avg_baseline = results[
        "Baseline Score"
    ].mean()

    avg_retrieval = results[
        "Retrieval Score"
    ].mean()

    retrieval_wins = (
        results["Winner"] == "Retrieval"
    ).sum()

    total_questions = len(results)

    win_percentage = (
        retrieval_wins /
        total_questions
    ) * 100

    b1, b2, b3 = st.columns(3)

    b1.metric(
        "Average Baseline",
        f"{avg_baseline:.2f}%"
    )

    b2.metric(
        "Average Retrieval",
        f"{avg_retrieval:.2f}%"
    )

    b3.metric(
        "Retrieval Wins",
        f"{win_percentage:.1f}%"
    )
    st.divider()

    st.subheader(
        "📈 Benchmark Comparison"
    )

    chart = pd.DataFrame({

        "Model":[

            "Baseline",

            "Retrieval"

        ],

        "Average Reliability":[

            avg_baseline,

            avg_retrieval

        ]

    })

    st.bar_chart(
        chart.set_index("Model")
    )
    st.subheader(
        "📉 Reliability Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(6,4)
    )

    ax.hist(

        results[
            "Retrieval Score"
        ],

        bins=10

    )

    ax.set_xlabel(
        "Reliability Score"
    )

    ax.set_ylabel(
        "Questions"
    )

    ax.set_title(
        "Distribution"
    )

    st.pyplot(fig)
    benchmark = pd.read_csv(
        "data/benchmark_questions.csv"
    )

    merged = pd.concat(
        [

            benchmark,

            results

        ],

        axis=1
    )

    st.subheader(
        "📚 Category Performance"
    )

    category = (

        merged.groupby(
            "category"
        )["Retrieval Score"]

        .mean()

    )

    st.bar_chart(category)
    st.divider()

    csv = results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        label="⬇ Download Benchmark Results",

        data=csv,

        file_name="benchmark_results.csv",

        mime="text/csv"

    )
st.divider()

st.caption(
"""
🧠 TrustEval v2.0

AI Answer Reliability Evaluation Framework

Modules:
• Baseline AI
• Retrieval AI
• Source Agreement
• Semantic Similarity
• Citation Support
• Entity Consistency
• Contradiction Detection
• Claim Verification
• Benchmark Suite

Built using Python, Streamlit, Scikit-learn,
Matplotlib, Wikipedia API and DuckDuckGo Search.
"""
)
