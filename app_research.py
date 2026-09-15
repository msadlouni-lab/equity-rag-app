"""
app_research.py — Ontario Education Equity Analyser
Built on Manal Saleh Al Adlouni's research:
"The Impact of Socioeconomic Factors on Grade 6 Reading Achievement in Ontario"
Using 2023–2024 EQAO data merged with 2021 Canada Census data (6,513 schools, 321 municipalities)
"""

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import plotly.express as px
from data_lookup import get_lowest_reading, get_equity_risk_schools, get_highest_lowincome

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ontario Equity Analyser",
    page_icon="🎓",
    layout="wide"
)

# ── Route question function ───────────────────────────────────────────────────
def route_question(question):
    q = question.lower()
    if any(w in q for w in ["lowest", "worst", "bottom", "least"]) and \
       any(w in q for w in ["score", "reading", "achievement"]):
        return "📊 **Lowest reading scores in your research dataset:**\n\n```\n" + get_lowest_reading() + "\n```"
    if any(w in q for w in ["highest", "most", "top"]) and \
       any(w in q for w in ["low income", "lowincome", "poverty"]):
        return "📊 **Highest low-income schools:**\n\n```\n" + get_highest_lowincome() + "\n```"
    if any(w in q for w in ["equity risk", "at risk"]):
        return "📊 **Equity risk schools (reading < 60%):**\n\n```\n" + get_equity_risk_schools() + "\n```"
    return None

# ── Format docs helper ────────────────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# ── Load RAG chain once ───────────────────────────────────────────────────────
@st.cache_resource
def load_chain():
    # Auto-build vector store if it doesn't exist
    if not os.path.exists("./equity_db"):
        st.info("🔨 Building vector store for first time — takes ~2 minutes...")
        from langchain_community.document_loaders import CSVLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        loader = CSVLoader("research_data_clean.csv")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        Chroma.from_documents(
            chunks,
            OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory="./equity_db"
        )
        st.success("✅ Vector store built successfully!")

    vectorstore = Chroma(
        persist_directory="./equity_db",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}
    )

    RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an educational equity analyst assistant for Ontario school boards.
You are powered by original research on Grade 6 reading achievement across Ontario,
combining 2023-2024 EQAO school performance data with 2021 Canada Census data
covering 6,513 schools and 321 municipalities.

Key research findings you can reference:
- Low-income rates and parental education (NoDegree) are the strongest predictors of achievement
- Each 10 percentage point increase in low-income households = ~4.2 point drop in reading scores
- Each 10 percentage point increase in parents without a degree = ~4.5 point drop in reading scores
- Population density has minimal impact (r=0.05, explains only 0.2% of variance)
- Provincial median Grade 6 reading achievement = 85%
- Notable outlier municipalities: Rainy River (41.5%), Magnetawan (42%), Cobalt (47%)
- The negative effect of low income is slightly stronger in densely populated areas

Answer using ONLY the context provided below. Be specific about school names,
board names, municipalities, and scores. If comparing schools, mention actual numbers.
If the answer is not in the context, say "I don't have that data in my research dataset."
Never make up school names, scores, or statistics.

Context:
{context}

Question: {question}

Provide a clear, specific answer. Where relevant, connect findings to the broader
research context about socioeconomic factors and educational equity in Ontario.
""")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

# ── Load visualisation data once ──────────────────────────────────────────────
@st.cache_data
def load_viz_data():
    import pandas as pd
    import numpy as np
    df = pd.read_csv("research_data_clean.csv")

    def clean_pct(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().replace('%', '')
        if s in ('N/R', 'N/A', 'N/D', 'SP', '', 'nan'):
            return np.nan
        try:
            return float(s)
        except:
            return np.nan

    for col in ['Grade6Reading_pct', 'Grade6Math_pct']:
        if col in df.columns:
            df[col] = df[col].apply(clean_pct)

    return df

# ── Initialise ────────────────────────────────────────────────────────────────
rag_chain = load_chain()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 Ontario Equity Analyser")
    st.markdown("""
    **Research-powered AI assistant** built on original analysis of Grade 6 reading
    achievement across Ontario.

    **Dataset:** 6,513 schools · 321 municipalities
    **Sources:** EQAO 2023–24 + Canada Census 2021
    """)
    st.divider()
    st.markdown("**📊 Key Research Findings**")
    st.markdown("""
    - Provincial median reading: **85%**
    - Low income explains **16%** of achievement variance
    - Strongest predictor: **low-income rate** (r = -0.33)
    - Population density: **minimal impact** (r = 0.05)
    """)
    st.divider()
    st.markdown("**💬 Sample Questions**")
    sample_questions = [
        "Which schools have the lowest reading scores?",
        "Which municipalities have high low-income rates and low achievement?",
        "Show me schools in Toronto with equity risk",
        "Which boards show the largest reading-math gap?",
        "What are the outlier municipalities and why do they matter?",
        "Which schools improved the most over three years?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.prefill_question = q

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Ontario Education Equity Analyser")
st.caption("Research-powered AI assistant · 6,513 schools · 321 municipalities · EQAO 2023–24 + Census 2021")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Ask the Data", "📊 Visualisations"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    # Research summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Schools Analysed", "6,513")
    col2.metric("Municipalities", "321")
    col3.metric("Provincial Median Reading", "85%")
    col4.metric("Equity Risk Schools", "291")
    st.divider()

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Handle sidebar prefill
    if "prefill_question" in st.session_state:
        prefill = st.session_state.pop("prefill_question")
        st.session_state.messages.append({"role": "user", "content": prefill})
        with st.chat_message("user"):
            st.write(prefill)
        with st.chat_message("assistant"):
            with st.spinner("Analysing..."):
                direct = route_question(prefill)
                if direct:
                    st.markdown(direct)
                    response = direct
                else:
                    response = st.write_stream(rag_chain.stream(prefill))
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if question := st.chat_input("Ask about Ontario school equity..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Analysing research data..."):
                direct_answer = route_question(question)
                if direct_answer:
                    st.markdown(direct_answer)
                    response = direct_answer
                else:
                    response = st.write_stream(rag_chain.stream(question))
        st.session_state.messages.append({"role": "assistant", "content": response})

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    import pandas as pd
    df_viz = load_viz_data()

    st.subheader("Key Research Findings")
    st.caption("Visualisations from: The Impact of Socioeconomic Factors on Grade 6 Reading Achievement in Ontario")

    # ── Chart 1: Low Income vs Reading ──────────────────────────────────────
    st.markdown("#### Low-Income Rate vs Grade 6 Reading Achievement")
    df_scatter = df_viz[[
        'School Name', 'Board Name', 'Municipality',
        'Grade6Reading', 'LowIncome_pct', 'NoDegree_pct'
    ]].dropna()
    fig1 = px.scatter(
        df_scatter,
        x='LowIncome_pct',
        y='Grade6Reading',
        hover_data=['School Name', 'Board Name', 'Municipality'],
        color='NoDegree_pct',
        color_continuous_scale='RdYlGn_r',
        labels={
            'LowIncome_pct': 'Low-Income Rate (%)',
            'Grade6Reading': 'Grade 6 Reading Achievement (%)',
            'NoDegree_pct': 'Parents Without Degree (%)'
        },
        title="Low-Income Rate vs Grade 6 Reading (colour = parental education)",
        opacity=0.6
    )
    fig1.update_layout(height=450)
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Each point is one school. Hover to see school name and board. r = -0.33 (OLS regression finding).")

    # ── Chart 2: Feature Importance ─────────────────────────────────────────
    st.markdown("#### Equity Risk Classifier — Feature Importance")
    feat_data = {
        'Feature': ['Grade 3 Math', 'Grade 3 Reading', 'Population Density',
                    'Low-Income Rate', 'Parents No Degree', 'Special Ed Rate'],
        'Importance': [0.282, 0.242, 0.151, 0.126, 0.101, 0.099]
    }
    feat_df = pd.DataFrame(feat_data).sort_values('Importance')
    fig2 = px.bar(
        feat_df, x='Importance', y='Feature', orientation='h',
        color='Importance', color_continuous_scale='Teal',
        title="Random Forest Feature Importance — Equity Risk Classifier (ROC-AUC: 0.959)"
    )
    fig2.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Grade 3 scores rank highest — suggesting early intervention is more predictive than socioeconomic targeting alone.")

    # ── Chart 3: Board-level averages ────────────────────────────────────────
    st.markdown("#### Grade 6 Reading Achievement by Board — Bottom 15")
    board_avg = (
        df_viz.groupby('Board Name')['Grade6Reading']
        .mean()
        .reset_index()
        .sort_values('Grade6Reading')
        .head(15)
    )
    fig3 = px.bar(
        board_avg, x='Grade6Reading', y='Board Name', orientation='h',
        color='Grade6Reading', color_continuous_scale='RdYlGn',
        labels={'Grade6Reading': 'Average Reading Achievement (%)'},
        title="15 Boards with Lowest Average Grade 6 Reading Achievement"
    )
    fig3.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Table: Equity Risk Schools ───────────────────────────────────────────
    st.markdown("#### Equity Risk Schools — Reading Achievement Below 60%")
    risk_df = (
        df_viz[df_viz['EquityRisk'] == True][[
            'School Name', 'Board Name', 'Municipality',
            'Grade6Reading', 'LowIncome_pct', 'NoDegree_pct'
        ]]
        .drop_duplicates(subset=['School Name', 'Board Name'])
        .sort_values('Grade6Reading')
        .rename(columns={
            'Grade6Reading': 'Reading %',
            'LowIncome_pct': 'Low Income %',
            'NoDegree_pct': 'No Degree %'
        })
    )
    st.dataframe(risk_df, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(risk_df)} schools with Grade 6 reading below 60%. Sortable by clicking column headers.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Research by Manal Saleh Al Adlouni · "
    "The Impact of Socioeconomic Factors on Grade 6 Reading Achievement in Ontario · "
    "June 2025"
)