# Ontario Education Equity Analyser

An AI-powered application built on original research investigating the impact 
of socioeconomic factors on Grade 6 reading achievement across Ontario.

## Research
- **Dataset:** 2023–24 EQAO school performance data + 2021 Canada Census
- **Scope:** 6,513 schools across 321 Ontario municipalities
- **Key finding:** Low-income rates and parental education are the strongest 
  predictors of Grade 6 reading achievement (R² = 0.157)

## Application
A hybrid RAG + pandas AI application that lets policy makers query the 
research findings conversationally.

- **RAG pipeline:** LangChain + Chroma vector database + OpenAI
- **Direct lookup:** Pandas router for numeric ranking queries
- **UI:** Streamlit chat interface with streaming responses

## Setup
```bash
pip install -r requirements.txt
python vector_store_research.py  # builds the vector store
streamlit run app_research.py    # launches the app
```

## Research by
Manal Saleh Al Adlouni — M.S. Data Science & AI, University of London