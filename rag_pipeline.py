from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Load existing vector store
vectorstore = Chroma(
  persist_directory="./equity_db",
  embedding_function=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Format retrieved docs
def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

# RAG prompt — grounded to context only
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an educational equity analyst for Ontario school boards.
Answer using ONLY the context below. Be specific about school names and scores.
If the answer is not in the context, say "I don't have that data."
Never make up school names, scores, or statistics.

Context: {context}
Question: {question}
""")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Full RAG chain
rag_chain = (
  {"context": retriever | format_docs,
   "question": RunnablePassthrough()}
  | RAG_PROMPT | llm | StrOutputParser()
)

# Ask your first real question
questions = [
  "Which schools have the lowest math scores?",
  "Which schools have low math but high reading scores?",
  "What is the weather in Toronto today?"
]

print(rag_chain.invoke(
    "Which school boards show the largest gap between reading and math performance?"
))

import time
import openai

def safe_rag_call(chain, question, retries=3):
    for attempt in range(retries):
        try:
            return chain.invoke(question)
        except openai.RateLimitError:
            time.sleep(2 ** attempt)
        except openai.APITimeoutError:
            return "Request timed out. Please try again."
    return "Service temporarily unavailable."

for q in questions:
    print(f"Q: {q}")
    print("A: ", end="")
    for chunk in rag_chain.stream(q):
        print(chunk, end="", flush=True)
    print("\n---")