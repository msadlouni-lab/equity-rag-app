"""
vector_store_research.py
Builds the Chroma vector store from Manal's actual research dataset.
Run this ONCE after prepare_research_data.py to rebuild the vector store.
The existing equity_db folder will be replaced.
"""

import shutil
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ── Step 1: Clear old vector store ──────────────────────────────────────────
if os.path.exists("./equity_db"):
    shutil.rmtree("./equity_db")
    print("🗑️  Old vector store cleared.")

# ── Step 2: Load your research data ─────────────────────────────────────────
print("📂 Loading research_data_clean.csv...")
loader = CSVLoader(
    file_path="research_data_clean.csv",
    csv_args={"delimiter": ","}
)
docs = loader.load()
print(f"   Loaded {len(docs)} school records.")

# ── Step 3: Split into chunks ────────────────────────────────────────────────
print("✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)
print(f"   Created {len(chunks)} chunks.")

# ── Step 4: Embed and store ──────────────────────────────────────────────────
print("🔢 Embedding and storing in Chroma (this takes ~2-3 minutes)...")
vectorstore = Chroma.from_documents(
    chunks,
    OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./equity_db"
)
print("✅ Vector store built and saved to ./equity_db")

# ── Step 5: Quick test ───────────────────────────────────────────────────────
print("\n🔍 Testing semantic search...")
results = vectorstore.similarity_search(
    "schools with high low-income rates and low reading scores", k=3
)
for r in results:
    print("---")
    print(r.page_content[:200])

print("\n✅ Your research data vector store is ready.")
print("   Next: run app.py to launch the Streamlit UI on your real research data.")
