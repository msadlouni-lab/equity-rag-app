from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Load your data
loader = CSVLoader("eqao_clean.csv")
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)
chunks = splitter.split_documents(docs)

# Embed and store in Chroma
vectorstore = Chroma.from_documents(
    chunks,
    OpenAIEmbeddings(),
    persist_directory="./equity_db"
)

# Test a semantic search
results = vectorstore.similarity_search(
    "schools with low graduation rates", k=3
)
for r in results:
    print(r.page_content)
    print("---")