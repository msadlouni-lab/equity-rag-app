from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
prompt = ChatPromptTemplate.from_template(
    "You are an education equity analyst. Answer briefly: {question}"
)
chain = prompt | llm | StrOutputParser()

print(chain.invoke({"question": "What is the equity gap in Canadian schools?"}))