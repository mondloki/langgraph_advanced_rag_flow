from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
rag_prompt = hub.pull("rlm/rag-prompt")

generation_chain = rag_prompt | llm | StrOutputParser()
