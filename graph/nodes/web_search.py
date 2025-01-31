from typing import Any, Dict

from graph.state import GraphState
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document

web_search_tool = TavilySearchResults(max_results=5)

def web_search_node(state:GraphState):
    question = state["question"]
    raw_tavily_outputs = web_search_tool.invoke(question)
    web_results = "\n".join(output["content"] for output in raw_tavily_outputs)
    if "documents" in state.keys():
        documents = state["documents"]
    else:
        documents = []
    web_search_doc = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_search_doc)
    else:
        documents = [web_search_doc]
    return {"documents": documents,
            "question": question}