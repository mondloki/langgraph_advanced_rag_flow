from typing import Any, Dict

from graph.state import GraphState
from graph.chains.retrieval_grader import retrieval_grader


def grade_documents_node(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECKING DOCUMENT RELEVANCE TO QUESTION---")
    question =  state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False

    for doc in documents:
        score = retrieval_grader.invoke({"question": question,
                                                "document": doc.page_content})
        binary_score = score.binary_score
        if binary_score.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue
    
    return {"documents": filtered_docs, "question": question, "web_search": web_search}
