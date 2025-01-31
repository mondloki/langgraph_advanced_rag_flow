from typing import TypedDict

class GraphState(TypedDict):
    """ Represents the state of the graph
    Attributes:
        question: question
        generation: Response generated by LLM
        web_search: Whether to use web search
        documents: list of documents
    """
    question: str
    generation: str
    web_search: bool
    documents: list[str, None] = []



