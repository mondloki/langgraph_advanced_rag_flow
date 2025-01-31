from graph.state import GraphState
from typing import Dict, Any
from ingestion import retriever


def retriever_node(state: GraphState) -> Dict[str, Any]:
    question =  state["question"]
    documents =  retriever.invoke(question)

    return {"documents": documents,
            "question": question } # question inclusion is not mandatory as it is already captured in state


