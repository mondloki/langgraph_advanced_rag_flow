from graph.state import GraphState
from graph.chains.generation import generation_chain


def generate_node(state:GraphState):

    question = state["question"]
    documents = state["documents"]
    generation = generation_chain.invoke({"question": question,
                                   "context": documents})
    
    return {"documents": documents, "question": question, "generation": generation}

