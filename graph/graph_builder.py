from graph.consts import RETRIEVE, GRADE_DOCUMENTS, WEB_SEARCH, GENERATE
from graph.state import GraphState
from langgraph.graph import StateGraph, END
from graph.nodes.retriever import retriever_node
from graph.nodes.grade_documents import grade_documents_node
from graph.nodes.web_search import web_search_node
from graph.nodes.generate import generate_node
from graph.chains.answer_grader_selfrag import answer_grader
from graph.chains.hallucination_grader_selfrag import hallucination_grader
from graph.chains.router_adaptiverag import RouteQuery, question_router


def decide_to_generate(state:GraphState):

    if state["web_search"]:

        return WEB_SEARCH
    
    else:
        return GENERATE
    
def grade_generation_grounded_in_documents_and_question(state:GraphState):
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    hallucination_score = hallucination_grader.invoke({"documents": documents, "generation": generation}) 
    if hallucination_score.binary_score: # if answer grounded from documents/facts
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        
        answer_score = answer_grader.invoke({"question": question,
                                             "generation": generation})
        if answer_score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]

    source: RouteQuery = question_router.invoke({"question": question})

    if source.datasource == WEB_SEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEB_SEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE
    


builder = StateGraph(GraphState)

    
builder.add_node(RETRIEVE, retriever_node)
builder.add_node(GRADE_DOCUMENTS, grade_documents_node)
builder.add_node(WEB_SEARCH, web_search_node)
builder.add_node(GENERATE, generate_node)


builder.add_edge(RETRIEVE, GRADE_DOCUMENTS)
builder.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate,
                              path_map={
                                  WEB_SEARCH: WEB_SEARCH,
                                  GENERATE: GENERATE
                              }) # path_map is not required for now since decide_to_generate returns the right node name only
builder.add_edge(WEB_SEARCH, GENERATE)
builder.add_conditional_edges(GENERATE, grade_generation_grounded_in_documents_and_question, # SELF RAG component
                              path_map={
                                  "useful": END,
                                  "not useful": WEB_SEARCH,
                                  "not supported": GENERATE
                              })
builder.add_edge(GENERATE, END)
# builder.set_entry_point(RETRIEVE) # for corrective RAG and self RAG
builder.set_conditional_entry_point(route_question)

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")








