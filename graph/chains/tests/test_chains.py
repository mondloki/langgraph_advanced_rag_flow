from ingestion import retriever
from graph.chains.retrieval_grader import retrieval_grader, GradeDocuments
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader_selfrag import hallucination_grader, GradeHallucinations
from graph.chains.router_adaptiverag import question_router, RouteQuery
from ingestion import retriever

def test_retrieval_grader_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    doc_txt = docs[1].page_content

    res: GradeDocuments =  retrieval_grader.invoke(
        {
            "question": question,
            "document": doc_txt
        }
    )

    assert res.binary_score == "yes"

def test_retrieval_grader_no() -> None:
    question = "chicken burger"
    docs = retriever.invoke(question)

    doc_txt = docs[1].page_content

    res: GradeDocuments =  retrieval_grader.invoke(
        {
            "question": question,
            "document": doc_txt
        }
    )

    assert res.binary_score == "no"

def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"question": question, "context": docs})
    print("$"*100)
    print(generation)
    print("$"*100)

def test_hallucination_grader_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"question": question, "context": docs})

    res:GradeHallucinations = hallucination_grader.invoke({
        "documents": docs, "generation" : generation
    })

    assert res.binary_score

def test_hallucination_grader_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"question": question, "context": docs})

    res:GradeHallucinations = hallucination_grader.invoke({
        "documents": docs, 
        "generation" : "In order to make pizza we need to first start with the dough"
    })

    assert not res.binary_score

def test_router_to_vectorstore() -> None:
    question = "agent memory"

    res: RouteQuery = question_router.invoke({"question":question})

    assert res.datasource == "vectorstore"

def test_router_to_websearch() -> None:
    question = "how to make pizza"

    res: RouteQuery = question_router.invoke({"question":question})

    assert res.datasource == "websearch"



