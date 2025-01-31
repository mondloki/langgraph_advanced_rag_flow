from dotenv import load_dotenv

load_dotenv()

from graph.graph_builder import graph

if __name__=="__main__":
    print("-"*100)
    print(graph.invoke(input={"question": "How to make burger?"}))
    print("-"*100)