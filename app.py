import streamlit as st
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI

class SupportState:
    def __init__(self, query, classification=None, response=None):
        self.query = query
        self.classification = classification
        self.response = response

llm = ChatOpenAI(model="gpt-4")

def classify(state: SupportState):
    state.classification = llm.predict(f"Classify support query: {state.query}")
    return state

def retrieve(state: SupportState):
    if "delivery" in state.query.lower():
        state.response = "Your order may be delayed. Please check your tracking link."
    return state

def decide(state: SupportState):
    if state.response:
        return state
    state.response = "Escalating to human support."
    return state

graph = StateGraph(SupportState)
graph.add_node("classify", classify)
graph.add_node("retrieve", retrieve)
graph.add_node("decide", decide)
graph.set_entry_point("classify")
graph.add_edge("classify", "retrieve")
graph.add_edge("retrieve", "decide")

workflow = graph.compile()

st.title("HelpDesk AI")
query = st.text_input("Enter your support query:")
if query:
    state = SupportState(query)
    result = workflow.invoke(state)
    st.write("Response:", result.response)
