# app.py
import streamlit as st
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI

# Define the state class
class SupportState:
    def __init__(self, query, classification=None, response=None):
        self.query = query
        self.classification = classification
        self.response = response

# Initialize LLM
llm = ChatOpenAI(model="gpt-4")

# Node functions
def classify(state: SupportState):
    state.classification = llm.predict(f"Classify support query: {state.query}")
    return state   # ✅ must return state

def retrieve(state: SupportState):
    if "delivery" in state.query.lower():
        state.response = "Your order may be delayed. Please check your tracking link."
    elif "password" in state.query.lower():
        state.response = "You can reset your password by clicking 'Forgot Password' on the login page."
    return state   # ✅ return state

def decide(state: SupportState):
    if state.response:
        return state
    state.response = "Escalating to human support."
    return state   # ✅ return state

# Build workflow graph
graph = StateGraph(SupportState)
graph.add_node("classify", classify)
graph.add_node("retrieve", retrieve)
graph.add_node("decide", decide)

graph.set_entry_point("classify")
graph.add_edge("classify", "retrieve")
graph.add_edge("retrieve", "decide")

workflow = graph.compile()

# --- Streamlit UI ---
st.title("HelpDesk AI")
query = st.text_input("Enter your support query")

if query:
    state = SupportState(query)
    result = workflow.invoke(state)
    st.write(result.response)
