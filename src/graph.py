import logging
from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, END
from src.agents import nurse_agent_node, doctor_agent_node, auditor_agent_node
from src.schemas import PatientDemographics, SOAPNote, ClinicalRisk, NurseReport

# --- 1. State Definition ---
class GraphState(TypedDict):
    """
    Represents the state of our graph. 
    Agents read from and write to this shared dictionary.
    """
    dialogue_id: str
    raw_dialogue: str
    scrubbed_dialogue: str
    
    # Clinical Data (Populated by Agents)
    demographics: Optional[PatientDemographics]
    risk_analysis: Optional[ClinicalRisk]
    soap_note: Optional[SOAPNote]
    critique_comments: Optional[str]
    
    # Error Tracking
    error: Optional[str]

# --- 2. Robustness Layer (Error Handling) ---
error_logger = logging.getLogger("GraphError")

def safe_node(node_func):
    """
    Decorator-like wrapper to catch exceptions inside any agent.
    Prevents the entire pipeline from crashing if one LLM call fails.
    """
    def wrapper(state: GraphState):
        try:
            return node_func(state)
        except Exception as e:
            error_msg = f"Crash in {node_func.__name__}: {str(e)}"
            error_logger.error(error_msg)
            return {"error": error_msg}
    return wrapper

# --- 3. Conditional Router ---
def route_next_step(state: GraphState) -> Literal["doctor", "auditor", END]: # type: ignore
    """
    Determines the next step based on the current state.
    If an error exists, stop immediately. Otherwise, proceed.
    """
    if state.get("error"):
        return END
        
    # Logic: This function is called AFTER a node finishes.
    # We define the flow in the add_conditional_edges below.
    return "next" 

# --- 4. Graph Construction ---
workflow = StateGraph(GraphState)

# Add Nodes (Wrapped in safety logic)
workflow.add_node("nurse", safe_node(nurse_agent_node))
workflow.add_node("doctor", safe_node(doctor_agent_node))
workflow.add_node("auditor", safe_node(auditor_agent_node))

# Define Entry Point
workflow.set_entry_point("nurse")

# Define Flow with Error Checking
# Pattern: Node -> Check Error -> Next Node
workflow.add_conditional_edges(
    "nurse",
    lambda state: END if state.get("error") else "doctor"
)

workflow.add_conditional_edges(
    "doctor",
    lambda state: END if state.get("error") else "auditor"
)

workflow.add_edge("auditor", END)

# Compile
app = workflow.compile()

# --- 5. Execution Utility ---
def run_pipeline(dialogue_text: str, dialogue_id: str = "001"):
    """
    Main entry point called by Streamlit or Main script.
    """
    inputs = {
        "dialogue_id": dialogue_id,
        "raw_dialogue": dialogue_text,
        "scrubbed_dialogue": dialogue_text, # PII scrubbing happens in loader/UI
        "demographics": None,
        "soap_note": None,
        "risk_analysis": None,
        "critique_comments": None,
        "error": None
    }
    
    return app.invoke(inputs)