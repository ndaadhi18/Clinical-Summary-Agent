from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.config import GOOGLE_API_KEY, LLM_MODEL_NAME, log_llm_interaction
from src.schemas import SOAPNote, NurseReport, PatientDemographics, ClinicalRisk
import json

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL_NAME,
    temperature=0.2, 
    google_api_key=GOOGLE_API_KEY
)

# --- Agent 1: The Nurse
def nurse_agent_node(state):
    """
    Role: Extract Demographics AND Clinical Risks in a SINGLE pass.
    Optimization: Reduces 2 sequential LLM calls to 1.
    """
    dialogue = state["scrubbed_dialogue"]
    
    # Use the combined parser
    parser = PydanticOutputParser(pydantic_object=NurseReport)
    
    prompt = ChatPromptTemplate.from_template(
        "You are a Triage Nurse. Analyze the dialogue and extract both patient demographics "
        "and any urgent clinical risks.\n"
        "{format_instructions}\n\n"
        "Dialogue:\n{dialogue}"
    )
    
    chain = prompt | llm | parser
    
    nurse_output = chain.invoke({
        "dialogue": dialogue, 
        "format_instructions": parser.get_format_instructions()
    })
    
    # Log interaction
    log_llm_interaction("Nurse_Agent_Combined", dialogue, nurse_output.model_dump())

    # Update State
    return {
        "demographics": nurse_output.demographics, 
        "risk_analysis": nurse_output.risk_analysis
    }

# --- Agent 2: The Doctor (SOAP Writer) ---
def doctor_agent_node(state):
    """
    Role: Synthesize the dialogue into a formal SOAP Note.
    """
    dialogue = state["scrubbed_dialogue"]
    risks = state["risk_analysis"]
    
    parser = PydanticOutputParser(pydantic_object=SOAPNote)
    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert physician. Draft a structured SOAP note based on the dialogue.\n"
        "Take into account these identified risks: {risks}\n"
        "{format_instructions}\n\n"
        "Dialogue:\n{dialogue}"
    )
    
    chain = prompt | llm | parser
    soap_note = chain.invoke({
        "dialogue": dialogue,
        "risks": risks.model_dump(),
        "format_instructions": parser.get_format_instructions()
    })

    # Log interaction
    log_llm_interaction("Doctor_Agent_SOAP", dialogue, soap_note.model_dump())

    return {"soap_note": soap_note}

# --- Agent 3: The Auditor (Critique & Refine) ---
def auditor_agent_node(state):
    """
    Role: Review the SOAP note against the original dialogue.
    """
    dialogue = state["scrubbed_dialogue"]
    current_note = state["soap_note"]
    
    prompt = ChatPromptTemplate.from_template(
        "You are a Clinical Auditor. Review the following SOAP Note against the transcript.\n"
        "Identify if any critical info was missed or if there are hallucinations.\n"
        "If the note is good, output 'APPROVED'. If issues found, list them briefly.\n\n"
        "Original Dialogue:\n{dialogue}\n\n"
        "Generated SOAP Note:\n{note}"
    )
    
    chain = prompt | llm
    critique = chain.invoke({
        "dialogue": dialogue,
        "note": current_note.model_dump()
    })
    
    # Log interaction
    log_llm_interaction("Auditor_Agent", str(current_note.model_dump()), critique.content)

    return {"critique_comments": critique.content}