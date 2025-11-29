import streamlit as st
from src.graph import run_pipeline
from src.data_loader import load_meddialog_dataset
from src.evaluation import calculate_metrics

st.set_page_config(page_title="Origin Medical Task", layout="wide")

# --- Title ---
st.title("Agentic AI Clinical Summarizer")
st.markdown("Extract SOAP Notes from complex medical dialogues.")

# --- Data Loading (Cached) ---
@st.cache_data
def get_data():
    return load_meddialog_dataset(n=50)

try:
    dataset = get_data()
    # Create a simplified dictionary: "ID (Length) -> Dialogue"
    options = {f"{d['id']} ({d['length']} words)": d['dialogue'] for d in dataset}
except:
    st.error("Failed to fetch dataset.")
    options = {}

# --- Input Section ---
st.divider()
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Dialogue")
    
    # Dropdown to pick a hard case
    selected_key = st.selectbox("Select a Dataset Case:", options=list(options.keys()))
    
    # Allow overwriting the text
    if selected_key:
        default_text = options[selected_key]
    else:
        default_text = ""

    dialogue_input = st.text_area("Transcript", value=default_text, height=550)
    
    start_btn = st.button("Generate SOAP Note", type="primary")

# --- Output Section ---
with col2:
    st.subheader("Structured Output")
    
    if start_btn and dialogue_input:
        with st.spinner("Agents working (Nurse -> Doctor -> Auditor)..."):
            
            # 1. Run the Agent Pipeline
            case_id = selected_key.split()[0] if selected_key else "custom"
            result = run_pipeline(dialogue_input, dialogue_id=case_id)
            
            if result.get("error"):
                st.error(result['error'])
            else:
                # 2. Display Raw JSON
                soap_data = result['soap_note'].model_dump()
                st.json(soap_data)
                
                st.divider()
                
                # 3. Display Evaluation Metrics (ROUGE)
                st.subheader("📈 Evaluation Metrics")
                
                # Convert JSON to text for comparison
                generated_text = (
                    f"Subjective: {soap_data['subjective']} "
                    f"Objective: {soap_data['objective']} "
                    f"Assessment: {soap_data['assessment']} "
                    f"Plan: {' '.join(soap_data['plan'])}"
                )
                
                # Calculate scores (Generated vs Original Transcript)
                scores = calculate_metrics(generated_text, dialogue_input)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("ROUGE-1 (Recall)", f"{scores['rouge1'].fmeasure:.4f}", help="Unigram overlap")
                m2.metric("ROUGE-L (Structure)", f"{scores['rougeL'].fmeasure:.4f}", help="Sentence structure similarity")
                m3.metric("Auditor Status", "APPROVED" if "APPROVED" in result['critique_comments'] else "FLAGGED")
                
                st.caption(f"Auditor Feedback: {result['critique_comments']}")