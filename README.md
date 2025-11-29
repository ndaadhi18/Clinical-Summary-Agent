# Agentic AI Clinical Summarizer 🩺

**Role Challenge Submission - Origin Medical**  
**Role:** Agentic AI Research Intern  

## 🚀 Project Overview
This project implements an **Agentic AI Pipeline** designed to parse complex doctor-patient dialogues into structured **SOAP Notes** (Subjective, Objective, Assessment, Plan). 

Unlike simple LLM summarization, this system uses a **Multi-Agent Architecture** orchestrated by **LangGraph**, ensuring clinical accuracy through self-reflection and distinct role separation.

### 🧠 Architecture
The pipeline consists of three specialized agents working in a cyclic graph:
1.  **Nurse Agent:** Scrubs PII and extracts Demographics & Clinical Risks in a single optimized pass.
2.  **Doctor Agent:** Synthesizes the dialogue and risk factors into a formal SOAP note.
3.  **Auditor Agent:** Performs a self-reflection loop, comparing the generated note against the transcript to detect hallucinations or missed red flags.

---

## 🛠️ Tech Stack
-   **Orchestration:** [LangGraph](https://github.com/langchain-ai/langgraph) (Stateful Multi-Agent Workflow)
-   **LLM:** Google Gemini 1.5 Flash / Grok Beta (via LangChain)
-   **Validation:** Pydantic (Strict JSON Output parsers)
-   **Frontend:** Streamlit
-   **Dataset:** MedDialog English / ChatDoctor (HuggingFace)

---

## 📂 Project Structure
```text
OriginMedical_Challenge/
├── src/
│   ├── agents.py       # Agent definitions (Nurse, Doctor, Auditor)
│   ├── graph.py        # LangGraph StateGraph construction
│   ├── schemas.py      # Pydantic data models (SOAPNote, RiskAnalysis)
│   ├── data_loader.py  # ETL pipeline for HuggingFace datasets
│   ├── evaluation.py   # ROUGE metric calculation logic
│   └── config.py       # Configuration and Logging setup
├── data/               # Local cache for datasets (GitIgnored)
├── app.py              # Streamlit Dashboard entry point
├── main.py             # CLI execution entry point
├── requirements.txt    # Python dependencies
└── llm_api_logs.jsonl  # (Generated) Logs of all LLM interactions