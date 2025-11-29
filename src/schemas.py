from pydantic import BaseModel, Field
from typing import List, Optional

# ... (Keep PatientDemographics, SOAPNote, ClinicalRisk exactly as they are) ...

class PatientDemographics(BaseModel):
    age: Optional[str] = Field(description="Patient age if mentioned, else 'Unknown'")
    gender: Optional[str] = Field(description="Patient gender if mentioned, else 'Unknown'")

class ClinicalRisk(BaseModel):
    red_flags: List[str] = Field(description="Urgent concerns or severe symptoms.")
    confidence_score: int = Field(description="Confidence score (1-10).")

class SOAPNote(BaseModel):
    subjective: str = Field(description="Patient's complaints and symptoms.")
    objective: str = Field(description="Measurable data/vitals.")
    assessment: str = Field(description="Diagnosis.")
    plan: List[str] = Field(description="Actionable steps.")

# --- ADD THIS NEW CLASS ---
class NurseReport(BaseModel):
    """
    Combined extraction model to reduce API calls.
    """
    demographics: PatientDemographics
    risk_analysis: ClinicalRisk
# --------------------------

class AgentState(BaseModel):
    dialogue_id: str
    raw_dialogue: str
    scrubbed_dialogue: Optional[str] = None
    demographics: Optional[PatientDemographics] = None
    soap_note: Optional[SOAPNote] = None
    risk_analysis: Optional[ClinicalRisk] = None
    critique_comments: Optional[str] = None
    final_summary: Optional[str] = None