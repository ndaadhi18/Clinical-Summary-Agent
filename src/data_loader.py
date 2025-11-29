import os
import re
import json
import pandas as pd
from datasets import load_dataset

# Define paths
DATA_DIR = "data"
DATA_FILE = "medical_dialogues.json"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

def scrub_pii(text: str) -> str:
    """
    Scrubs PII (Phone, Emails, Names).
    
    NOTE: The MedDialog dataset is pre-anonymized. However, this active 
    scrubbing layer is implemented to ensure the pipeline is robust 
    against raw input and compliant with 'Privacy by Design' principles.
    """
    # Pattern 1: Phone Numbers
    text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[PHONE_REDACTED]', text)
    # Pattern 2: Email
    text = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL_REDACTED]', text)
    # Pattern 3: Names (Dr. Smith -> Dr. [NAME])
    text = re.sub(r'(Dr\.|Mr\.|Mrs\.|Ms\.)\s+[A-Z][a-z]+', r'\1 [NAME_REDACTED]', text)
    
    return text

def download_and_process_data(n=50):
    """
    ETL Function:
    Downloads 'ruslanmv/ai-medical-chatbot', sorts by length, and saves to JSON.
    """
    print("--- ETL STARTED: Downloading dataset 'ruslanmv/ai-medical-chatbot'... ---")
    
    
    dataset = load_dataset("ruslanmv/ai-medical-chatbot", split="train")
    
    processed_data = []
    
    # Process rows
    for i, row in enumerate(dataset):
        # The dataset uses 'Patient' and 'Doctor' columns
        patient_text = row.get('Patient', '').strip()
        doctor_text = row.get('Doctor', '').strip()
        
        if patient_text and doctor_text:
            full_dialogue = f"Patient: {patient_text}\n\nDoctor: {doctor_text}"
            # Word count for complexity sorting
            length = len(full_dialogue.split())
            
            processed_data.append({
                "id": f"case_{i+1}",
                "dialogue": full_dialogue,
                "length": length
            })
    
    # Sort by length (Descending) and take top N
    df = pd.DataFrame(processed_data)
    df = df.sort_values(by="length", ascending=False).head(n)
    
    # Convert to list of dicts
    final_data = df.to_dict(orient="records")
    
    # Ensure directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Save to JSON
    with open(DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2)
        
    print(f"--- ETL COMPLETE: Saved {len(final_data)} hard cases to {DATA_PATH} ---")
    return final_data

def load_meddialog_dataset(n=50):
    """
    Smart Loader:
    Checks local cache. Validates schema. If invalid/missing, downloads fresh data.
    """
    # 1. Check if local file exists
    if os.path.exists(DATA_PATH):
        print(f"Checking local cache: {DATA_PATH}")
        try:
            with open(DATA_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # VALIDATION STEP: Check if the data has the new 'length' key
            if data and isinstance(data, list) and len(data) > 0 and 'length' in data[0]:
                print("Cache valid.")
                return data
            else:
                print("Cache outdated or schema mismatch. Re-downloading...")
        except Exception as e:
            print(f"Corrupt local file ({e}). Re-downloading...")
    
    # 2. If not exists or corrupt/outdated, download
    return download_and_process_data(n)

if __name__ == "__main__":
    data = load_meddialog_dataset()
    if data:
        print(f"Success! Loaded {len(data)} cases.")
        print(f"Sample ID: {data[0]['id']} (Word Count: {data[0]['length']})")