import json
from src.data_loader import load_meddialog_dataset, scrub_pii
from src.graph import run_pipeline
from src.config import logger

def main():
    print("--- Starting Origin Medical Agentic Pipeline ---")
    
    # 1. Load Data
    print("Loading Dataset...")
    dialogues = load_meddialog_dataset(n=1) # Test with 1 first
    sample = dialogues[0]
    
    print(f"\nProcessing Dialogue ID: {sample.get('id', 'unknown')}")
    
    # 2. Pre-process (PII Scrubbing)
    clean_text = scrub_pii(sample['dialogue'])
    
    # 3. Run Agent Pipeline
    print("Invoking Agents (Nurse -> Doctor -> Auditor)...")
    result = run_pipeline(clean_text, dialogue_id=str(sample.get('id', '001')))
    
    # 4. Display Results
    print("\n" + "="*50)
    print("FINAL SOAP NOTE")
    print("="*50)
    print(json.dumps(result['soap_note'].dict(), indent=2))
    
    print("\n" + "-"*20)
    print("AUDITOR CRITIQUE")
    print("-"*20)
    print(result['critique_comments'])
    
    print("\n" + "="*50)
    print("Process Completed. Check 'llm_api_logs.jsonl' for call logs.")

if __name__ == "__main__":
    main()