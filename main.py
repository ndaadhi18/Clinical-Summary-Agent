import json
from src.data_loader import load_meddialog_dataset, scrub_pii
from src.graph import run_pipeline

def main():
    print("--- Starting Origin Medical Agentic Pipeline (CLI Mode) ---")
    
    # 1. Load Data
    print("Loading Dataset (Fetching top 50)...")
    dialogues = load_meddialog_dataset(n=50) 
    
    if not dialogues:
        print("Error: No data loaded.")
        return

    sample = dialogues[-1] # last case has less words
    
    print(f"\nProcessing Case Index 50 (ID: {sample.get('id', 'unknown')})")
    print(f"Word Count: {sample.get('length', 'N/A')}")
    
    # 2. Pre-process (PII Scrubbing) OPTIONAL STEP [the dataset is already pre-anonymized]
    clean_text = scrub_pii(sample['dialogue'])
    
    # 3. Run Agent Pipeline
    print("Invoking Agents (Nurse -> Doctor -> Auditor)...")
    result = run_pipeline(clean_text, dialogue_id=str(sample.get('id', '001')))
    
    # 4. Display Results
    if result.get("error"):
        print("\n❌ PIPELINE FAILED")
        print(f"Error: {result['error']}")
    else:
        print("\n" + "="*50)
        print("FINAL SOAP NOTE")
        print("="*50)
        
        print(json.dumps(result['soap_note'].model_dump(), indent=2))
        
        print("\n" + "-"*20)
        print("AUDITOR CRITIQUE")
        print("-"*20)
        print(result['critique_comments'])
    
    print("\n" + "="*50)
    print("Process Completed. Check 'llm_api_logs.jsonl' for detailed logs.")

if __name__ == "__main__":
    main()