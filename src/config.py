import os
import logging
import json
from datetime import datetime
from dotenv import load_dotenv

# Load API Keys
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# --- LLM Configuration ---
LLM_MODEL_NAME = "gemini-2.5-flash" 

# --- Logging Configuration ---
LOG_FILE = "llm_api_logs.jsonl"

def setup_logger():
    """Sets up a logger that writes to a file and console."""
    logger = logging.getLogger("AgentLogger")
    logger.setLevel(logging.INFO)
    
    # File Handler
    file_handler = logging.FileHandler(LOG_FILE)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()

def log_llm_interaction(agent_name, input_data, output_data):
    """
    Mandatory requirement: Log every LLM call.
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "input": str(input_data)[:500] + "...", # Truncate long inputs
        "output": str(output_data),
    }
    logger.info(json.dumps(log_entry))