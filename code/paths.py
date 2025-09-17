import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT_DIR, ".env")
CODE_DIR = os.path.join(ROOT_DIR, "code")
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
PROMPT_CONFIG_PATH = os.path.join(CODE_DIR, "config", "prompt_config.yaml")
REASONING_CONFIG_PATH = os.path.join(CODE_DIR, "config", "reasoning_config.yaml")
VECTOR_DB_DIR = os.path.join(OUTPUT_DIR, "vector_db")