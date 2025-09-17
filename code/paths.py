import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FPATH = os.path.join(ROOT_DIR, ".env")
CODE_DIR = os.path.join(ROOT_DIR, "code")
PROMPT_CONFIG = os.path.join(CODE_DIR, "config", "prompt_config.yml")
REASONING_CONFIG = os.path.join(CODE_DIR, "config", "reasoning_config.yml")
