import os
import yaml
from dotenv import load_dotenv
from pathlib import Path
from typing import Union, Optional
from paths import DATA_DIR, ENV_PATH


def load_exploit(exploit_external_id="1dfc5bee07ff"):
  """
  Load exploit data from a JSON file based on the given external ID.
  Args:
      exploit_external_id (str): The external ID of the exploit to load. Defaults to "1dfc5bee07ff".
  Returns:
      dict: The loaded exploit data.
  Raises:
      FileNotFoundError: If the exploit file does not exist. 
      IOError: If there is an error reading the file.
  """
  exploit_fpath = Path(os.path.join(DATA_DIR, f"{exploit_external_id}.json"))

  if not exploit_fpath.exists():
    raise FileNotFoundError(f"Exploit file not found: {exploit_fpath}")
  
  try:
    with open(exploit_fpath, "r") as f:
      exploit_data = f.read()
  except Exception as e:
    raise IOError(f"Error reading exploit file: {e}")


def load_all_exploits(exploit_dir: str = DATA_DIR) -> list:
  """
  Load all exploit data from JSON files in the specified directory.
  Args:
      exploit_dir (str): Directory containing exploit JSON files. Defaults to DATA_DIR.
  Returns:
      list: A list of loaded exploit data dictionaries. 
  Raises:
      IOError: If there is an error reading any of the files.
  """
  exploits = []

  for exploit_id in os.listdir(exploit_dir):
    if exploit_id.endswith(".json"):
      try:
        exploit = load_exploit(exploit_external_id=exploit_id.replace(".json", ""))
        exploits.append(exploit)
      except Exception as e:
        print(f"Error loading exploit {exploit_id}: {e}")

  return exploits


def load_yaml_config(file_path: Union[str, Path]) -> dict:
  """
  Load a YAML configuration file and return its contents as a dictionary.
  Args: 
      file_path (str or Path): Path to the YAML configuration file.
  Returns:
      dict: The contents of the YAML file as a dictionary.
  Raises:
      FileNotFoundError: If the file does not exist.
      IOError: If there is an error reading the file.
  """
  file_path = Path(file_path)

  if not file_path.exists():
    raise FileNotFoundError(f"YAML config file not found: {file_path}")
  
  try:
    with open(file_path, "r") as f:
      config = yaml.safe_load(f)
  except Exception as e:
    raise IOError(f"Error reading YAML config file: {e}")  
  
  return config


def load_env(api_key_type="GOOGLE_API_KEY") -> str:
  """
  Load environment variables from a .env file and retrieve a specific API key.
  Args:
      api_key_type (str): The type of API key to retrieve. Defaults to "GOOGLE_API_KEY".
  Returns: 
      str: The requested API key.
  Raises:
      ValueError: If the specified API key is not found in the environment variables.
  """
  load_dotenv(ENV_PATH, override=True)

  api_key = os.getenv(api_key_type)
  if not api_key:
    raise ValueError(f"{api_key_type} not found in environment variables.")
  return api_key 


def save_text_to_file(
    text: str,
    file_path: Union[str, Path],
    header: Optional[str] = None
  ) -> None:
  """
  Save text to a file, optionally adding a header.
  Args:
      text (str): The text content to save.
      file_path (str or Path): The path to the file where the text will be saved 
      header (str, optional): An optional header to add at the top of the file.
  Raises:
      IOError: If there is an error writing to the file.
  Returns: None
  """
  try: 
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
      if header:
        f.write(f"{header}\n\n")
        f.write("# " + "=" * 60 + "\n\n")
      f.write(text)
  
  except Exception as e:
    raise IOError(f"Error writing to file {file_path}: {e}")