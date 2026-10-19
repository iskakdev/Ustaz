from dotenv import load_dotenv
import os

load_dotenv()
SECRET_KEY = os.getenv('SECRET_KEY')
ALGORITHM = 'HS256'
ACCESS_TOKEN_LIFETIME = 60 * 24
DB_URL = os.getenv('DB_URL', 'sqlite:///./ustaz.db')
STORE_DIR = os.getenv('STORE_DIR', './stores')
LLM_MODEL = os.getenv('LLM_MODEL', 'ollama:qwen2.5:7b')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'ollama:bge-m3')
