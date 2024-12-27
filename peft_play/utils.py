import os
from dotenv import load_dotenv
from huggingface_hub import login

def get_hf_token():

    load_dotenv()
    return os.getenv("HF_TOKEN")
