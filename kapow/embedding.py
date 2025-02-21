from dotenv import load_dotenv
from litellm import embedding

load_dotenv()

def embed(text):
    return embedding(
        input=[text],
        model='gemini/text-embedding-004'
    )['data'][0]['embedding']