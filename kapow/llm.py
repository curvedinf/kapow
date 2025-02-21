from dotenv import load_dotenv
from litellm import completion

load_dotenv()

def call_llm(prompt):
    response = completion(
        model='gemini/gemini-1.5-flash-latest',
        messages = [{"content": prompt, "role": "user"}]
    )
    return response['choices'][0]['message']['content']