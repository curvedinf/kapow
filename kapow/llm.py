from litellm import completion

def call_llm(prompt):
    response = completion(
        input=prompt,
        model='gemini-flash'
    )
    return response['data'][0]['text']