from litellm import embedding

def embed(text):
    return embedding(
        input=[text],
        model='voyage/voyage-3-lite'
    )['data'][0]['embedding']