import requests
from .config import GROQ_API_KEY, GROQ_MODEL

def build_context(results):
    return '\n\n'.join(
        f"SOURCE {i}\nFile: {r['payload']['filename']}\nPage: {r['payload']['page']}\nLanguage: {r['payload'].get('language','unknown')}\nContent:\n{r['payload']['text']}"
        for i, r in enumerate(results, 1)
    )

def fallback_answer(question, results):
    if not results:
        return 'I could not find sufficient information in the provided documents.'
    return 'LLM is not configured. Relevant retrieved passages:\n\n' + build_context(results)

def generate_answer(question, results):
    if not GROQ_API_KEY:
        return fallback_answer(question, results)
    prompt = f'''You are a multilingual document-grounded RAG assistant.\nAnswer ONLY from the sources below. Do not invent facts. If the sources do not contain enough information, say so. Preserve numbers, dates and names. Give concise answers and cite sources as [filename, p. X].\n\nQuestion:\n{question}\n\nSources:\n{build_context(results)}'''
    response = requests.post(
        'https://api.groq.com/openai/v1/chat/completions',
        headers={'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json'},
        json={'model': GROQ_MODEL, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.0},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']
