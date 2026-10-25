from transformers import AutoTokenizer
from .config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

_tokenizer = None

def tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    return _tokenizer


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text.strip():
        return []
    tok = tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    if not ids:
        return []
    chunks = []
    start = 0
    while start < len(ids):
        end = min(start + size, len(ids))
        value = tok.decode(ids[start:end], skip_special_tokens=True).strip()
        if value:
            chunks.append(value)
        if end >= len(ids):
            break
        start = max(end - overlap, start + 1)
    return chunks


def create_chunks(pages, document_id, filename):
    output = []
    for page in pages:
        for n, text in enumerate(chunk_text(page['text']), start=1):
            output.append({
                'chunk_id': f'{document_id}_p{page["page"]}_c{n}',
                'document_id': document_id,
                'filename': filename,
                'page': page['page'],
                'language': page.get('language', 'unknown'),
                'source_type': page.get('source_type', 'native'),
                'bbox': page.get('bbox'),
                'text': text,
            })
    return output
