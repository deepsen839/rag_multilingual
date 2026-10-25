import json
from pathlib import Path
from .config import PDF_DIR, REGISTRY_PATH, BM25_PATH
from .pdf_processor import sha256, process_pdf, save_processed
from .text_cleaning import repeated_lines, remove_repeated_lines
from .chunker import create_chunks
from .qdrant_store import QdrantStore
from .bm25_store import BM25Store


def load_registry():
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding='utf-8'))
    return {}

def save_registry(reg):
    REGISTRY_PATH.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding='utf-8')

def all_processed_chunks(registry):
    chunks = []
    for doc_id in registry:
        path = Path(__file__).resolve().parent.parent / 'data' / 'processed' / f'{doc_id}.json'
        if path.exists():
            pages = json.loads(path.read_text(encoding='utf-8'))
            chunks.extend(create_chunks(pages, doc_id, registry[doc_id]['filename']))
    return chunks

def main():
    registry = load_registry()
    qdrant = QdrantStore()
    changed = False
    for path in sorted(PDF_DIR.glob('*.pdf')):
        doc_id = path.stem
        digest = sha256(path)
        if doc_id in registry and registry[doc_id]['sha256'] == digest:
            print(f'SKIP unchanged: {path.name}')
            continue
        if doc_id in registry:
            print(f'REBUILD changed: {path.name}')
            qdrant.delete_document(doc_id)
        print(f'PROCESS: {path.name}')
        pages = process_pdf(path)
        repeated = repeated_lines(pages)
        pages = remove_repeated_lines(pages, repeated)
        save_processed(doc_id, pages)
        chunks = create_chunks(pages, doc_id, path.name)
        qdrant.upsert_chunks(chunks)
        registry[doc_id] = {'filename': path.name, 'sha256': digest, 'pages': len(pages), 'chunks': len(chunks)}
        changed = True
    # Remove files no longer present.
    current = {p.stem for p in PDF_DIR.glob('*.pdf')}
    for doc_id in list(registry):
        if doc_id not in current:
            qdrant.delete_document(doc_id)
            del registry[doc_id]
            changed = True
    save_registry(registry)
    if changed:
        chunks = all_processed_chunks(registry)
        BM25Store(BM25_PATH).build(chunks)
        print(f'BM25 rebuilt: {len(chunks)} chunks')
    print('Ingestion complete.')

if __name__ == '__main__':
    main()
