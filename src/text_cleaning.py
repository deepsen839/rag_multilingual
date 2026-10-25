import re
import unicodedata
from collections import Counter


def normalize_text(text):
    text = unicodedata.normalize('NFKC', text or '')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def repeated_lines(pages, threshold=0.5):
    counter = Counter()
    for page in pages:
        seen = set()
        for line in page['text'].splitlines():
            line = line.strip()
            if line:
                seen.add(line)
        for line in seen:
            counter[line] += 1
    minimum = max(2, int(len(pages) * threshold))
    return {line for line, count in counter.items() if count >= minimum}


def remove_repeated_lines(pages, repeated):
    out = []
    for page in pages:
        lines = [x.strip() for x in page['text'].splitlines() if x.strip() and x.strip() not in repeated]
        p = dict(page)
        p['text'] = normalize_text('\n'.join(lines))
        out.append(p)
    return out
