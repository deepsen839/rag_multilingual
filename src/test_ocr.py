import os

os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

import fitz
from paddleocr import PaddleOCR


PDF = "data/pdfs/WhatsApp Image 2026-09-24 at 10.49.10 AM.pdf"
OUTPUT = "/tmp/ocr_test.png"


print("Opening PDF...", flush=True)

doc = fitz.open(PDF)
page = doc[0]

# Render at 2x
pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
pix.save(OUTPUT)

doc.close()

print("Image created:", OUTPUT, flush=True)
print("Image size:", pix.width, pix.height, flush=True)


print("Creating PaddleOCR...", flush=True)

ocr = PaddleOCR(
    lang="hi",
    device="cpu",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

print("PaddleOCR created.", flush=True)
print("STARTING predict()", flush=True)

results = ocr.predict(OUTPUT)

print("predict() returned", flush=True)

results = list(results)

print("RESULT COUNT:", len(results), flush=True)

for i, result in enumerate(results):
    print(f"\n===== RESULT {i} =====", flush=True)

    result.print()

    data = result.json

    print("\nJSON:", data, flush=True)

    if isinstance(data, dict):
        res = data.get("res", data)

        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])

        print("\n===== TEXT =====", flush=True)

        for text, score in zip(texts, scores):
            print(f"{float(score):.3f}  {text}", flush=True)