import hashlib
import json
import fitz

from .config import MIN_NATIVE_CHARS, PROCESSED_DIR
from .ocr import ocr_page_auto
from .language import detect_language
from .text_cleaning import normalize_text


def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def process_pdf(path):

    pages = []

    doc = fitz.open(path)

    for page_no, page in enumerate(
        doc,
        start=1,
    ):

        native = normalize_text(
            page.get_text("text")
        )

        # -----------------------------------
        # NORMAL TEXT PDF
        # -----------------------------------

        if len(native) >= MIN_NATIVE_CHARS:

            text = native

            source = "native"

            ocr_model = None

            logger_msg = (
                f"Page {page_no}: native text "
                f"chars={len(native)}"
            )

            print(logger_msg)

        # -----------------------------------
        # SCANNED / IMAGE PDF
        # -----------------------------------

        else:

            print(
                f"Page {page_no}: insufficient native "
                f"text -> running OCR"
            )

            text, source, ocr_model = ocr_page_auto(
                page
            )

            text = normalize_text(text)

            print(
                f"Page {page_no}: OCR source={source}, "
                f"chars={len(text)}"
            )

        # -----------------------------------
        # LANGUAGE
        # -----------------------------------

        lang = detect_language(text)

        rect = page.rect

        pages.append(
            {
                "page": page_no,
                "text": text,
                "language": lang,
                "source_type": source,
                "ocr_language": ocr_model,
                "bbox": [
                    0,
                    0,
                    float(rect.width),
                    float(rect.height),
                ],
            }
        )

    doc.close()

    return pages


def save_processed(
    document_id,
    pages,
):

    path = (
        PROCESSED_DIR
        / f"{document_id}.json"
    )

    path.write_text(
        json.dumps(
            pages,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


def is_reliable_native_text(text):
    text = normalize_text(text)

    if len(text) < MIN_NATIVE_CHARS:
        return False

    words = text.split()

    if len(words) < 10:
        return False

    # Reject pages where extracted text is mostly garbage
    alpha_chars = sum(c.isalpha() for c in text)

    if alpha_chars / max(len(text), 1) < 0.40:
        return False

    return True