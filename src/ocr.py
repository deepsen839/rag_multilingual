from .paddle_ocr import paddle_ocr_page


def ocr_page_auto(page):

    try:

        result = paddle_ocr_page(page)

        text = result["text"]

        confidence = result["confidence"]

        print(
            f"PaddleOCR: "
            f"chars={len(text)} "
            f"confidence={confidence:.3f}"
        )

        if text.strip():

            return (
                text,
                "paddleocr",
                "PP-OCR"
            )

    except Exception as exc:

        print(
            f"PaddleOCR failed: {exc}"
        )

    # -------------------------
    # Tesseract fallback
    # -------------------------

    try:

        from .tesseract_ocr import tesseract_ocr

        text = tesseract_ocr(page)

        print(
            f"Tesseract fallback: "
            f"chars={len(text)}"
        )

        return (
            text,
            "tesseract",
            "tesseract"
        )

    except Exception as exc:

        print(
            f"Tesseract failed: {exc}"
        )

    return (
        "",
        "failed",
        None
    )