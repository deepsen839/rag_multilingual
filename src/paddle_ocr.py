import logging
import os
import tempfile

import fitz

# Paddle CPU workaround must be set before importing Paddle.
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

from paddleocr import PaddleOCR

from .config import OCR_DPI


logger = logging.getLogger(__name__)

_ocr = None


def get_ocr():
    global _ocr

    if _ocr is None:
        logger.info("Loading PaddleOCR Hindi/English CPU model...")

        _ocr = PaddleOCR(
            lang="hi",
            device="cpu",
            enable_mkldnn=False,

            # CPU-safe settings
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,

            cpu_threads=4,

            # Important: use lightweight models
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="devanagari_PP-OCRv5_mobile_rec",
        )

        logger.info("PaddleOCR loaded.")

    return _ocr


def render_page_to_file(
    page,
    output_path,
    dpi=144,
):
    """
    Render PDF page to PNG.
    """

    scale = dpi / 72.0

    pix = page.get_pixmap(
        matrix=fitz.Matrix(
            scale,
            scale,
        ),
        alpha=False,
    )

    pix.save(
        str(output_path)
    )

    return pix.width, pix.height


def _parse_ocr_results(results):
    """
    Convert PaddleOCR results into:
        text
        confidence
        boxes
    """

    texts = []
    scores = []
    boxes = []

    for result in results:

        data = result.json

        if callable(data):
            data = data()

        if not isinstance(data, dict):
            continue

        res = data.get(
            "res",
            data,
        )

        rec_texts = res.get(
            "rec_texts",
            [],
        )

        rec_scores = res.get(
            "rec_scores",
            [],
        )

        rec_boxes = res.get(
            "rec_boxes",
            [],
        )

        for i, raw_text in enumerate(
            rec_texts
        ):

            text = str(
                raw_text
            ).strip()

            if not text:
                continue

            score = 0.0

            if i < len(rec_scores):

                try:

                    score = float(
                        rec_scores[i]
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    score = 0.0

            box = None

            if i < len(rec_boxes):
                box = rec_boxes[i]

            texts.append(text)
            scores.append(score)
            boxes.append(box)

    if not texts:

        return {
            "text": "",
            "confidence": 0.0,
            "boxes": [],
        }

    # ---------------------------------------------------------
    # Reading order
    # ---------------------------------------------------------

    items = list(
        zip(
            texts,
            scores,
            boxes,
        )
    )

    def sort_key(item):

        box = item[2]

        if box is None:
            return (
                0.0,
                0.0,
            )

        try:

            x1 = float(
                box[0]
            )

            y1 = float(
                box[1]
            )

            return (
                y1,
                x1,
            )

        except (
            TypeError,
            ValueError,
            IndexError,
        ):

            return (
                0.0,
                0.0,
            )

    items.sort(
        key=sort_key
    )

    ordered_texts = [
        item[0]
        for item in items
    ]

    ordered_scores = [
        item[1]
        for item in items
    ]

    ordered_boxes = [
        item[2]
        for item in items
    ]

    text = "\n".join(
        ordered_texts
    ).strip()

    confidence = (
        sum(ordered_scores)
        / len(ordered_scores)
    )

    logger.info(
        "PaddleOCR detected %d text regions, "
        "average confidence=%.3f",
        len(ordered_texts),
        confidence,
    )

    return {
        "text": text,
        "confidence": confidence,
        "boxes": ordered_boxes,
    }


def paddle_ocr_page(page):
    """
    OCR one PDF page.

    Uses the PNG file-path workflow that was
    verified to work with your PaddleOCR setup.
    """

    ocr = get_ocr()

    with tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False,
    ) as tmp:

        image_path = tmp.name

    try:

        width, height = render_page_to_file(
            page,
            image_path,
            dpi=OCR_DPI,
        )

        logger.info(
            "Running PaddleOCR: image=%sx%s path=%s",
            width,
            height,
            image_path,
        )

        results = list(
            ocr.predict(
                image_path
            )
        )

        return _parse_ocr_results(
            results
        )

    finally:

        try:

            os.remove(
                image_path
            )

        except OSError:

            pass


def paddle_ocr_image(
    image_path
):
    """
    OCR an existing image file.
    """

    ocr = get_ocr()

    results = list(
        ocr.predict(
            image_path
        )
    )

    return _parse_ocr_results(
        results
    )