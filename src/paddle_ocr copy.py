import logging
import os

import fitz
import numpy as np

# Paddle CPU workaround must be set before importing Paddle.
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

from paddleocr import PaddleOCR

from .config import OCR_DPI

logger = logging.getLogger(__name__)

_ocr = None


def get_ocr():
    """
    Lazy-load PaddleOCR.

    Hindi/English mixed OCR:
    lang="hi" uses the Devanagari multilingual
    recognition model that can recognize Hindi + English.
    """

    global _ocr

    if _ocr is None:
        logger.info("Loading PaddleOCR Hindi/English model...")

        _ocr = PaddleOCR(
            lang="hi",
            device="cpu",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            cpu_threads=4,
        )

        logger.info("PaddleOCR loaded.")

    return _ocr


def render_page_to_file(page, output_path, dpi=144):
    """
    Render PDF page exactly like the known-working OCR test.
    """

    scale = dpi / 72.0

    pix = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        alpha=False,
    )

    pix.save(str(output_path))

    return pix.width, pix.height


def paddle_ocr_page(page):
    """
    OCR one PDF page using the same PNG path workflow
    that was verified to work.
    """

    import tempfile

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
            dpi=144,
        )

        logger.info(
            "Running PaddleOCR: image=%sx%s path=%s",
            width,
            height,
            image_path,
        )

        # IMPORTANT:
        # Pass the PNG PATH, exactly like the working test.
        results = list(
            ocr.predict(image_path)
        )

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

            logger.warning(
                "PaddleOCR detected no text."
            )

            return {
                "text": "",
                "confidence": 0.0,
                "boxes": [],
            }

        # Reading order
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
                return (0, 0)

            try:
                return (
                    float(box[1]),
                    float(box[0]),
                )
            except (
                TypeError,
                ValueError,
                IndexError,
            ):
                return (0, 0)

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

    finally:

        try:
            os.remove(image_path)
        except OSError:
            pass

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

    def get_box_position(box):

        if box is None:
            return 0.0, 0.0

        try:
            # rec_boxes:
            # [x1, y1, x2, y2]

            x1 = float(box[0])
            y1 = float(box[1])

            return x1, y1

        except (
            TypeError,
            ValueError,
            IndexError,
        ):
            return 0.0, 0.0

    # Basic top-to-bottom / left-to-right ordering.
    items.sort(
        key=lambda item: (
            get_box_position(item[2])[1],
            get_box_position(item[2])[0],
        )
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

    # ---------------------------------------------------------
    # Final text
    # ---------------------------------------------------------

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


def paddle_ocr_image(image_path):
    ocr = get_ocr()

    results = list(
        ocr.predict(image_path)
    )

    texts = []
    scores = []
    boxes = []

    for result in results:

        data = result.json

        if callable(data):
            data = data()

        if not isinstance(data, dict):
            continue

        res = data.get("res", data)

        rec_texts = res.get(
            "rec_texts",
            []
        )

        rec_scores = res.get(
            "rec_scores",
            []
        )

        rec_boxes = res.get(
            "rec_boxes",
            []
        )

        for i, raw_text in enumerate(rec_texts):

            text = str(raw_text).strip()

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
                    ValueError
                ):
                    score = 0.0

            box = (
                rec_boxes[i]
                if i < len(rec_boxes)
                else None
            )

            texts.append(text)
            scores.append(score)
            boxes.append(box)

    if not texts:

        return {
            "text": "",
            "confidence": 0.0,
            "boxes": [],
        }

    items = list(
        zip(
            texts,
            scores,
            boxes
        )
    )

    items.sort(
        key=lambda item: (
            float(item[2][1]),
            float(item[2][0])
        )
        if item[2] is not None
        else (0, 0)
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

    return {
        "text": "\n".join(
            ordered_texts
        ).strip(),

        "confidence": (
            sum(ordered_scores)
            / len(ordered_scores)
        ),

        "boxes": ordered_boxes,
    }