import cv2
import fitz
import numpy as np
import pytesseract

from .config import OCR_DPI


def render_page(page, dpi=300):
    scale = dpi / 72

    matrix = fitz.Matrix(scale, scale)

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    img = np.frombuffer(
        pix.samples,
        dtype=np.uint8
    )

    img = img.reshape(
        pix.height,
        pix.width,
        pix.n
    )

    if pix.n == 3:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR
        )

    elif pix.n == 4:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGBA2BGR
        )

    return img


def rotate_image(image, angle):

    if angle == 0:
        return image

    if angle == 90:
        return cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE
        )

    if angle == 180:
        return cv2.rotate(
            image,
            cv2.ROTATE_180
        )

    if angle == 270:
        return cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )

    return image


def preprocess(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    # Upscale
    gray = cv2.resize(
        gray,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )

    # Mild denoise
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return gray, binary


def ocr_candidate(image, psm):

    config = f"--oem 3 --psm {psm}"

    data = pytesseract.image_to_data(
        image,
        config=config,
        output_type=pytesseract.Output.DICT
    )

    texts = []
    confidences = []

    for text, conf in zip(
        data["text"],
        data["conf"]
    ):

        text = text.strip()

        try:
            conf = float(conf)
        except Exception:
            conf = -1

        if text and conf >= 0:

            texts.append(text)
            confidences.append(conf)

    result = " ".join(texts)

    if confidences:
        avg_conf = sum(confidences) / len(confidences)
    else:
        avg_conf = 0

    useful_words = len(texts)

    # Combined score
    score = (
        avg_conf
        + min(useful_words * 2, 40)
        + min(len(result) / 20, 40)
    )

    return {
        "text": result,
        "confidence": avg_conf,
        "words": useful_words,
        "score": score
    }


def tesseract_ocr(page):

    image = render_page(
        page,
        dpi=300
    )

    candidates = []

    rotations = [
        0,
        90,
        180,
        270
    ]

    psms = [
        3,
        6,
        11,
        12
    ]

    for angle in rotations:

        rotated = rotate_image(
            image,
            angle
        )

        gray, binary = preprocess(
            rotated
        )

        variants = [
            gray,
            binary
        ]

        for variant in variants:

            for psm in psms:

                result = ocr_candidate(
                    variant,
                    psm
                )

                result["angle"] = angle
                result["psm"] = psm

                candidates.append(result)

    if not candidates:
        return ""

    best = max(
        candidates,
        key=lambda x: x["score"]
    )

    print(
        "Best Tesseract result:",
        f"angle={best['angle']}",
        f"psm={best['psm']}",
        f"confidence={best['confidence']:.1f}",
        f"words={best['words']}",
        f"chars={len(best['text'])}"
    )

    print(
        "OCR text:",
        repr(best["text"])
    )

    return best["text"].strip()