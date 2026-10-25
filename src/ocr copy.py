import re
from io import BytesIO
import pytesseract
from PIL import Image
from langdetect import detect, DetectorFactory
from .config import OCR_DPI, OCR_GENERIC_LANG

DetectorFactory.seed = 0

# Extend this map for the languages in your corpus.
TESSERACT_LANG_MAP = {
    'en': 'eng', 'hi': 'hin', 'de': 'deu', 'fr': 'fra', 'es': 'spa',
    'it': 'ita', 'pt': 'por', 'ru': 'rus', 'ar': 'ara', 'zh-cn': 'chi_sim',
    'zh-tw': 'chi_tra', 'ja': 'jpn', 'ko': 'kor', 'nl': 'nld', 'tr': 'tur',
    'pol': 'pol', 'uk': 'ukr', 'vi': 'vie', 'ind': 'ind', 'tha': 'tha',
}


def render_page(page):
    pix = page.get_pixmap(dpi=OCR_DPI, alpha=False)
    return Image.open(BytesIO(pix.tobytes('png')))


def detect_language(text):
    if not text or len(re.sub(r'\s+', '', text)) < 20:
        return 'unknown'
    try:
        return detect(text)
    except Exception:
        return 'unknown'


def installed_tesseract_languages():
    try:
        return set(pytesseract.get_languages(config=''))
    except Exception:
        return {'eng'}


# def generic_ocr(image):
#     return pytesseract.image_to_string(image, lang=OCR_GENERIC_LANG, config='--psm 3')

def generic_ocr(image):
    return pytesseract.image_to_string(
        image,
        lang="eng+hin",
        config="--psm 6"
    )

def ocr_with_language(image, lang_code):
    tess_lang = TESSERACT_LANG_MAP.get(lang_code)
    if not tess_lang:
        return None, None
    installed = installed_tesseract_languages()
    # Every component in a combined language string must be installed.
    if not all(x in installed for x in tess_lang.split('+')):
        return None, None
    try:
        text = pytesseract.image_to_string(image, lang=tess_lang, config='--psm 3')
        return text, tess_lang
    except pytesseract.TesseractError:
        return None, None


# def ocr_page_auto(page):
#     """Generic OCR -> language detection -> language-specific OCR."""
#     image = render_page(page)
#     raw_text = generic_ocr(image)
#     detected_lang = detect_language(raw_text)

#     refined_text, tess_lang = ocr_with_language(image, detected_lang)
#     if refined_text and len(refined_text.strip()) >= len(raw_text.strip()) * 0.35:
#         return refined_text, detected_lang, tess_lang or OCR_GENERIC_LANG

#     return raw_text, detected_lang, OCR_GENERIC_LANG
def ocr_page_auto(page):
    """
    OCR pipeline:
    1. Render page
    2. Try multiple orientations
    3. Select OCR result with strongest text signal
    4. Detect language
    5. Run language-specific OCR
    """

    image = render_page(page)

    candidates = []

    for angle in [0, 90, 180, 270]:

        rotated = image.rotate(
            angle,
            expand=True
        )

        try:
            text = pytesseract.image_to_string(
                rotated,
                lang="eng+hin",
                config="--psm 6"
            )
        except pytesseract.TesseractError:
            text = ""

        score = ocr_quality_score(text)

        candidates.append(
            (
                score,
                angle,
                text,
                rotated
            )
        )

    # Best orientation
    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    best_score, best_angle, raw_text, best_image = candidates[0]

    detected_lang = detect_language(
        raw_text
    )

    refined_text, tess_lang = (
        ocr_with_language(
            best_image,
            detected_lang
        )
    )

    if (
        refined_text
        and len(refined_text.strip())
        >= len(raw_text.strip()) * 0.35
    ):
        return (
            refined_text,
            detected_lang,
            tess_lang or "eng+hin"
        )

    return (
        raw_text,
        detected_lang,
        "eng+hin"
    )

def ocr_quality_score(text):
    if not text:
        return 0

    compact = re.sub(
        r"\s+",
        "",
        text
    )

    alphanumeric = re.findall(
        r"[A-Za-z0-9\u0900-\u097F]",
        text
    )

    return (
        len(alphanumeric)
        + min(len(compact), 500) * 0.1
    )