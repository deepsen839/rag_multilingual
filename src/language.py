from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

def detect_language(text):
    if not text or len(text.strip()) < 20:
        return 'unknown'
    try:
        return detect(text)
    except Exception:
        return 'unknown'
