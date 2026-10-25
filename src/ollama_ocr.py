import logging
from pathlib import Path

from ollama import Client

from .config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)


class OllamaOCR:
    """
    OCR backend using Ollama + GLM-OCR.

    Intended only for scanned/image-only PDF pages.
    """

    def __init__(self):
        self.client = Client(
            host=OLLAMA_HOST,
            timeout=OLLAMA_TIMEOUT,
        )

        self.model = OLLAMA_MODEL

    def health_check(self):
        """
        Check whether Ollama is reachable and model is available.
        """

        try:
            response = self.client.list()

            models = []

            # Ollama SDK versions can return objects or dictionaries.
            if hasattr(response, "models"):
                for model in response.models:
                    if hasattr(model, "model"):
                        models.append(model.model)
                    elif isinstance(model, dict):
                        models.append(model.get("model"))

            elif isinstance(response, dict):
                for model in response.get("models", []):
                    if isinstance(model, dict):
                        models.append(model.get("name") or model.get("model"))

            logger.info("Ollama models: %s", models)

            return self.model in models

        except Exception as exc:
            logger.error("Ollama health check failed: %s", exc)
            return False

    def extract_text(self, image_path: str | Path) -> str:
        """
        Extract text from a document image.

        Parameters
        ----------
        image_path:
            Path to PNG/JPEG image.

        Returns
        -------
        str
            OCR text.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"OCR image does not exist: {image_path}"
            )

        prompt = (
            "Text Recognition:\n"
            "Extract all readable text from this document image.\n"
            "Preserve the original reading order.\n"
            "Preserve headings, paragraphs, numbers and labels.\n"
            "Do not summarize.\n"
            "Do not explain the image.\n"
            "Return only the extracted document text."
        )

        logger.info(
            "Running Ollama OCR: model=%s image=%s",
            self.model,
            image_path.name,
        )

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [str(image_path)],
                }
            ],
        )

        text = response.message.content
        print("\n========== GLM-OCR OUTPUT ==========")
        print(repr(text))
        print("====================================\n")
        if not text:
            logger.warning(
                "Ollama returned empty OCR result for %s",
                image_path,
            )
            return ""

        text = text.strip()

        logger.info(
            "Ollama OCR completed: chars=%d",
            len(text),
        )

        return text