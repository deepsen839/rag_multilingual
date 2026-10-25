import fitz

from .paddle_ocr import paddle_ocr_page


PDF_PATH = "data/pdfs/WhatsApp Image 2026-09-24 at 10.49.10 AM.pdf"


def main():

    print("Opening PDF...", flush=True)

    doc = fitz.open(PDF_PATH)

    print(
        f"PDF opened: {len(doc)} pages",
        flush=True,
    )

    for page_no, page in enumerate(
        doc,
        start=1,
    ):

        print(
            f"\n{'=' * 60}",
            flush=True,
        )

        print(
            f"PAGE {page_no}",
            flush=True,
        )

        print(
            f"{'=' * 60}",
            flush=True,
        )

        try:

            print(
                "Running PaddleOCR...",
                flush=True,
            )

            result = paddle_ocr_page(page)

            print(
                "\nOCR RESULT",
                flush=True,
            )

            print(
                f"Confidence: "
                f"{result['confidence']:.3f}",
                flush=True,
            )

            print(
                f"Detected boxes: "
                f"{len(result['boxes'])}",
                flush=True,
            )

            print(
                "\n----- OCR TEXT -----",
                flush=True,
            )

            print(
                result["text"],
                flush=True,
            )

            print(
                "----- END OCR TEXT -----",
                flush=True,
            )

        except Exception as e:

            print(
                "\nOCR ERROR:",
                flush=True,
            )

            print(
                type(e).__name__,
                flush=True,
            )

            print(
                str(e),
                flush=True,
            )

            import traceback

            traceback.print_exc()

    doc.close()

    print(
        "\nOCR test finished.",
        flush=True,
    )


if __name__ == "__main__":
    main()