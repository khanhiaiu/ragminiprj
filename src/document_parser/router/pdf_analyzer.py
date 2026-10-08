import unicodedata
from dataclasses import asdict, dataclass

from ..config import PDFAnalyzerConfig


def text_metrics(text: str) -> tuple[float, float, float]:
    if not text:
        return 1.0, 0.0, 1.0
    invalid = sum(
        c == "\ufffd" or (unicodedata.category(c) in {"Cc", "Cs", "Co", "Cn"} and c not in "\n\r\t")
        for c in text
    )
    printable = sum(c.isprintable() or c in "\n\r\t" for c in text) / len(text)
    ratio = invalid / len(text)
    return printable, ratio, max(0.0, printable * (1 - ratio))


def rectangle_union_area(rectangles: list[tuple]) -> float:
    """Sweep overlapping image rectangles so coverage cannot double-count."""
    xs = sorted({r[0] for r in rectangles} | {r[2] for r in rectangles})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        spans = sorted((r[1], r[3]) for r in rectangles if r[0] < right and r[2] > left)
        bottom = float("-inf")
        height = 0.0
        for top, end in spans:
            height += max(0, end - max(top, bottom))
            bottom = max(bottom, end)
        area += (right - left) * height
    return area


@dataclass
class PageAnalysis:
    page_number: int
    page_type: str
    text_length: int
    text_block_count: int
    image_count: int
    image_coverage: float
    text_quality_score: float
    printable_ratio: float
    invalid_char_ratio: float
    text_density: float
    page_area: float
    blank: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class PDFAnalyzer:
    def __init__(self, config: PDFAnalyzerConfig | None = None):
        self.config = config or PDFAnalyzerConfig()

    def classify(
        self,
        *,
        text: str,
        text_block_count: int,
        image_count: int,
        image_coverage: float,
        page_area: float,
        page_number: int = 1,
    ) -> PageAnalysis:
        cfg = self.config
        printable, invalid, quality = text_metrics(text)
        length = len(text.strip())
        density = length / max(page_area, 1)
        usable = (
            length >= cfg.minimum_text_chars
            and text_block_count >= cfg.minimum_text_blocks
            and printable >= cfg.minimum_printable_ratio
            and invalid <= cfg.max_invalid_char_ratio
            and quality >= cfg.minimum_text_quality
            and density >= cfg.minimum_text_density
        )
        blank = not text.strip() and image_count == 0
        # Sparse but valid title pages use native extraction unless dominated by an image.
        sparse_usable = (
            length > 0
            and quality >= cfg.minimum_text_quality
            and invalid <= cfg.max_invalid_char_ratio
        )
        if blank:
            kind = "digital"
        elif not usable and (
            not sparse_usable or image_coverage >= cfg.scan_image_coverage_threshold
        ):
            kind = "scanned"
        elif image_count and image_coverage >= cfg.hybrid_image_coverage_threshold:
            kind = "hybrid"
        else:
            kind = "digital"
        return PageAnalysis(
            page_number,
            kind,
            length,
            text_block_count,
            image_count,
            min(1.0, image_coverage),
            quality,
            printable,
            invalid,
            density,
            page_area,
            blank,
        )

    def analyze_page(self, page) -> PageAnalysis:
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0]
        images = page.get_image_info()
        rectangles = []
        for image in images:
            r = image["bbox"]
            clipped = (
                max(page.rect.x0, r[0]),
                max(page.rect.y0, r[1]),
                min(page.rect.x1, r[2]),
                min(page.rect.y1, r[3]),
            )
            if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                rectangles.append(clipped)
        area = page.rect.width * page.rect.height
        return self.classify(
            text="\n".join(b[4] for b in text_blocks),
            text_block_count=len(text_blocks),
            image_count=len(images),
            image_coverage=rectangle_union_area(rectangles) / max(area, 1),
            page_area=area,
            page_number=page.number + 1,
        )
