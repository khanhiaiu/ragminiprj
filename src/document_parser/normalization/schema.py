import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ElementType = Literal["heading", "paragraph", "list", "table", "image", "formula", "header", "footer", "unknown"]
PageType = Literal["digital", "scanned", "hybrid"]


class Element(BaseModel):
    model_config = ConfigDict(extra="forbid")
    element_id: str
    element_type: ElementType
    text: str = ""
    page_number: int | None = Field(default=None, ge=1)
    bbox: tuple[float, float, float, float] | None = None
    parent_id: str | None = None
    section_id: str | None = None
    order: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_bbox(self):
        if self.bbox and not all(math.isfinite(value) for value in self.bbox):
            raise ValueError("bbox coordinates must be finite")
        if self.bbox and (self.bbox[2] < self.bbox[0] or self.bbox[3] < self.bbox[1]):
            raise ValueError("bbox must be [left, top, right, bottom]")
        return self


class Page(BaseModel):
    page_number: int = Field(ge=1)
    width: float | None = None
    height: float | None = None
    page_type: PageType | None = None
    status: Literal["ok", "failed"] = "ok"
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0"
    document_id: str
    filename: str
    file_type: str
    source_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    pages: list[Page] = Field(default_factory=list)
    elements: list[Element] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_references(self):
        ids = {element.element_id for element in self.elements}
        if len(ids) != len(self.elements):
            raise ValueError("Element IDs must be unique")
        page_numbers = {page.page_number for page in self.pages}
        if len(page_numbers) != len(self.pages):
            raise ValueError("Page numbers must be unique")
        for element in self.elements:
            if element.parent_id and element.parent_id not in ids:
                raise ValueError(f"Missing parent: {element.parent_id}")
            if page_numbers and element.page_number is not None and element.page_number not in page_numbers:
                raise ValueError(f"Missing page: {element.page_number}")
        return self
