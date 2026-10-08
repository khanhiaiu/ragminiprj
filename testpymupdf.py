from docling.document_converter import DocumentConverter

PDF_PATH = "documents/VanBanGoc_Thông tư 65-2025-TT-NHNN.pdf"

converter = DocumentConverter()
result = converter.convert(PDF_PATH)

markdown = result.document.export_to_markdown()

print(markdown[:10000])

with open("docling_default.md", "w", encoding="utf-8") as f:
    f.write(markdown)