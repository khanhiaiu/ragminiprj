# Báo cáo chạy document parsing pipeline

Đã parse toàn bộ 4 file trong `documents/` trên Python 3.12, Linux CPU. Output ở `parsed_test_document/`: JSON chuẩn, Markdown để đọc và assets. Batch exit 0: **4 ok, 0 partial, 0 failed**; 75 trang PDF được giữ đủ. File nguồn không thay đổi, đã đối chiếu SHA256 trước/sau.

## Cấu trúc và dependencies

`src/document_parser/` gồm loader, router/PDFAnalyzer, parsers theo format, normalization/schema, quality và utils; `pipeline.py` điều phối. Có scripts CLI/tải model, cấu hình OCR, tests, requirements/constraints và README với Mermaid.

| Dependency | Version đã chạy |
| --- | --- |
| pymupdf | 1.28.2 |
| paddleocr | 3.7.0 |
| paddlex | 3.7.2 |
| paddlepaddle | 3.3.0 |
| docling-slim | 2.133.0 |
| docling-core | 2.99.0 |
| openpyxl | 3.1.5 |
| pydantic | 2.13.5 |
| pillow | 12.3.0 |
| pytest | 9.1.1 |

Docling dùng Word backend chính thức qua `docling-slim[format-docx]`. Không cài framework RAG; không có embedding/retrieval.

## PDF routing và kết quả từng file

Analyzer chạy từng trang, kết hợp text/block count, printable/invalid ratio, density và diện tích hợp các ảnh. Digital dùng PyMuPDF; scanned render 150 DPI rồi PPStructureV3; hybrid dùng native text + assets, đánh dấu enrichment pending. Quality không đạt kích hoạt OCR fallback từng trang. Lỗi một trang được ghi metadata và giữ các trang khác.

| File | Parser / routing | Elements | Assets tham chiếu | Output |
| --- | --- | ---: | ---: | --- |
| 2602.22678v1.pdf | PyMuPDF; 38 digital, 4 hybrid | 767 | 109 | [2602.22678v1/](parsed_test_document/2602.22678v1/document.json) |
| Cam kết CTV_Nguyễn Trọng Khánh.pdf | PyMuPDF; 2 digital | 38 | 0 | [Cam kết CTV_Nguyễn Trọng Khánh/](parsed_test_document/Cam%20kết%20CTV_Nguyễn%20Trọng%20Khánh/document.json) |
| hp.pdf | PaddleOCR PPStructureV3; 31 scanned | 551 | 32 | [hp/](parsed_test_document/hp/document.json) |
| rag_system_docs.docx | Docling Word backend; structural order | 58 | 0 | [rag_system_docs/](parsed_test_document/rag_system_docs/document.json) |

`2602.22678v1.pdf` giữ 15 native tables và 109 image assets. DOCX giữ 12 headings, 19 list items và 3 tables; không giả lập số trang Word. Không có XLSX hoặc ảnh standalone trong input gốc; các nhánh này được kiểm tra bằng fixture và backend OCR giả trong tests.

## Kiểm tra thực tế

- **36 tests passed**: routing, analyzer, schema/IDs, normalizer, PDF fallback/cách ly lỗi, Excel regions/formulas/merges, DOCX structures/images, OCR adapter và batch failures.
- `pip check`: No broken requirements found.
- Toàn bộ JSON validate qua CanonicalDocument; đối chiếu số trang với PDF gốc, element IDs/order/parent refs, các asset references và SHA256 nguồn.
- Đối chiếu ảnh/text mẫu ở trang 1, 15 và 31 của hp.pdf. Layout bỏ sót một số dòng dù OCR toàn trang nhận được; adapter hiện giữ thêm các dòng chưa được layout chứa, loại trùng theo text và bbox.
- `hp.pdf` giữ 138 dòng bổ sung với `layout_fallback=true`; confidence và tọa độ được lưu trong metadata. Không tự sửa nội dung từ kiến thức bên ngoài.

## Giới hạn

OCR vẫn sai dấu/chữ, đôi khi lặp ký tự; chưa đo CER/WER và chưa xác nhận mọi dòng đều đúng. Quality `ok` chỉ là kiểm tra cấu trúc/text cơ bản. Reading order nhiều cột và native heading/table detection là heuristic. Hybrid chưa selective OCR; DOCX không có pagination/bbox khi backend không cung cấp; openpyxl không tính lại công thức.

Profile tiếng Việt dùng PP-DocLayout-S, detection mobile v5 và [recognition tiếng Việt của tieubaoca](https://huggingface.co/tieubaoca/pp-ocrv6-medium-rec-vietnamese/tree/edc19ca6890bedb7a68c055d21190b44fc1aaa00), pin revision cùng SHA256 model files. Profile tắt OCR table/formula recognition vì scan test là văn bản; native PDF/XLSX table extraction vẫn hoạt động. Default PPStructureV3 adapter hỗ trợ table HTML khi bật table recognition. Recognition bật MKLDNN riêng; layout tắt do lỗi OneDNN/PIR trên máy test. Các lựa chọn được ghi trong JSON parser_config và README.

## Chạy lại

```bash
source .venv/bin/activate
python scripts/download_ocr_models.py
python scripts/parse_test_documents.py --ocr-options config/ocr_scan_vi.json
python -m pytest -q
```

Model đã được cache trên máy này; lệnh tải model dùng lại cache. Cài mới theo README/requirements.txt. Xem [summary.json](parsed_test_document/summary.json) và [pipeline.log](parsed_test_document/pipeline.log) để kiểm tra trạng thái từng file/trang.
