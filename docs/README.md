# RAG Document Parser

Pipeline Python đọc PDF, DOCX, XLSX và ảnh, chuẩn hóa thành `CanonicalDocument`, xuất JSON,
Markdown và asset. Project có lớp chuẩn bị element cho retrieval nhưng chưa triển khai chunking,
embedding, vector database hoặc truy vấn.

## Cấu hình và chạy

Chỉ chỉnh [config.yaml](../src/document_parser/config.yaml):

```yaml
device: gpu:0
models:
  layout: PP-DocLayout_plus-L
  detection: PP-OCRv6_medium_det
  recognition: tieubaoca/pp-ocrv6-medium-rec-vietnamese
paths:
  input: output/pdf
  output: parsed_test_document/ppocrv6_vi_medium_clean
ocr:
  dpi: 150
  options:
    text_recognition_batch_size: 1
    text_det_limit_side_len: 1280
    use_table_recognition: true
    precision: fp32
```

- `device`: `cpu` hoặc `gpu`. `gpu` dùng GPU đầu tiên; `gpu:1` chọn GPU thứ hai nếu cần.
- `layout`: model phát hiện bố cục trang.
- `detection`: model phát hiện vùng chữ.
- `recognition`: model nhận dạng chữ. Mặc định dùng `PP-OCRv6_medium_rec` đã fine-tune tiếng Việt từ `tieubaoca/pp-ocrv6-medium-rec-vietnamese`, nạp trọng số và bộ ký tự trong `paths.recognition_model`.

CPU và GPU dùng **cùng các model đã chọn**. Config không gắn với tên card. Backend GPU hiện dùng Paddle CUDA; chọn bản Paddle hỗ trợ GPU/driver của máy theo [hướng dẫn chính thức](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html).

File YAML chứa toàn bộ tham số cấu hình của parser: `paths` chọn input/output và model tùy chỉnh; `ocr.dpi` chọn DPI render PDF; `ocr.options` truyền trực tiếp các tham số khởi tạo `PPStructureV3` như batch, precision, CPU threads, ngưỡng detection/recognition, layout và các module bật/tắt. Có thể thêm các tham số PPStructureV3 khác trong `ocr.options`; khai báo device và tên ba model trong `device`/`models` để tránh trùng lặp.

## Kiến trúc OCR và table

OCR được tách thành năm thành phần trong `document_parser.ocr`: `LayoutDetector`, `OCRDetector`, `OCRRecognizer`, `TableRecognizer` và `DocumentParser`. `PPStructureV3` chịu trách nhiệm layout, reading order và cấu trúc bảng. Model trong `models.recognition` chỉ nhận ảnh dòng chữ/cell đã detect; nó không xác định table, row hay column.

Với block `table`, parser đọc `table_res_list` (`cell_box_list`, `pred_html`, `table_ocr_pred`), gắn text recognition về đúng cell và lưu `rows`, `columns`, `cells`, bbox, confidence, `rowspan`, `colspan`, Markdown và HTML. Bảng chữ nhật đơn giản được xuất Markdown; bảng có merged/nested cell giữ HTML để không mất cấu trúc. `document.json` giữ danh sách canonical `elements` để tương thích và thêm `pages[].blocks[]` làm representation theo trang cho chunking/RAG.

`pdf` chứa ngưỡng phân loại PDF/quality, `max_excel_region_cells` giới hạn số ô Excel, `environment` chứa đường dẫn cache và biến môi trường Paddle. `recognition_download` chứa repository, revision, danh sách file và số luồng tải model tùy chỉnh; chỉ được dùng khi `models.recognition` chọn repository đó. Tham số chạy không tự đổi khi chuyển CPU/GPU; chỉnh giá trị trong YAML theo thiết bị nếu cần.

Config hiện đọc `output/pdf/` (chứa `hp_first_5_pages.pdf`) và ghi kết quả vào `parsed_test_document/ppocrv6_vi_medium_clean/`. Đổi `paths.input`/`paths.output` trong YAML để xử lý thư mục khác, rồi chạy:

```bash
python scripts/parse_test_documents.py
```

Đổi thiết bị hoặc model cho một lần chạy:

```bash
python scripts/parse_test_documents.py --device gpu
python scripts/parse_test_documents.py --device cpu
python scripts/parse_test_documents.py --device gpu \
  --layout-model PP-DocLayout-S --detection-model PP-OCRv6_tiny_det
```

Có thể đổi input/output hoặc dùng file config riêng:

```bash
python scripts/parse_test_documents.py --input /path/documents --output /path/results
python scripts/parse_test_documents.py --config /path/config.yaml
```

File config riêng hỗ trợ cùng các mục với config chính; mục bị bỏ qua lấy giá trị mặc định từ config chính. Các đường dẫn trong YAML tính từ gốc repository; với config riêng tính từ thư mục chứa config. Khi cài wheel, các đường dẫn mặc định tính từ working directory. Đường dẫn truyền trực tiếp qua CLI tính từ working directory.

## Cài đặt

Python 3.11–3.13; khuyến nghị Python 3.12. Chạy các lệnh tại gốc repository. Cần internet khi cài dependency và tải model lần đầu.

### CPU

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[cpu]'
python scripts/download_ocr_models.py
python scripts/parse_test_documents.py --device cpu
```

`requirements.txt` là file dependency chung duy nhất. `pyproject.toml` đọc trực tiếp file này; extra `cpu` hoặc `gpu` chọn backend Paddle. Trên PowerShell kích hoạt bằng `.venv\Scripts\Activate.ps1`.

Nếu Debian/Ubuntu thiếu `ensurepip`:

```bash
python3.12 -m venv --without-pip .venv
python -m pip --python .venv/bin/python install pip
source .venv/bin/activate
python -m pip install -e '.[cpu]'
```

### GPU

Dùng môi trường riêng; `paddlepaddle` và `paddlepaddle-gpu` cùng cung cấp module `paddle`. Cài wheel GPU phù hợp theo [hướng dẫn Paddle](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html), rồi cài project với extra `gpu`.

Ví dụ môi trường CUDA 11.8 đã dùng cho MX230:

```bash
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install paddlepaddle-gpu==3.3.0 \
  --index-url https://www.paddlepaddle.org.cn/packages/stable/cu118/
python -m pip install -e '.[gpu]'
python scripts/download_ocr_models.py
python scripts/check_gpu.py
python scripts/parse_test_documents.py --device gpu
```

Parser báo lỗi khi yêu cầu GPU nhưng môi trường không có Paddle CUDA hoặc không thấy thiết bị; không tự chuyển sang CPU.

## Model và Python API

```bash
python scripts/download_ocr_models.py
```

Model Paddle chính thức được tải tự động trong lần OCR đầu tiên. Script trên chỉ tải model tùy chỉnh khi `models.recognition` khớp `recognition_download.repository`, dùng revision trong YAML và ghi SHA256 vào `model_manifest.json`. Các model và cache nằm trong `.cache/`, đã bỏ qua bởi Git.

```python
from document_parser import DocumentPipeline, ProjectConfig

config = ProjectConfig.load().parser_config(device="gpu")
summary = DocumentPipeline(config).parse_all()
print(summary["counts"])
```

`DocumentPipeline()` tự đọc cấu hình mặc định. API có thể ghi đè model bằng `parser_config(layout_model=..., detection_model=..., recognition_model=...)`.

## Benchmark và output

```bash
python scripts/benchmark_ocr.py --device gpu \
  --input documents/hp.pdf --pages 1 15 31 \
  --output parsed_test_document/benchmark.json
```

Benchmark hỗ trợ cùng các tùy chọn thiết bị/model với parser. Trang đầu bao gồm thời gian khởi tạo model; khi chạy GPU có lấy mẫu VRAM.

```text
parsed_test_document/
  summary.json
  tai_lieu/
    document.json
    document.md
    assets/
```

`document.json` chứa nội dung và metadata parser/config/quality. `document.md` dùng để đọc kiểm tra. `assets/` giữ ảnh gốc, ảnh trích xuất và trang render. `summary.json` ghi kết quả batch và lỗi từng file/trang.

### Hậu xử lý OCR bằng code

`line_match_ratio: 1.0` mặc định chỉ cho ghép dòng khi toàn bộ ký tự khớp sau khi bỏ khoảng trắng; nội dung khác được giữ nguyên để tránh mất từ. `line_assignment_ratio` là ngưỡng diện tích dòng nằm trong block, còn `retry_crop_padding` là phần lề crop tính bằng pixel.

Các tùy chọn nằm trong `postprocessing` của config chính. Pipeline ghép lại block từ các dòng OCR có tọa độ để giữ khoảng trắng giữa dòng, loại dòng trùng có cùng text và vùng ảnh, chuẩn hóa Unicode/khoảng trắng/dấu câu, đánh dấu số trang ở lề và sửa thứ tự đọc trên trang một cột. `reading_order: auto` giữ thứ tự layout khi phát hiện hai cột; dùng `preserve` để giữ thứ tự cho tài liệu có bố cục phức tạp. `enabled: false` tắt bước hậu xử lý.

JSON giữ `ocr_lines` (text, bbox pixel, confidence), `text_original` và `text_changes` cho những block đã sửa. Số trang vẫn nằm trong JSON với `exclude_from_content: true`, được bỏ khỏi Markdown; bước tạo chunk RAG cũng cần bỏ các element có cờ này. Các dòng trùng được ghi vào `duplicate_sources` hoặc `duplicate_ocr_lines`.

Lỗi năm tách rời và chữ lặp có ứng viên trong `spelling_vocabulary` chỉ được gợi ý trong `ocr_review`, không tự thay. Với số Điều/Chương lệch chuỗi, `retry_numbering` OCR lại crop từ ảnh nguồn ở các `retry_scales`; chỉ sửa khi có đủ `retry_min_agreement` lần đọc ở các scale khác nhau đồng ý với số dự kiến và đạt `retry_min_confidence`. Crop dùng cùng model/device trong config. Đây là kiểm tra nhất quán OCR, không bảo đảm tuyệt đối độ chính xác. Các trường hợp chưa chắc giữ nguyên text và có `status: needs_review`; không gọi LLM.

`document.metadata.ocr_postprocessing` thống kê số block đã thay đổi, số trang bị loại khỏi nội dung và số vấn đề cần kiểm tra. Quality `ok` vẫn chỉ nói pipeline thành công, không thay cho đo CER/WER.

### Sửa text OCR bằng model

`text_correction` chọn một trong bốn backend `protonx`, `protonx_legal`, `bmd1905` hoặc
`bravend`; CLI có thể ghi đè bằng `--corrector`. Model được load lazy và chỉ nhận element OCR có
dấu hiệu lỗi. Candidate phải qua safety gate bảo toàn số, ngày, mã định danh, URL/email và cấu trúc
pháp lý; candidate không an toàn chỉ được lưu để review, không thay text.

### Chuẩn bị retrieval

`RetrievalPreprocessor` tạo bản dữ liệu mới, bỏ footer/header/image và element có
`exclude_from_content`, giữ source ID/page/bbox/section, đồng thời thêm `heading_path` vào
`text_for_embedding`. Lớp này không mutate `CanonicalDocument` và chưa thực hiện chunking,
embedding hay truy vấn vector.

Output phải nằm ngoài thư mục input. CLI trả `0` khi mọi file thành công; trả `1` khi input rỗng, có lỗi hoặc có tài liệu partial. Lỗi một trang không dừng cả batch.

## Cấu trúc và kiểm tra

```text
docs/                       Hướng dẫn và báo cáo
documents/                  Tài liệu đầu vào
requirements.txt            Dependency chung
pyproject.toml              Đóng gói và cấu hình công cụ
scripts/                    Parse, tải model, benchmark, kiểm tra GPU
src/document_parser/
  config.yaml               Chọn thiết bị và model
  config.py                 Bộ nạp YAML và phân giải đường dẫn
  pipeline.py               Điều phối xử lý
  ocr/                      Layout, detection, recognition và table
  correction/               Candidate correction và safety gate
  retrieval/                Schema và tiền xử lý retrieval
  loader/                   Kiểm tra file
  router/                   Routing và PDF analyzer
  parsers/                  Adapter PDF, DOCX, XLSX, ảnh
  normalization/            Schema chuẩn và normalizer
  quality/                  Kiểm tra chất lượng
  utils/                    File và serialization
tests/                      Unit và integration tests
```

```bash
python -m pytest -q -m 'not integration'
ruff check src scripts tests
ruff format --check src scripts tests
```

Test dùng tài liệu/model thật chạy riêng bằng `python -m pytest -q -m integration`.

Mọi trang PDF và mọi ảnh đều chạy PP-StructureV3/PaddleOCR; PDF không còn nhánh đọc text/table native. PyMuPDF chỉ rasterize trang PDF thành PNG trước khi OCR. DOCX dùng Docling Word backend; XLSX dùng openpyxl. Table recognition được bật và model recognition tiếng Việt chỉ đọc nội dung text/cell.

Reading order/heading PDF được lấy từ layout rồi hậu xử lý hình học; DOCX thường không có pagination/bbox thật; openpyxl không tính lại công thức. Quality `ok` kiểm tra cấu trúc/text cơ bản, chưa đo CER/WER. Báo cáo lịch sử nằm trong [PARSE_REPORT.md](PARSE_REPORT.md).

Nếu OCR không khởi tạo được, kiểm tra backend Paddle và chạy script tải model. Khi CLI trả `1`, xem `summary.json`, `failed_pages` và lỗi trong metadata.
