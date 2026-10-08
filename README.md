# RAG Document Parser

Pipeline Python dùng để đọc tài liệu `PDF`, `DOCX`, `XLSX` và ảnh, sau đó chuẩn hóa nội dung và xuất ra JSON, Markdown cùng các asset liên quan.

> Project tập trung vào ingestion/parsing và có lớp chuẩn bị dữ liệu retrieval; chưa triển khai
> chunking, embedding, vector database hoặc truy vấn.

## Yêu cầu

- Python `3.11` đến `3.13` (khuyến nghị `3.12`)
- Internet ở lần cài dependency và tải model đầu tiên
- Chọn **một** backend Paddle: CPU hoặc GPU

Các lệnh bên dưới được chạy tại thư mục gốc của project (`ragminiprj`).

## Chạy nhanh bằng CPU

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e '.[cpu]'
python scripts/download_ocr_models.py

python scripts/parse_test_documents.py \
  --device cpu \
  --input documents \
  --output parsed_test_document/cpu
```

Trên Windows PowerShell, kích hoạt môi trường bằng:

```powershell
.venv\Scripts\Activate.ps1
```

## Chạy bằng GPU

Tạo một virtual environment riêng và cài bản `paddlepaddle-gpu` phù hợp với CUDA/driver của máy. Ví dụ với CUDA 11.8:

```bash
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate

python -m pip install --upgrade pip
python -m pip install paddlepaddle-gpu==3.3.0 \
  --index-url https://www.paddlepaddle.org.cn/packages/stable/cu118/
python -m pip install -e '.[gpu]'
python scripts/download_ocr_models.py

python scripts/check_gpu.py
python scripts/parse_test_documents.py \
  --device gpu \
  --input documents \
  --output parsed_test_document/gpu
```

Nếu dùng phiên bản CUDA khác, chọn wheel Paddle tương ứng theo [hướng dẫn cài đặt PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html). Parser sẽ báo lỗi nếu GPU không khả dụng, không tự động chuyển sang CPU.

## Cấu hình

File cấu hình mặc định nằm tại [`src/document_parser/config.yaml`](src/document_parser/config.yaml). Các mục thường cần chỉnh:

```yaml
device: gpu:0 # cpu, gpu hoặc gpu:N

paths:
  input: documents
  output: parsed_test_document/results

ocr:
  dpi: 150
```

Có thể không sửa file cấu hình mà ghi đè trực tiếp khi chạy:

```bash
python scripts/parse_test_documents.py \
  --device cpu \
  --input /path/to/documents \
  --output /path/to/results
```

Hoặc dùng một file YAML riêng:

```bash
python scripts/parse_test_documents.py --config /path/to/config.yaml
```

Các tùy chọn CLI chính:

```text
--device cpu|gpu|gpu:N
--input PATH
--output PATH
--config PATH
--layout-model MODEL
--detection-model MODEL
--recognition-model MODEL
--corrector protonx|protonx_legal|bmd1905|bravend
```

Thư mục output phải nằm ngoài thư mục input. Lệnh trả exit code `1` nếu không tìm thấy tài liệu hỗ trợ, có file lỗi hoặc có tài liệu chỉ được xử lý một phần.

## Kết quả

Mỗi tài liệu được ghi vào một thư mục riêng:

```text
parsed_test_document/
  summary.json
  ten_tai_lieu/
    document.json
    document.md
    assets/
```

- `summary.json`: tổng hợp trạng thái của cả batch và lỗi theo file
- `document.json`: nội dung chuẩn hóa, metadata, quality và thông tin OCR
- `document.md`: bản dễ đọc để kiểm tra kết quả
- `assets/`: ảnh gốc, ảnh trích xuất hoặc trang PDF đã render

## Dùng Python API

```python
from document_parser import DocumentPipeline, ProjectConfig

config = ProjectConfig.load().parser_config(device="cpu")
summary = DocumentPipeline(config).parse_all("documents", "parsed_test_document/api")
print(summary["counts"])
```

Chuẩn bị element sạch và có context heading cho bước chunk/embed sau này:

```python
from document_parser import RetrievalPreprocessor

document = DocumentPipeline(config).parse(
    "documents/report.pdf", "parsed_test_document/report"
)
retrieval_document = RetrievalPreprocessor().process(document)
```

## Kiểm tra project

```bash
python -m pytest -q -m 'not integration'
ruff check src scripts tests
ruff format --check src scripts tests
```

Test dùng tài liệu/model thật chạy riêng bằng `python -m pytest -q -m integration`.

Benchmark OCR một số trang PDF:

```bash
python scripts/benchmark_ocr.py \
  --device gpu \
  --input documents/hp.pdf \
  --pages 1 15 31 \
  --output parsed_test_document/benchmark.json
```

Tài liệu chi tiết về model, hậu xử lý OCR, cấu trúc output và các giới hạn hiện tại nằm tại [`docs/README.md`](docs/README.md). Báo cáo parse gần nhất nằm tại [`docs/PARSE_REPORT.md`](docs/PARSE_REPORT.md).

## Xử lý lỗi thường gặp

- **Không tìm thấy tài liệu:** kiểm tra `--input`; chỉ các định dạng PDF, DOCX, XLSX và ảnh được hỗ trợ.
- **Không tải được model:** kiểm tra kết nối mạng rồi chạy lại `python scripts/download_ocr_models.py`.
- **GPU verification failed:** kiểm tra wheel Paddle, CUDA và driver bằng `python scripts/check_gpu.py`.
- **CLI trả mã lỗi `1`:** mở `summary.json`, sau đó kiểm tra `failed_pages` và lỗi trong metadata của tài liệu.
