# Đặc tả kỹ thuật toàn bộ source code — RAG Document Parser

## 1. Thông tin tài liệu

| Thuộc tính | Giá trị |
|---|---|
| Tên hệ thống | `rag-document-parser` |
| Phiên bản package | `0.1.0` |
| Phạm vi khảo sát | `src/`, `scripts/`, `tests/`, cấu hình đóng gói và runtime |
| Ngôn ngữ triển khai | Python 3.11–3.13 |
| Mục tiêu | Ingestion, chuẩn hóa và chuẩn bị dữ liệu đầu vào cho retrieval |
| Ngoài phạm vi | Thuật toán chunking, embedding model, vector database, truy vấn, reranking và sinh câu trả lời |
| Nguồn sự thật | Hành vi trong source code hiện tại; test được dùng để xác nhận ý đồ thiết kế |

Tài liệu này mô tả hệ thống ở mức kiến trúc, hợp đồng dữ liệu, luồng xử lý, thuật toán, API, CLI, cấu hình, lỗi, kiểm thử và giới hạn. Những nội dung ghi là “hiện tại” phản ánh snapshot source code tại thời điểm lập tài liệu.

## 2. Mục tiêu và yêu cầu nghiệp vụ

Hệ thống nhận một file hoặc một thư mục tài liệu, chọn parser theo phần mở rộng, trích xuất nội dung và chuẩn hóa thành một schema duy nhất là `CanonicalDocument`. Kết quả được lưu dưới ba dạng:

- JSON có cấu trúc đầy đủ để máy xử lý;
- Markdown để con người kiểm tra nhanh;
- asset nhị phân như ảnh gốc, ảnh trích từ PDF/DOCX và ảnh trang PDF render cho OCR.

Các yêu cầu hành vi chính:

1. Hỗ trợ PDF, DOCX, XLSX, PNG, JPG và JPEG.
2. Mọi trang PDF đều được render và xử lý bằng PP-StructureV3/PaddleOCR, không đọc text layer native.
3. `PDFAnalyzer` vẫn tồn tại như utility phân tích/quality độc lập, không tham gia routing PDF hiện tại.
4. Lỗi ở một trang PDF không làm mất kết quả của các trang còn lại.
5. Lỗi ở một file trong batch không dừng các file khác.
6. Mọi backend phải trả về mô hình dữ liệu chuẩn, JSON-serializable.
7. Không ghi output vào chính thư mục input hoặc thư mục con của input.
8. Không âm thầm chuyển GPU về CPU khi người dùng đã yêu cầu GPU.
9. Hậu xử lý OCR phải bảo thủ: giữ text gốc và lịch sử sửa; các sửa không chắc chắn chỉ được gắn cờ review.
10. Source document không được chỉnh sửa trong quá trình parse.

## 3. Kiến trúc tổng thể

```text
CLI / Python API
       |
       v
ProjectConfig -> ParserConfig
       |
       v
DocumentPipeline
  |-- FileLoader: kiểm tra/discover file
  |-- DocumentRouter: chọn adapter theo extension
  |     |-- PDFParser
  |     |     `-- PaddlePDFParser -> PaddleEngine
  |     |-- DoclingDOCXParser
  |     |-- ExcelParser
  |     `-- PaddleImageParser -> PaddleEngine
  |-- Normalizer -> CanonicalDocument (Pydantic)
  |-- OCR postprocessing/review
  |-- Candidate-based text correction + safety gate
  |-- ParseQualityValidator
  |-- Serialization -> JSON / Markdown / assets / summary
  `-- RetrievalPreprocessor (API tùy chọn, sau parsing)
```

### 3.1 Nguyên tắc thiết kế

- **Adapter độc lập định dạng:** từng parser chuyển dữ liệu thư viện ngoài thành dictionary hoặc `CanonicalDocument`, không để object của thư viện ngoài rò vào schema.
- **Lazy OCR:** `PPStructureV3` chỉ khởi tạo khi thực sự có trang/ảnh cần OCR.
- **Page-level fault isolation:** PDF bắt lỗi trong vòng lặp từng trang.
- **Canonical boundary:** `Normalizer` và model Pydantic là ranh giới kiểm tra dữ liệu.
- **Auditability:** metadata lưu parser, model config, bbox, confidence, text gốc, thay đổi và cảnh báo.
- **Atomic metadata/text output:** JSON, Markdown và summary được ghi qua file tạm rồi `os.replace`.

## 4. Cấu trúc repository

```text
ragminiprj/
├── pyproject.toml                 # package, Python range, pytest, Ruff
├── requirements.txt              # dependency chung, pin version
├── README.md                      # hướng dẫn chạy nhanh
├── docs/
│   ├── README.md                  # tài liệu vận hành hiện có
│   ├── PARSE_REPORT.md            # báo cáo parse mẫu
│   └── SOURCE_CODE_SPECIFICATION.md
├── scripts/
│   ├── parse_test_documents.py    # CLI parse batch
│   ├── download_ocr_models.py     # tải model recognition tùy chỉnh
│   ├── benchmark_ocr.py           # benchmark trang PDF
│   └── check_gpu.py               # kiểm tra Paddle CUDA
├── src/document_parser/
│   ├── config.py                  # đọc config.yaml chung ở thư mục gốc
│   ├── config.py                  # load/merge/validate/resolve config
│   ├── pipeline.py                # orchestration
│   ├── errors.py                  # exception domain
│   ├── loader/                    # kiểm tra và discover file
│   ├── router/                    # route định dạng và utility phân tích PDF
│   ├── parsers/                   # adapter theo định dạng
│   ├── ocr/                       # layout, detection, recognition, table, orchestration
│   ├── correction/                # detector, model adapters và safety gate
│   ├── retrieval/                 # schema và tiền xử lý retrieval, chưa chunk/embed
│   ├── normalization/             # schema, normalize, OCR postprocess
│   ├── quality/                   # quality gate
│   └── utils/                     # hash, Markdown table, serialization
└── tests/                         # unit/integration tests
```

Các file `__init__.py` vừa định nghĩa package vừa có thể re-export API theo từng namespace;
`document_parser/__init__.py` công khai API cấp cao.

## 5. Hợp đồng dữ liệu chuẩn

### 5.1 `CanonicalDocument`

| Field | Kiểu | Bắt buộc/mặc định | Ý nghĩa |
|---|---|---|---|
| `schema_version` | `str` | `"1.0"` | Phiên bản schema |
| `document_id` | `str` | bắt buộc | 20 ký tự đầu của SHA-256 file nguồn |
| `filename` | `str` | bắt buộc | Tên file gồm extension |
| `file_type` | `str` | bắt buộc | Extension viết thường, không có dấu chấm |
| `source_path` | `str` | bắt buộc | Đường dẫn tuyệt đối của file nguồn |
| `metadata` | `dict[str, Any]` | `{}` | Metadata cấp tài liệu |
| `pages` | `list[Page]` | `[]` | Danh sách trang nếu backend có khái niệm trang |
| `elements` | `list[Element]` | `[]` | Nội dung theo thứ tự đọc |

Ràng buộc:

- không chấp nhận field lạ (`extra="forbid"`);
- `element_id` phải duy nhất;
- `page_number` trong `pages` phải duy nhất;
- `parent_id`, nếu có, phải trỏ tới element tồn tại;
- nếu document có `pages`, mọi `element.page_number` khác `None` phải tồn tại trong danh sách trang;
- toàn bộ object phải serialize được sang JSON.

### 5.2 `Page`

| Field | Kiểu | Mặc định | Ý nghĩa |
|---|---|---|---|
| `page_number` | `int >= 1` | bắt buộc | Số trang 1-based |
| `width`, `height` | `float | None` | `None` | Kích thước trang |
| `page_type` | `digital | scanned | hybrid | None` | `None` | Phân loại trang |
| `status` | `ok | failed` | `ok` | Trạng thái xử lý trang |
| `metadata` | `dict` | `{}` | Analysis, parser, lỗi, asset render... |

### 5.3 `Element`

| Field | Kiểu | Mặc định | Ý nghĩa |
|---|---|---|---|
| `element_id` | `str` | sinh bởi normalizer nếu thiếu | ID ổn định trong document |
| `element_type` | enum | bắt buộc | `heading`, `paragraph`, `list`, `table`, `image`, `formula`, `header`, `footer`, `unknown` |
| `text` | `str` | `""` | Nội dung text/Markdown/HTML tùy element |
| `page_number` | `int | None` | `None` | Trang chứa element |
| `bbox` | tuple 4 số | `None` | `[left, top, right, bottom]` |
| `parent_id` | `str | None` | `None` | Quan hệ cấu trúc cha/con |
| `section_id` | `str | None` | `None` | Section gần nhất do heading mở |
| `order` | `int >= 0` | được normalizer gán lại | Thứ tự toàn document |
| `metadata` | `dict` | `{}` | Provenance và dữ liệu phụ |

Ràng buộc bbox:

- cả bốn tọa độ phải hữu hạn;
- `right >= left`, `bottom >= top`;
- đơn vị nằm trong `metadata.coordinate_unit`, thường là `pt` cho PDF và `px` cho ảnh OCR.

### 5.4 Chuẩn hóa label

`Normalizer` ánh xạ label backend về taxonomy chung. Các ánh xạ quan trọng:

| Label nguồn | Canonical type |
|---|---|
| `text`, `caption`, `reference`, `algorithm`, `code`, `abstract`, `content` | `paragraph` |
| `title`, `doc_title`, `paragraph_title`, `section_header` | `heading` |
| `list_item` | `list` |
| `picture`, `figure`, `chart`, `seal` | `image` |
| `display_formula`, `equation` | `formula` |
| `page_header` | `header` |
| `page_footer`, `page_number`, `number`, `footnote` | `footer` |
| label không biết | `unknown` |

Normalizer đồng thời:

- xóa null byte và trim text;
- gán lại `order` theo vị trí hiện tại;
- sinh `element_id = <document_id>-e<N>` nếu thiếu;
- khi gặp heading, sinh section slug tối đa 80 ký tự và áp dụng section đó cho các element tiếp theo;
- giữ `section_id` có sẵn nếu parser đã cung cấp.

## 6. Cấu hình

### 6.1 Luồng load cấu hình

`ProjectConfig.load(path)` thực hiện:

1. resolve đường dẫn YAML;
2. đọc và validate cấu trúc cấp cao;
3. deep-merge file người dùng lên `_DEFAULTS` lấy từ `config.yaml` ở thư mục gốc;
4. đặt root resolve path:
   - config mặc định: root repository;
   - config tùy chỉnh: thư mục chứa config;
5. `parser_config(...)` tạo immutable `ParserConfig` và áp dụng override CLI/API.

File riêng bắt buộc có `device` và đủ ba khóa `models`; các section lồng khác có thể chỉ khai báo
phần muốn đổi nhờ deep merge. Field lạ cấp cao bị từ chối. Các tên model bắt buộc là chuỗi không rỗng.

### 6.2 Thiết bị và model

| Cấu hình | Hành vi |
|---|---|
| `cpu` | chạy CPU |
| `gpu` | chuẩn hóa thành `gpu:0` |
| `gpu:N` | chọn GPU chỉ số N |
| giá trị khác | `ValueError` |

Ba model nằm tại `models.layout`, `models.detection`, `models.recognition`. Các option tương ứng được tạo tự động cho Paddle:

- `layout_detection_model_name`;
- `text_detection_model_name`;
- `text_recognition_model_name`.

Không được khai báo lại `device` hay ba tên model trong `ocr.options`. Recognition chứa dấu `/` chỉ hợp lệ khi đúng bằng repository tùy chỉnh đã cấu hình. Khi chọn repository tùy chỉnh, engine dùng `recognition_download.model_name` và thêm `text_recognition_model_dir` đã resolve.

### 6.3 Cấu hình PDF

| Key mặc định | Giá trị | Vai trò |
|---|---:|---|
| `minimum_text_chars` | 40 | số ký tự tối thiểu cho text layer “đủ” |
| `minimum_text_blocks` | 1 | số block text tối thiểu |
| `minimum_printable_ratio` | 0.95 | tỷ lệ ký tự printable tối thiểu |
| `max_invalid_char_ratio` | 0.03 | tỷ lệ ký tự lỗi tối đa |
| `minimum_text_quality` | 0.85 | quality tổng hợp tối thiểu |
| `minimum_text_density` | 0.00005 | ký tự/diện tích trang tối thiểu |
| `scan_image_coverage_threshold` | 0.70 | image coverage để coi trang là scan |
| `hybrid_image_coverage_threshold` | 0.08 | image coverage để coi trang là hybrid |

### 6.4 Cấu hình hậu xử lý OCR

| Key | Mặc định | Tác dụng |
|---|---:|---|
| `enabled` | `true` | bật toàn bộ postprocess/review |
| `reconstruct_lines` | `true` | ghép raw OCR line thành block |
| `line_assignment_ratio` | `0.5` | coverage tối thiểu để gán line vào block |
| `line_match_ratio` | `1.0` | similarity tối thiểu trước khi thay block text |
| `line_row_tolerance` | `0.5` | dung sai gom line cùng hàng |
| `duplicate_overlap` | `0.8` | ngưỡng overlap loại duplicate |
| `reading_order` | `auto` | `auto`, `single_column`, `preserve` |
| `column_min_width_ratio` | `0.18` | bề rộng block tối thiểu khi dò cột |
| `column_gap_ratio` | `0.05` | khoảng cách ngang để nhận diện hai cột |
| `column_overlap_ratio` | `0.5` | overlap dọc tối thiểu của hai cột |
| `page_number_margin_ratio` | `0.08` | biên trên/dưới chứa số trang |
| `page_number_center_tolerance` | `0.2` | dung sai vị trí giữa trang |
| `normalize_punctuation` | `true` | chuẩn hóa whitespace/dấu câu |
| `flag_years` | `true` | phát hiện năm bị tách, ví dụ `19 30` |
| `flag_spelling` | `true` | phát hiện lặp ký tự dựa trên vocabulary |
| `retry_numbering` | `true` | OCR lại heading có số thứ tự nghi vấn |
| `retry_scales` | `[1.5, 2.0]` | các scale độc lập khi OCR lại crop |
| `retry_crop_padding` | `3` | padding crop tính bằng pixel |
| `retry_min_confidence` | `0.8` | confidence tối thiểu của reading xác nhận |
| `retry_min_agreement` | `2` | số scale khác nhau phải đồng thuận |

Các ratio phải thuộc `(0, 1]`; scale thuộc `[1, 4]`; số lần agreement không vượt số scale phân biệt.

### 6.5 Sửa lỗi OCR bằng model

`text_correction` cấu hình một backend duy nhất trong `protonx`, `protonx_legal`, `bmd1905`
hoặc `bravend`. Model Transformers được khởi tạo lazy, chỉ khi detector tìm thấy dấu hiệu OCR đáng
ngờ. `only_ocr=true` giới hạn bước này cho element có `metadata.parser=paddleocr`; footer, image,
table và các element bị loại khỏi content không được sửa theo cấu hình mặc định.

Detector coi confidence thấp là tín hiệu hỗ trợ, không đủ để tự gọi model. Inference chỉ được cân
nhắc khi có pattern OCR tiếng Việt đã biết, ký tự gốc lặp bất thường, spelling flag từ bước review
hoặc lỗi dấu câu. Candidate sau inference phải qua safety gate: giữ nguyên số, ngày, mã định danh,
URL/email, marker pháp lý và cấu trúc; không vượt `max_auto_edit_ratio`; và chỉ sửa đúng token đã
được detector cho phép. Candidate bị từ chối được lưu để review, không thay text.

### 6.6 Biến môi trường

`ParserConfig.apply_environment()` dùng `os.environ.setdefault`, vì vậy biến đã do caller đặt luôn được ưu tiên. Các cache path quen thuộc được resolve thành đường dẫn tuyệt đối. Cấu hình mặc định còn tắt device fallback của Paddle nhằm tránh đổi thiết bị ngoài ý muốn.

## 7. Luồng xử lý cấp cao

### 7.1 Parse một file

`DocumentPipeline.parse(path, output_dir=None)`:

1. `FileLoader.load` resolve và kiểm tra file.
2. Xác định output: tham số trực tiếp hoặc `<config.output_dir>/<stem>`.
3. Từ chối output bằng thư mục nguồn hoặc nằm dưới thư mục nguồn.
4. Tạo router với factory parser; parser chỉ được khởi tạo khi route đến.
5. Parser tương ứng trả `CanonicalDocument` hoặc dữ liệu đã normalized.
6. Normalizer chạy lại để đảm bảo schema chung.
7. `review_document` rà soát OCR.
8. `TextCorrectionService` chỉ gửi các candidate đáng ngờ tới backend đã chọn rồi áp dụng safety gate.
9. Nếu postprocessing bật, section ID của element PaddleOCR bị xóa và normalizer chạy lại để section phản ánh thứ tự sau hậu xử lý/correction.
10. Gắn toàn bộ `ParserConfig` vào `document.metadata.parser_config`.
11. Chạy quality validator và gắn `document.metadata.quality`.
12. Ghi `document.json`, `document.md` và bảo đảm có thư mục `assets/`.
13. Trả về `CanonicalDocument`.

### 7.2 Parse batch

`DocumentPipeline.parse_all(input_dir=None, output_dir=None)`:

1. Discover đệ quy mọi file hỗ trợ.
2. Từ chối output bằng hoặc nằm trong input root.
3. Phát hiện trùng stem không phân biệt hoa/thường.
4. Tên output thông thường là stem; nếu trùng thì dùng `<stem>-<ext>-<8 ký tự hash file>`.
5. Nếu tên vẫn va chạm, nối thêm 8 ký tự SHA-256 của đường dẫn.
6. Parse từng file tuần tự.
7. Exception ở một file được bắt, log, ghi `summary.json` cục bộ cho file lỗi và tiếp tục.
8. Ghi summary batch với count `ok`, `partial`, `failed`.

Cấu trúc record thành công:

```json
{
  "filename": "example.pdf",
  "source_path": "/abs/example.pdf",
  "output_dir": "/abs/results/example",
  "status": "ok",
  "parser": "paddleocr",
  "quality": {},
  "page_parsers": [
    {"page": 1, "type": "scanned", "parser": "paddleocr", "status": "ok"}
  ]
}
```

## 8. Loader và router

### 8.1 `FileLoader`

`load` báo lỗi nếu đường dẫn không phải file, extension không hỗ trợ hoặc file rỗng. `discover` yêu cầu input là directory và dùng `rglob("*")`; file không hỗ trợ bị bỏ qua, không được ghi là failed.

Extension hợp lệ: `.pdf`, `.docx`, `.xlsx`, `.png`, `.jpg`, `.jpeg`, không phân biệt hoa thường.

### 8.2 `DocumentRouter`

Router ánh xạ extension sang tên adapter `pdf`, `docx`, `excel`, `image`. `route` gọi factory tương ứng; thiếu mapping extension hoặc thiếu factory đều ném `UnsupportedFormatError`.

## 9. Đặc tả xử lý PDF

### 9.1 Utility phân loại trang (`PDFAnalyzer`)

Analyzer lấy:

- text blocks từ `page.get_text("blocks")`;
- ảnh từ `page.get_image_info()`;
- union area của các bbox ảnh đã clip trong trang;
- page area, text length, block count và metric chất lượng ký tự.

`text_metrics(text)` trả ba giá trị:

```text
printable_ratio = printable_character_count / text_length
invalid_ratio   = invalid_character_count / text_length
quality         = max(0, printable_ratio * (1 - invalid_ratio))
```

Ký tự invalid gồm replacement character `U+FFFD` và Unicode category `Cc/Cs/Co/Cn`, ngoại trừ newline, carriage return và tab. Với text rỗng, hàm trả `(1.0, 0.0, 1.0)`.

Quy tắc phân loại:

1. Không text và không ảnh: `digital`, `blank=true`.
2. Text không đạt đầy đủ threshold và đồng thời không phải sparse text hợp lệ hoặc ảnh phủ lớn hơn ngưỡng scan: `scanned`.
3. Có ảnh và coverage lớn hơn ngưỡng hybrid: `hybrid`.
4. Còn lại: `digital`.

Sparse title page có ít text nhưng ký tự tốt vẫn được utility phân loại là digital, trừ khi bị ảnh lớn chi phối. Union area dùng sweep theo trục X nên ảnh chồng nhau không bị đếm diện tích hai lần. Kết quả này hiện chỉ phục vụ phân tích/test; `PDFParser` không dùng nó để chọn backend.

### 9.2 Điều phối PDF (`PDFParser`)

Mọi trang PDF luôn đi qua OCR, không phân nhánh theo text layer:

```text
PDF page -> render PNG -> PP-StructureV3 -> OCR/table recognition
```

- `Page.page_type` luôn là `scanned` và `metadata.parser=paddleocr`.
- `metadata.ocr_forced=true`; không có native extraction hay OCR fallback.
- Output từng trang được validate tại page boundary; lỗi chỉ làm hỏng trang liên quan.
- `failed_pages` được tổng hợp ở document metadata.
- PyMuPDF chỉ mở PDF và rasterize trang; không đọc text, bảng hoặc ảnh nhúng.

### 9.3 OCR PDF (`PaddlePDFParser`)

Trang được render PNG ở `ocr_dpi` vào `assets/pNNNN_render.png`. Paddle hoạt động trên pixel; sau khi parse, bbox được scale ngược về PDF point:

```text
scale_x = PDF_page_width / rendered_image_width
scale_y = PDF_page_height / rendered_image_height
```

Mỗi element nhận page number, `coordinate_unit=pt`, rendered asset và DPI. Pipeline chính gọi parser này cho mọi trang PDF.

## 10. Đặc tả OCR ảnh và Paddle

OCR hiện được tách trong `document_parser/ocr/`:

- `LayoutDetector`: khởi tạo/chạy PP-StructureV3 và chuẩn hóa layout block cùng reading order;
- `OCRDetector`: chỉ trả geometry và detector confidence, không chứa text;
- `OCRRecognizer`: dùng model recognition trong config cho ảnh text/cell đã crop hoặc output recognition đã detect;
- `TableRecognizer`: đọc `table_res_list`, parse HTML structure, map OCR vào cell và tạo Markdown/HTML;
- `DocumentParser`: route block và assemble element. `PaddleEngine` chỉ là alias tương thích của orchestrator này.

`models.recognition` không được dùng để dự đoán table/row/column. Cấu trúc đó thuộc PP-StructureV3 table recognition.

### 10.1 Khởi tạo engine

`PaddleEngine` giữ một backend dùng chung trong một pipeline. Khi `predict` lần đầu:

1. áp dụng environment;
2. nếu device là GPU, import Paddle và chạy preflight;
3. import và tạo `PPStructureV3(**ocr_options)`;
4. cache backend cho các lần sau.

Nếu khởi tạo lỗi, thông báo lỗi được cache trong `_initialization_error`; các lần gọi sau ném lại `DependencyUnavailableError` mà không thử khởi tạo lại trong cùng instance.

GPU preflight xác nhận Paddle build có CUDA, parse index và kiểm tra index nằm trong số GPU nhìn thấy. `gpu:-1`, index không phải số, index âm hoặc vượt giới hạn đều thất bại.

### 10.2 Chuyển kết quả PPStructureV3

Mỗi result có thể cung cấp `.json` là property, callable hoặc JSON string. Payload phải có `parsing_res_list`; nếu không, ném `DocumentParseError`.

Mỗi block tạo element với:

- label, content và `block_bbox`;
- parser `paddleocr`;
- original label, coordinate unit pixel, block ID/order;
- `table_html` nếu là bảng;
- crop asset nếu là image/figure/chart/seal và có `assets_dir`.

Sau đó engine:

1. gắn raw OCR lines vào layout block;
2. khôi phục OCR lines bị layout bỏ sót;
3. gắn `rendered_source` phục vụ retry crop OCR;
4. loại duplicate, nhận page number, chuẩn hóa text và reading order.

### 10.3 Khôi phục OCR line ngoài layout

`recover_unassigned_ocr_lines` ưu tiên `rec_boxes`, fallback sang bounding rectangle của `rec_polys`. Một line được coi đã cover khi:

- text compact trùng và containment vượt `duplicate_overlap`; hoặc
- coverage vượt threshold và line nằm trong text block; hoặc
- coverage vượt threshold và block là table/formula có nội dung.

Line chưa cover được tạo thành paragraph với `layout_fallback=true`, confidence và `ocr_lines`. Vị trí chèn được chọn theo block có vùng X giao nhau và tọa độ Y gần nhất; layout order vẫn là nguồn chính.

### 10.4 Parse file ảnh

`PaddleImageParser`:

- copy ảnh gốc vào `assets/original.<ext>` nếu có output assets;
- tạo một page `scanned` với kích thước pixel;
- luôn tạo element image gốc phủ toàn ảnh;
- append các element OCR;
- nếu OCR lỗi, vẫn giữ image gốc, đánh page failed và ghi lỗi.

## 11. Hậu xử lý OCR

### 11.1 Gắn và tái tạo dòng

Raw line từ `overall_ocr_res` được gán cho layout block có tỷ lệ diện tích line nằm trong block lớn nhất. Chỉ gán khi vượt `line_assignment_ratio`.

Với block text-like, các line được gom theo tâm Y và dung sai chiều cao, sort theo X rồi ghép bằng khoảng trắng. Text block chỉ bị thay khi `SequenceMatcher` giữa hai bản compact đạt `line_match_ratio`. Mặc định `1.0`, nghĩa là chỉ sửa khoảng trắng/cách ghép, không chấp nhận mất/thêm ký tự.

Mọi thay đổi text ghi:

```json
{
  "text_original": "...",
  "text_corrected": "...",
  "text_changes": [
    {"reason": "...", "before": "...", "after": "..."}
  ]
}
```

### 11.2 Loại duplicate

Hai element được coi trùng khi text compact giống nhau, không rỗng, và overlap trên diện tích nhỏ hơn đạt threshold. Element đầu được giữ; bản sau nằm trong `duplicate_sources` để audit.

### 11.3 Số trang

Block được nhận là số trang khi:

- nằm trong biên trên hoặc dưới theo `page_number_margin_ratio`;
- tâm ngang gần tâm trang;
- text ngắn, chứa số và chỉ gồm số/space/ngoặc/`C`.

Element được đổi thành `footer`, gắn `exclude_from_content=true` và vẫn tồn tại trong JSON. Serializer Markdown bỏ các element có cờ này.

### 11.4 Chuẩn hóa text

`clean_text` chuẩn hóa Unicode NFC, bỏ null byte, thu gọn horizontal whitespace, sửa khoảng trắng quanh dấu câu. URL, email, domain, dotted abbreviation và decimal/group separators được bảo toàn theo heuristic.

### 11.5 Reading order

- `preserve`: giữ thứ tự layout.
- `single_column`: luôn sort `(top, left)`.
- `auto`: dò cặp block text đủ rộng, cách nhau theo chiều ngang và overlap theo chiều dọc. Nếu phát hiện hai cột thì giữ layout order; nếu không thì sort hình học.

Trước khi sort, thứ tự cũ được ghi vào `order_before_postprocessing`.

### 11.6 Review số Điều/Chương

`review_document` chỉ xét element PaddleOCR chưa bị exclude. Nó theo dõi chuỗi:

- `Điều <số Arabic>`;
- `CHƯƠNG <số Roman>`.

Khi số hiện tại không bằng số trước + 1, tạo issue `numbering_sequence`. Nếu retry bật, engine crop dòng heading từ ảnh render và nhận dạng lại ở các scale. Chỉ sửa khi ít nhất `retry_min_agreement` scale phân biệt cùng đọc ra giá trị kỳ vọng và từng reading đủ confidence. Nếu không, giữ nguyên text và status `needs_review`. Bộ đếm nội bộ tiến theo expected value để tránh một lỗi tạo cascade cảnh báo giả.

### 11.7 Review năm và lỗi lặp chữ

- `năm 19 30` được gợi ý thành `năm 1930`, không tự sửa.
- Từ có hai ký tự kề nhau cùng base letter được thử bỏ từng ký tự; nếu kết quả có trong vocabulary thì tạo suggestions, không tự sửa.

Thống kê cuối cùng nằm tại `document.metadata.ocr_postprocessing`: số element thay đổi, số page number bị exclude và số issue cần review.

## 12. Sửa text OCR theo candidate

`CorrectorFactory` chỉ import và tạo backend được chọn. Bốn adapter đều dùng hợp đồng
`TextCorrector.correct(text) -> CorrectionCandidate`; model/tokenizer Hugging Face và PyTorch chỉ
được load ở inference đầu tiên. Text dài được chia segment theo token budget rồi ghép lại, không
truncate âm thầm. `bravend` chuẩn hóa biến thể dấu câu/dấu thanh theo yêu cầu model và chạy theo
segment câu; ba backend còn lại dùng adapter Seq2Seq chung.

`TextCorrectionService` xử lý sau OCR review và trước lần normalize cuối:

1. lọc element theo type, provenance OCR và `exclude_from_content`;
2. dùng `CorrectionDetector` quyết định có gọi model hay không;
3. lấy candidate từ backend;
4. kiểm tra candidate bằng `CorrectionSafetyGate`;
5. chỉ candidate `accepted` mới thay text qua `record_text_change`;
6. ghi audit `text_correction` ở element và thống kê ở document metadata.

Các trạng thái gồm `accepted`, `not_needed`, `needs_review`, `rejected`, `error`. Lỗi load/inference
model được ghi vào element và không làm fail cả document. Bản gốc đầu tiên luôn được giữ trong
`text_original`; correction không được tự thay số, ngày, tỷ lệ, tiền tệ, IP, version, mã văn bản,
tên kỹ thuật được bảo vệ hoặc cấu trúc pháp lý.

## 13. Chuẩn bị dữ liệu retrieval

`RetrievalPreprocessor.process(CanonicalDocument)` tạo `RetrievalDocument` mới và không mutate
document nguồn. Nó bỏ header/footer/image theo mặc định, bỏ mọi element có
`exclude_from_content=true`, loại text rỗng và không mang theo payload debug OCR/correction lớn.

Mỗi `RetrievalElement` giữ traceability gồm document/element ID, page, bbox, parent, section,
order và `source_element_ids`. `heading_path` theo dõi heading level hoặc cấu trúc pháp lý
`CHƯƠNG`/tiêu đề chương/`Điều`; `text_for_embedding` nối context heading với text element. Cleaner
chỉ chuẩn hóa Unicode/whitespace, nối từ bị ngắt dòng bằng dấu gạch ngang khi an toàn và giữ marker,
số, dấu câu cùng cấu trúc Markdown/HTML của table.

Module có schema `Chunk` và protocol `BaseChunker` để định nghĩa biên tích hợp, nhưng chưa có thuật
toán chunking, embedding provider, vector store hoặc truy vấn retrieval.

## 14. Đặc tả DOCX

`DoclingDOCXParser` có hai chế độ:

1. Mặc định gọi trực tiếp `MsWordDocumentBackend`, tránh import các OCR pipeline không liên quan của `DocumentConverter`.
2. Khi inject converter (chủ yếu cho test/tích hợp), chấp nhận `success` hoặc `partial_success`; status khác gây `DocumentParseError`.

Parser:

- ghi backend, conversion status và errors vào document metadata;
- chuyển `doc.pages` thành canonical pages khi Docling cung cấp pagination;
- iterate item đúng structural order;
- lưu label gốc, structural level và Docling reference;
- bảng có Markdown, HTML và raw table data;
- ảnh được lưu asset nếu Docling trả bytes, nếu không ghi warning;
- heading level, list marker/enumerated được lưu khi có;
- provenance bbox được chuyển về top-left origin;
- quan hệ parent Docling được resolve sang canonical `parent_id` sau khi đã thu thập tất cả reference.

DOCX có thể không có `pages` hoặc bbox; khi đó element page number là `None`, đây là trạng thái hợp lệ.

## 15. Đặc tả XLSX

`ExcelParser` mở workbook hai lần:

- `data_only=False` để đọc công thức;
- `data_only=True` để lấy cached value.

Mỗi worksheet:

1. Chặn sheet có `max_row * max_column` vượt `max_excel_region_cells` để tránh workbook bị phình do format.
2. Thu metadata sheet state, merged ranges, hidden rows/columns.
3. Xác định cell occupied chỉ theo cell có value; format-only cell không ảnh hưởng.
4. Merged range được coi occupied toàn vùng nếu ô top-left có value.
5. Tách contiguous row bands, rồi contiguous column bands trong từng row band.
6. Mỗi rectangular region trở thành một element `table`.

Mỗi cell có coordinate, row, column, value, formula, data type và number format. Date/time được chuyển ISO string; kiểu lạ chuyển `str`. Với formula, parser ưu tiên cached value nếu có; nếu cache rỗng thì đặt formula vào bảng. OpenPyXL không tính lại công thức, điều này được ghi rõ trong metadata.

XLSX không tạo canonical pages; sheet/range nằm trong metadata element. Markdown serializer chèn heading `## Sheet: <name>` khi sheet thay đổi.

## 16. Quality validation

### 16.1 Quality gate cấp trang PDF

`page_needs_ocr` bỏ qua blank page. Với trang khác, nối text của mọi element không phải image và yêu cầu:

- text không rỗng;
- printable ratio đủ ngưỡng;
- invalid ratio không vượt ngưỡng;
- quality đủ ngưỡng.

Gate này không kiểm tra CER/WER và không so sánh với ground truth. Đây là utility tương thích còn
được test độc lập; luồng `PDFParser` hiện tại luôn OCR nên không gọi gate để fallback/chọn backend.

### 16.2 Quality cấp document

Warnings phát sinh khi:

- không có element;
- không có text ngoài image;
- invalid character ratio quá cao;
- có page failed;
- Docling báo `partial_success`.

Nếu có bất kỳ warning, status là `partial`; nếu không là `ok`. Kết quả còn có element count, text length, invalid ratio, quality score, page count, successful/failed pages.

Lưu ý: document chỉ có ảnh hợp lệ vẫn được đánh `partial` vì không có text. `ok` chỉ phản ánh tính toàn vẹn/khả dụng cơ bản, không phải độ chính xác nội dung.

## 17. Serialization và output

### 17.1 Layout thư mục

```text
<output-root>/
├── summary.json
└── <document-name>/
    ├── document.json
    ├── document.md
    └── assets/
        ├── original.png
        ├── p0001_render.png
        └── ...
```

### 17.2 Markdown

Markdown bắt đầu bằng `# <filename>`. Khi page đổi, chèn comment `<!-- Page N -->`; khi sheet đổi, chèn heading sheet. Heading canonical dùng `metadata.heading_level`, clamp từ 1 đến 6, mặc định 2. List thêm `- `. Image thành placeholder `[IMAGE: <asset>]`. Table text được giữ nguyên vì parser đã xuất Markdown hoặc HTML.

### 17.3 Atomic write

`atomic_text` ghi `<filename>.tmp`, sau đó `os.replace` sang tên thật. Điều này giảm nguy cơ để lại JSON/Markdown dở dang khi process bị gián đoạn tại bước ghi text. Asset không dùng cơ chế atomic tương tự.

### 17.4 Markdown table utility

`table_markdown` dùng dòng đầu làm header, tự pad hàng thiếu cell, escape `|` thành `\|` và newline trong cell thành `<br>`.

## 18. API công khai

Package export:

```python
from document_parser import (
    CanonicalDocument,
    DocumentParser,
    Element,
    LayoutDetector,
    OCRDetector,
    OCRRecognizer,
    Page,
    TableRecognizer,
    DocumentPipeline,
    ParserConfig,
    PDFAnalyzerConfig,
    ProjectConfig,
    RetrievalDocument,
    RetrievalElement,
    RetrievalPreprocessor,
)
```

Ví dụ parse batch:

```python
from document_parser import DocumentPipeline, ProjectConfig

config = ProjectConfig.load().parser_config(device="cpu")
summary = DocumentPipeline(config).parse_all(
    input_dir="documents",
    output_dir="parsed_test_document/api",
)
```

Ví dụ parse một file:

```python
document = DocumentPipeline(config).parse(
    "documents/report.pdf",
    "parsed_test_document/report",
)

retrieval_document = RetrievalPreprocessor().process(document)
```

## 19. CLI và script vận hành

### 19.1 `parse_test_documents.py`

Nhận `--config`, `--device`, ba override model, `--corrector`, `--input`, `--output`. Exit code:

- `0`: có ít nhất một file và tất cả `ok`;
- `1`: không có file hỗ trợ, có file `failed`, hoặc có document `partial`;
- lỗi argparse/config: argparse kết thúc với mã lỗi thông thường của parser.

### 19.2 `download_ocr_models.py`

Chỉ tải khi model recognition đang chọn đúng `recognition_download.repository`. Dùng Hugging Face snapshot với revision cố định và allow-list file. Sau tải, ghi `model_manifest.json` gồm repository, revision, source URL và SHA-256 từng file. Model Paddle chính thức để Paddle tự tải lúc OCR.

### 19.3 `check_gpu.py`

Chọn đúng một GPU. Nếu config mặc định là CPU và caller không truyền device, script chủ động kiểm tra `gpu:0`. Sau preflight, chạy matrix multiplication và convolution để xác thực CUDA/cuBLAS/cuDNN, rồi xuất JSON version/device/capability và giá trị kiểm tra. Exit `1` nếu bất kỳ bước nào lỗi.

### 19.4 `benchmark_ocr.py`

Render các trang PDF được chọn, chạy chung một engine để đo cold-start ở trang đầu và warm inference ở trang sau. Khi dùng GPU, thread nền gọi `nvidia-smi` mỗi 0,2 giây để lấy peak total VRAM used và GPU utilization. Report được cập nhật sau từng trang và luôn ghi lần cuối trong `finally`. Trang ngoài phạm vi hoặc OCR lỗi đặt status `failed`, exit `1`.

## 20. Exception và chiến lược lỗi

| Exception | Trường hợp |
|---|---|
| `DocumentParseError` | file không tồn tại/rỗng, backend trả payload không hợp lệ, sheet quá lớn, conversion lỗi |
| `UnsupportedFormatError` | extension hoặc parser không được hỗ trợ |
| `DependencyUnavailableError` | thiếu dependency, Paddle/OCR không khởi tạo, GPU không dùng được |
| `ValueError` | config/path/output/schema/bbox không hợp lệ |

Biên bắt lỗi:

- `PDFParser`: bắt theo trang, đánh `Page.status=failed`;
- `PaddleImageParser`: giữ ảnh gốc, đánh trang failed;
- `DocumentPipeline.parse_all`: bắt theo file, ghi error summary và tiếp tục;
- `DocumentPipeline.parse`: không nuốt exception cấp file; caller nhận exception;
- CLI batch: dựa vào summary để quyết định exit code.

## 21. Dependency và đóng gói

Các dependency chức năng:

| Package | Vai trò |
|---|---|
| PyMuPDF | mở và render trang PDF thành ảnh; không trích xuất text/table |
| Pillow | đọc, crop, resize và lưu ảnh |
| PaddleOCR/PaddleX | layout analysis, OCR, text recognition |
| paddlepaddle hoặc paddlepaddle-gpu | runtime inference CPU/GPU |
| docling-slim/docling-core | đọc DOCX và structural model |
| openpyxl | đọc XLSX |
| Pydantic | schema và validation |
| PyYAML | cấu hình YAML |
| huggingface-hub | tải model recognition tùy chỉnh |
| Transformers/PyTorch/SentencePiece | các backend sửa lỗi OCR theo candidate |
| NumPy/OpenCV | dữ liệu ảnh và dependency OCR |
| pytest | test runner |

Package dùng `setuptools`, source layout tại `src`, cài `config.yaml` vào `share/rag-document-parser`. Người dùng phải chọn đúng một extra `cpu` hoặc `gpu`; hai distribution đều cung cấp module `paddle`.

## 22. Kiểm thử hiện có

### 22.1 Nhóm test

| File | Phạm vi |
|---|---|
| `test_router.py` | mapping extension và unsupported format |
| `test_pdf_analyzer.py` | utility phân tích PDF và text metrics |
| `test_normalizer.py` | label mapping, ID/section, punctuation, reading order, review OCR |
| `test_adapters.py` | GPU preflight, image OCR, omitted lines, Docling structure, real DOCX assets |
| `test_ocr_architecture.py` | ranh giới detection/recognition, layout confidence và cấu trúc table |
| `test_correction.py` | detector, adapter lazy, segmentation, safety gate và audit correction |
| `test_pipeline.py` | bắt buộc OCR mọi PDF, fault isolation, XLSX, path guard, batch collision |
| `test_config.py` | config merge/validation/path, CLI, benchmark, downloader, GPU checker |
| `test_retrieval.py` | cleaning bảo thủ, heading context, exclusion và traceability retrieval |

### 22.2 Ý đồ được test bảo vệ

- PDF có text layer vẫn bắt buộc chạy OCR.
- PDF nhiều trang OCR toàn bộ các trang.
- OCR failure chỉ làm failed trang tương ứng.
- Bbox OCR sai không phá trang khác.
- File nguồn giữ nguyên hash sau parse.
- XLSX giữ formula, merged region và hidden row/column.
- Cùng stem khác extension có output riêng.
- OCR line bị layout bỏ sót không bị mất và không nhân đôi text/table đã cover.
- Page number tồn tại trong JSON nhưng không xuất Markdown.
- Sửa numbering chỉ xảy ra khi nhiều retry crop đồng thuận đủ confidence.
- Override config không mutate shared defaults.
- Environment do caller đặt không bị ghi đè.
- Model tùy chỉnh tải đúng revision và sinh checksum manifest.
- Backend correction chỉ load lazy, và safety gate chặn sửa ngoài token được detector cho phép.
- Retrieval preprocessing không mutate canonical document và không đưa debug/candidate bị từ chối vào embedding text.

Các test đánh dấu `integration` cần fixture thật và dependency backend. Bốn test correction model thật
đồng thời mang marker `slow` và chỉ chạy inference khi biến môi trường opt-in tương ứng được đặt.

## 23. Bất biến hệ thống

1. `document_id` phụ thuộc nội dung file, không phụ thuộc đường dẫn.
2. `order` luôn được đánh lại liên tục từ 0 sau normalize.
3. JSON không chứa object thư viện ngoài không serialize được.
4. Một element không thể tham chiếu parent/page không tồn tại.
5. Output batch không nằm trong input tree.
6. Parser không tự thay source file.
7. Yêu cầu GPU không được fallback thầm sang CPU.
8. Page number bị loại khỏi content vẫn được bảo toàn trong dữ liệu audit.
9. Text correction phải để lại bản gốc và lý do thay đổi.
10. Lỗi file/trang phải được phản ánh trong summary/quality, không được biến thành `ok`.

## 24. Giới hạn và rủi ro kỹ thuật hiện tại

- Pipeline mới chuẩn bị element cho retrieval; chưa tạo chunk, embedding, index hay truy vấn RAG.
- Heading và reading order là heuristic, có thể sai với layout phức tạp.
- Mọi trang PDF đều OCR nên tốn tài nguyên hơn native extraction và có thể làm giảm độ chính xác của PDF có text layer tốt.
- Table recognition bật mặc định; formula/chart recognition đang tắt mặc định trong config.
- DOCX có thể không có pagination/bbox thật tùy backend và tài liệu.
- OpenPyXL không tính công thức; cached value phụ thuộc lần save trước của Excel.
- Quality `ok` không đo CER/WER hoặc độ đúng ngữ nghĩa.
- Vocabulary phát hiện lỗi chính tả nhỏ và thiên về tiếng Việt pháp lý; chỉ mang tính gợi ý.
- `document_id` chỉ lưu 80 bit đầu SHA-256; đủ thực dụng cho tập nhỏ nhưng không phải toàn hash.
- `parse_all` chạy tuần tự, chưa có parallelism hoặc resource scheduling.
- Output JSON có thể lớn vì giữ raw cell data, provenance, OCR lines và parser config.
- Asset write không atomic và output cũ không được dọn trước lần parse mới; file asset dư có thể còn lại.
- Guard output của `parse(path)` coi toàn bộ thư mục chứa source là vùng cấm ghi; caller không thể đặt output ở một thư mục con khác của cùng source directory.
- `apply_environment` không ghi đè environment cũ; thuận lợi cho caller nhưng có thể khiến YAML không phải giá trị runtime cuối cùng.
- Cache lỗi khởi tạo OCR tồn tại suốt đời `PaddleEngine`; sửa environment/dependency trong cùng process cần tạo engine mới.

## 25. Yêu cầu phi chức năng suy ra từ code

### Tính tin cậy

- fault isolation theo page và file;
- atomic write cho JSON/Markdown;
- revision model tùy chỉnh được pin và có checksum;
- validation schema trước khi xuất.

### Khả năng quan sát

- logging theo file/page và parser được chọn;
- summary batch và summary lỗi;
- metadata analysis, model config, quality, OCR confidence và correction history;
- benchmark lưu timing và GPU samples.

### Hiệu năng

- OCR lazy và engine reuse;
- model correction lazy và chỉ chạy với candidate đáng ngờ;
- OCR PDF render theo DPI cấu hình;
- giới hạn cell sheet chống memory blow-up;
- model và cache được đặt trong thư mục cấu hình.

### Khả năng mở rộng

Để thêm format mới cần:

1. thêm extension vào `SUPPORTED_EXTENSIONS`;
2. thêm mapping trong `DocumentRouter.FORMATS`;
3. triển khai adapter theo `BaseDocumentParser`;
4. đăng ký factory trong `DocumentPipeline.parse`;
5. trả dữ liệu tương thích `CanonicalDocument`/`Normalizer`;
6. bổ sung test route, adapter, pipeline và serialization.

## 26. Tiêu chí nghiệm thu chức năng

Hệ thống được coi đạt yêu cầu hiện tại khi:

1. cài được package trên Python hỗ trợ với đúng một Paddle backend;
2. parse được từng extension đã công bố;
3. mọi trang PDF gọi OCR đúng một lần và không đọc text layer native;
4. lỗi một trang/file không làm mất kết quả độc lập khác;
5. `document.json` validate theo schema và parse lại được;
6. `document.md` không chứa page number đã exclude;
7. asset tham chiếu tồn tại hoặc metadata nêu rõ lỗi/warning;
8. summary count khớp status file;
9. CLI trả mã khác 0 với empty/partial/failed batch;
10. unit test và integration test phù hợp môi trường đều pass.

## 27. Trạng thái xác minh khi lập tài liệu

Source đã được đối chiếu giữa implementation, YAML, script và test hiện có. Trong môi trường
`.venv-paddle-vl`, nhóm không-integration đã được xác minh ngày 2026-10-07:

```text
148 passed, 6 deselected in 11.06s
```

Lệnh kiểm tra nhanh không tải/chạy model integration:

```bash
.venv-paddle-vl/bin/python -m pytest -q -m 'not integration'
```

Các test integration chạy riêng vì cần fixture/model/backend thật và có thể mất nhiều thời gian:

```bash
.venv-paddle-vl/bin/python -m pytest -q -m integration
```

Các lệnh lint dự kiến:

```bash
ruff check src scripts tests
ruff format --check src scripts tests
```

Ruff chưa được xác minh trong environment này vì executable/module chưa được cài ở thời điểm cập
nhật tài liệu.

Xác minh runtime bổ sung cùng ngày:

- `scripts/check_gpu.py --device gpu:0` pass trên NVIDIA GeForce RTX 3090;
- integration DOCX thật: `1 passed`;
- CLI end-to-end với PDF 5 trang: 1 document `ok`, đủ 5 trang, 80 elements, schema/hash/assets hợp lệ;
- chưa chạy lại integration PDF 42 trang và bốn correction backend cần opt-in riêng.
