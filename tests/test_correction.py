from copy import deepcopy

import pytest

from document_parser.config import ParserConfig
from document_parser.correction.base import CorrectionCandidate
from document_parser.correction.bmd1905 import BMD1905Corrector
from document_parser.correction.bravend import BravendCorrector
from document_parser.correction.factory import CorrectorFactory
from document_parser.correction.protonx import ProtonXCorrector
from document_parser.correction.protonx_legal import ProtonXLegalCorrector
from document_parser.correction.safety import CorrectionSafetyGate
from document_parser.correction.service import TextCorrectionService
from document_parser.normalization.normalizer import Normalizer


class FakeCorrector:
    def __init__(self, corrections=None, error=None, backend="protonx"):
        self.corrections = corrections or {}
        self.error = error
        self.calls = []
        self.backend = backend
        self.model_name = f"fake-{backend}"

    @property
    def initialized(self):
        return False

    def correct(self, text):
        self.calls.append(text)
        if self.error:
            raise self.error
        corrected = self.corrections.get(text, text)
        return CorrectionCandidate(
            text, corrected, self.backend, self.model_name, corrected != text
        )


def document(elements, file_type="pdf"):
    return Normalizer().normalize(
        {
            "document_id": "doc",
            "filename": f"scan.{file_type}",
            "file_type": file_type,
            "source_path": f"/scan.{file_type}",
            "pages": [{"page_number": 1}],
            "elements": elements,
        }
    )


def ocr_element(text, **metadata):
    return {
        "element_type": "paragraph",
        "text": text,
        "page_number": 1,
        "metadata": {"parser": "paddleocr", **metadata},
    }


@pytest.mark.parametrize(
    "original,corrected",
    [
        (
            "Nước Cộng hòna xã hội chủ nghĩa Vhiệt Nam",
            "Nước Cộng hòa xã hội chủ nghĩa Việt Nam",
        ),
        ("quyền lưực nhà nước", "quyền lực nhà nước"),
        ("phuục vụ Nhân dân", "phục vụ Nhân dân"),
    ],
)
def test_suspicious_ocr_candidate_is_accepted(original, corrected):
    backend = FakeCorrector({original: corrected})
    parsed = document([ocr_element(original)])
    TextCorrectionService(deepcopy(ParserConfig().text_correction), backend).process(parsed)
    element = parsed.elements[0]
    assert element.text == corrected
    assert element.metadata["text_original"] == original
    assert element.metadata["text_corrected"] == corrected
    assert element.metadata["text_changes"][-1] == {
        "reason": "protonx_ocr_correction",
        "before": original,
        "after": corrected,
    }
    assert element.metadata["text_correction"]["status"] == "accepted"


def test_first_original_is_preserved_across_existing_and_model_corrections():
    current = "phuục vụ Nhân dân"
    parsed = document(
        [
            ocr_element(
                current,
                text_original="phuụcvụ Nhân dân",
                text_corrected=current,
                text_changes=[
                    {
                        "reason": "join_ocr_lines_with_spaces",
                        "before": "phuụcvụ Nhân dân",
                        "after": current,
                    }
                ],
            )
        ]
    )
    backend = FakeCorrector({current: "phục vụ Nhân dân"})
    TextCorrectionService(deepcopy(ParserConfig().text_correction), backend).process(parsed)
    assert parsed.elements[0].metadata["text_original"] == "phuụcvụ Nhân dân"
    assert len(parsed.elements[0].metadata["text_changes"]) == 2


@pytest.mark.parametrize(
    "original,candidate",
    [
        ("Điều 15", "Điều 16"),
        ("ngày 2 tháng 9 năm 1945", "ngày 2 tháng 9 năm 1954"),
        ("Nghị định 123/2020/NĐ-CP", "Nghị định 124/2020/NĐ-CP"),
    ],
)
def test_safety_rejects_changes_to_protected_information(original, candidate):
    config = deepcopy(ParserConfig().text_correction["safety"])
    decision = CorrectionSafetyGate(config).evaluate(
        CorrectionCandidate(original, candidate, "protonx", "fake", True)
    )
    assert not decision.accepted
    assert decision.status == "rejected"
    assert decision.reason == "protected_tokens_changed"


def test_large_paraphrase_needs_review_and_is_not_applied():
    original = "Nhà nước bảo đảm quyền con ngưười."
    candidate = "Nhà nước có trách nhiệm bảo đảm đầy đủ các quyền cơ bản của con người."
    parsed = document([ocr_element(original)])
    backend = FakeCorrector({original: candidate})
    TextCorrectionService(deepcopy(ParserConfig().text_correction), backend).process(parsed)
    correction = parsed.elements[0].metadata["text_correction"]
    assert parsed.elements[0].text == original
    assert correction["status"] == "needs_review"
    assert correction["reason"] == "edit_ratio_exceeded"
    assert correction["candidate"] == candidate


def test_candidate_that_adds_a_sentence_is_rejected_as_structure_change():
    config = deepcopy(ParserConfig().text_correction["safety"])
    decision = CorrectionSafetyGate(config).evaluate(
        CorrectionCandidate(
            "Nội dung được bảo đảm.",
            "Nội dung được bảo đảm. Câu mới.",
            "protonx",
            "fake",
            True,
        )
    )
    assert not decision.accepted
    assert decision.status == "rejected"
    assert decision.reason == "structure_changed"


def test_clean_ocr_does_not_call_backend():
    backend = FakeCorrector()
    parsed = document([ocr_element("Nước Cộng hòa xã hội chủ nghĩa Việt Nam")])
    TextCorrectionService(deepcopy(ParserConfig().text_correction), backend).process(parsed)
    assert backend.calls == []
    assert "text_correction" not in parsed.elements[0].metadata


def test_low_confidence_alone_does_not_call_backend():
    backend = FakeCorrector()
    parsed = document([ocr_element("LỜI NÓI ĐẦU", ocr_confidence=0.5)])
    TextCorrectionService(deepcopy(ParserConfig().text_correction), backend).process(parsed)
    assert backend.calls == []


def test_candidate_editing_an_unflagged_word_is_rejected():
    original = "Moọi người có quyền bất khả xâm phạm."
    candidate = "Mọi người cho quyền bất khả xâm phạm."
    parsed = document([ocr_element(original)])
    backend = FakeCorrector({original: candidate})
    TextCorrectionService(deepcopy(ParserConfig().text_correction), backend).process(parsed)
    correction = parsed.elements[0].metadata["text_correction"]
    assert parsed.elements[0].text == original
    assert correction["status"] == "rejected"
    assert correction["reason"] == "unsupported_edit_scope"


def test_candidate_using_wrong_replacement_for_flagged_word_is_rejected():
    original = "quyền và lợi lích hợp pháp"
    candidate = "quyền và lợi lịch hợp pháp"
    parsed = document([ocr_element(original)])
    backend = FakeCorrector({original: candidate})
    TextCorrectionService(deepcopy(ParserConfig().text_correction), backend).process(parsed)
    correction = parsed.elements[0].metadata["text_correction"]
    assert parsed.elements[0].text == original
    assert correction["status"] == "rejected"
    assert correction["reason"] == "unsupported_edit_scope"


def test_protonx_backend_construction_does_not_load_model():
    backend = ProtonXCorrector(deepcopy(ParserConfig().text_correction))
    assert not backend.initialized


@pytest.mark.parametrize(
    "backend_name,corrector_type",
    [
        ("protonx", ProtonXCorrector),
        ("protonx_legal", ProtonXLegalCorrector),
        ("bmd1905", BMD1905Corrector),
        ("bravend", BravendCorrector),
    ],
)
def test_selected_backend_stays_lazy_when_clean_ocr_needs_no_inference(
    backend_name, corrector_type
):
    config = deepcopy(ParserConfig().text_correction)
    config["backend"] = backend_name
    corrector = CorrectorFactory.create(config)
    assert isinstance(corrector, corrector_type)
    parsed = document([ocr_element("Nước Cộng hòa xã hội chủ nghĩa Việt Nam")])
    TextCorrectionService(config, corrector).process(parsed)
    assert not corrector.initialized


@pytest.mark.parametrize(
    "backend_name,expected_module,expected_class",
    [
        ("protonx", "document_parser.correction.protonx", "ProtonXCorrector"),
        (
            "protonx_legal",
            "document_parser.correction.protonx_legal",
            "ProtonXLegalCorrector",
        ),
        ("bmd1905", "document_parser.correction.bmd1905", "BMD1905Corrector"),
        ("bravend", "document_parser.correction.bravend", "BravendCorrector"),
    ],
)
def test_factory_constructs_only_selected_backend(
    monkeypatch, backend_name, expected_module, expected_class
):
    import document_parser.correction.factory as factory_module

    imported = []
    constructed = []

    class SelectedCorrector:
        def __init__(self, config):
            constructed.append((expected_class, config["backend"]))

    class Module:
        pass

    module = Module()
    setattr(module, expected_class, SelectedCorrector)

    def fake_import(name):
        imported.append(name)
        return module

    monkeypatch.setattr(factory_module, "import_module", fake_import)
    config = deepcopy(ParserConfig().text_correction)
    config["backend"] = backend_name
    assert CorrectorFactory.create(config) is not None
    assert imported == [expected_module]
    assert constructed == [(expected_class, backend_name)]


def test_disabled_factory_has_no_active_corrector_or_backend_import(monkeypatch):
    import document_parser.correction.factory as factory_module

    monkeypatch.setattr(
        factory_module,
        "import_module",
        lambda name: pytest.fail(f"Unexpected backend import: {name}"),
    )
    config = deepcopy(ParserConfig().text_correction)
    config["enabled"] = False
    assert CorrectorFactory.create(config, override=FakeCorrector()) is None


def test_protonx_segments_long_text_without_truncation_or_detaching_legal_label():
    config = deepcopy(ParserConfig().text_correction)
    config["generation"]["max_input_tokens"] = 8
    backend = ProtonXCorrector(config)

    class WordTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(len(text.split()) + 2))}

    backend._tokenizer = WordTokenizer()
    text = (
        "Điều 15. Nội dung quy định quyền công dân trong trường hợp đặc biệt. "
        "Quy định này tiếp tục được áp dụng."
    )
    segments = backend._segments(text)
    assert "".join(segment.text for segment in segments) == text
    assert all(backend._token_length(segment.text) <= 8 for segment in segments)
    assert segments[0].text.startswith("Điều 15. Nội dung")


def test_bravend_uses_model_normalization_and_sentence_level_segments():
    config = deepcopy(ParserConfig().text_correction)
    config["backend"] = "bravend"
    backend = BravendCorrector(config)

    class WordTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(len(text.split()) + 2))}

    backend._tokenizer = WordTokenizer()
    text = "Điều 15. Nội dung thứ nhất. Nội dung thứ hai."
    segments = backend._segments(text)
    assert "".join(segment.text for segment in segments) == text
    assert len(segments) == 2
    assert segments[0].text.startswith("Điều 15.")
    assert backend._prepare_input("“Cộng hòa…”") == '"Cộng hoà..."'
    assert backend.trust_remote_code
    assert backend.revision == "31f323bd148f9f0e23722bf3226a752d124c1023"


def test_disabled_correction_does_not_call_backend():
    config = deepcopy(ParserConfig().text_correction)
    config["enabled"] = False
    backend = FakeCorrector()
    parsed = document([ocr_element("phuục vụ Nhân dân")])
    TextCorrectionService(config, backend).process(parsed)
    assert backend.calls == []
    assert "text_correction" not in parsed.metadata


@pytest.mark.parametrize("backend", ["protonx", "protonx_legal", "bmd1905", "bravend"])
def test_edit_ratio_policy_is_backend_independent(backend):
    original = "Nhà nước bảo đảm quyền con ngưười."
    corrected = "Nhà nước có trách nhiệm bảo đảm đầy đủ các quyền cơ bản của con người."
    safety = CorrectionSafetyGate(deepcopy(ParserConfig().text_correction["safety"]))
    decision = safety.evaluate(
        CorrectionCandidate(original, corrected, backend, f"fake-{backend}", True)
    )
    assert not decision.accepted
    assert decision.status == "needs_review"
    assert decision.reason == "edit_ratio_exceeded"


@pytest.mark.parametrize("backend", ["protonx", "protonx_legal", "bmd1905", "bravend"])
def test_protected_number_policy_is_backend_independent(backend):
    safety = CorrectionSafetyGate(deepcopy(ParserConfig().text_correction["safety"]))
    decision = safety.evaluate(
        CorrectionCandidate("Điều 15", "Điều 16", backend, f"fake-{backend}", True)
    )
    assert not decision.accepted
    assert decision.reason == "protected_tokens_changed"


@pytest.mark.parametrize(
    "original,candidate",
    [
        ("Máy chủ 10.210.22.89", "Máy chủ 10.210.22.98"),
        ("Phiên bản v2.10.3", "Phiên bản v2.11.3"),
        ("Dùng PP-StructureV3", "Dùng PP-StructureV4"),
        ("Vector store Qdrant", "Vector store Quadrant"),
        ("Tài khoản KhanhNT17_C", "Tài khoản KhanhNT18_C"),
    ],
)
def test_technical_identifiers_are_protected(original, candidate):
    safety = CorrectionSafetyGate(deepcopy(ParserConfig().text_correction["safety"]))
    decision = safety.evaluate(
        CorrectionCandidate(original, candidate, "bmd1905", "fake-bmd1905", True)
    )
    assert not decision.accepted
    assert decision.reason == "protected_tokens_changed"


@pytest.mark.parametrize("parser,file_type", [("pymupdf", "pdf"), ("docling", "docx")])
def test_native_and_docx_text_do_not_call_backend_by_default(parser, file_type):
    backend = FakeCorrector()
    parsed = document(
        [
            {
                "element_type": "paragraph",
                "text": "phuục vụ Nhân dân",
                "page_number": 1,
                "metadata": {"parser": parser},
            }
        ],
        file_type,
    )
    TextCorrectionService(deepcopy(ParserConfig().text_correction), backend).process(parsed)
    assert backend.calls == []


def test_footer_page_number_is_not_corrected():
    backend = FakeCorrector()
    parsed = document(
        [
            {
                **ocr_element("15", exclude_from_content=True),
                "element_type": "footer",
            }
        ]
    )
    TextCorrectionService(deepcopy(ParserConfig().text_correction), backend).process(parsed)
    assert backend.calls == []


@pytest.mark.parametrize("element_type", ["table", "image", "formula", "header", "unknown"])
def test_non_textual_ocr_elements_are_not_corrected(element_type):
    backend = FakeCorrector()
    parsed = document(
        [
            {
                **ocr_element("phuục vụ Nhân dân"),
                "element_type": element_type,
            }
        ]
    )
    TextCorrectionService(deepcopy(ParserConfig().text_correction), backend).process(parsed)
    assert backend.calls == []


def test_backend_failure_is_audited_without_failing_document():
    parsed = document([ocr_element("phuục vụ Nhân dân")])
    backend = FakeCorrector(error=RuntimeError("model unavailable"))
    TextCorrectionService(deepcopy(ParserConfig().text_correction), backend).process(parsed)
    assert parsed.elements[0].text == "phuục vụ Nhân dân"
    assert parsed.elements[0].metadata["text_correction"]["status"] == "error"
    assert "model unavailable" in parsed.elements[0].metadata["text_correction"]["error"]


def test_accepted_correction_audits_selected_backend_and_model():
    original = "phuục vụ Nhân dân"
    parsed = document([ocr_element(original)])
    backend = FakeCorrector({original: "phục vụ Nhân dân"}, backend="bmd1905")
    TextCorrectionService(deepcopy(ParserConfig().text_correction), backend).process(parsed)
    metadata = parsed.elements[0].metadata
    assert metadata["text_original"] == original
    assert metadata["text_corrected"] == "phục vụ Nhân dân"
    assert metadata["text_correction"]["backend"] == "bmd1905"
    assert metadata["text_correction"]["model_name"] == "fake-bmd1905"
    assert metadata["text_changes"][-1]["reason"] == "bmd1905_ocr_correction"


@pytest.mark.integration
@pytest.mark.slow
def test_real_protonx_model_requires_explicit_opt_in(monkeypatch):
    import os

    if os.environ.get("RUN_PROTONX_INTEGRATION") != "1":
        pytest.skip("Set RUN_PROTONX_INTEGRATION=1 to download and run the ProtonX model")
    from document_parser.correction.protonx import ProtonXCorrector

    parser_config = ParserConfig()
    parser_config.apply_environment()
    config = deepcopy(parser_config.text_correction)
    config["device"] = "cpu"
    result = ProtonXCorrector(config).correct("phuục vụ Nhân dân")
    assert result.corrected


@pytest.mark.integration
@pytest.mark.slow
def test_real_protonx_legal_model_requires_explicit_opt_in():
    import os

    if os.environ.get("RUN_PROTONX_LEGAL_INTEGRATION") != "1":
        pytest.skip("Set RUN_PROTONX_LEGAL_INTEGRATION=1 to download and run protonx-legal-tc")
    parser_config = ParserConfig()
    parser_config.apply_environment()
    config = deepcopy(parser_config.text_correction)
    config["backend"] = "protonx_legal"
    config["device"] = "cpu"
    result = ProtonXLegalCorrector(config).correct("phuục vụ Nhân dân")
    assert result.corrected


@pytest.mark.integration
@pytest.mark.slow
def test_real_bmd1905_model_requires_explicit_opt_in():
    import os

    if os.environ.get("RUN_BMD1905_INTEGRATION") != "1":
        pytest.skip("Set RUN_BMD1905_INTEGRATION=1 to download and run the BMD1905 model")
    parser_config = ParserConfig()
    parser_config.apply_environment()
    config = deepcopy(parser_config.text_correction)
    config["backend"] = "bmd1905"
    config["device"] = "cpu"
    result = BMD1905Corrector(config).correct("phuục vụ Nhân dân")
    assert result.corrected


@pytest.mark.integration
@pytest.mark.slow
def test_real_bravend_model_requires_explicit_opt_in():
    import os

    if os.environ.get("RUN_BRAVEND_INTEGRATION") != "1":
        pytest.skip("Set RUN_BRAVEND_INTEGRATION=1 to download and run the Bravend model")
    parser_config = ParserConfig()
    parser_config.apply_environment()
    config = deepcopy(parser_config.text_correction)
    config["backend"] = "bravend"
    config["device"] = "cpu"
    result = BravendCorrector(config).correct("Tôi đi hoc ở trường.")
    assert result.corrected
