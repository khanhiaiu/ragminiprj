#!/usr/bin/env python3
"""Extract image context, run an egress-gated caption pilot/batch, and resume safely."""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from document_parser.normalization.schema import CanonicalDocument  # noqa: E402
from rag.enrichment.cache import CaptionCache  # noqa: E402
from rag.enrichment.caption import CaptionRecord, CaptionStatus  # noqa: E402
from rag.enrichment.context import ImageAnchor, ImageContext, ImageContextExtractor  # noqa: E402
from rag.enrichment.service import CaptionService  # noqa: E402
from rag.enrichment.tokenizer import load_bge_m3_tokenizer  # noqa: E402
from rag.enrichment.vlm import EgressPolicy, GeminiVLMClient  # noqa: E402


def discover(input_dir: Path) -> list[Path]:
    paths = sorted(input_dir.glob("*/document.json"))
    if (input_dir / "document.json").exists():
        paths.append(input_dir / "document.json")
    return paths


def write_jsonl(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def synthetic_preflight_context() -> ImageContext:
    return ImageContext(
        document_id="synthetic-preflight",
        image_element_id="synthetic-image",
        before_context="Synthetic preflight image.",
        after_context="",
        before_token_count=4,
        after_token_count=0,
        total_token_count=4,
        source_spans=[],
        image_anchor=ImageAnchor(element_id="synthetic-image", order=0),
        pages=[],
        tokenizer_fingerprint="synthetic-preflight",
        context_hash="synthetic-preflight-v1",
    )


def preflight(client: GeminiVLMClient) -> dict:
    image = Image.new("RGB", (96, 64), "white")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return client.preflight(image_bytes=output.getvalue(), context=synthetic_preflight_context())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "parsed_test_document/all_documents_gpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", type=int, default=0, help="Process a representative deterministic sample")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--authorize-external-egress",
        metavar="ACK",
        help="Required exact value: I_AUTHORIZE_DOCUMENT_EGRESS",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_bge_m3_tokenizer(local_files_only=args.local_files_only)
    extractor = ImageContextExtractor(tokenizer)
    client = GeminiVLMClient()
    policy = EgressPolicy(
        allow_external=args.authorize_external_egress == "I_AUTHORIZE_DOCUMENT_EGRESS",
        acknowledgement=args.authorize_external_egress,
    )
    if args.preflight_only:
        result = preflight(client)
        (args.output / "preflight.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return 0

    contexts = []
    work = []
    for document_path in discover(args.input):
        document = CanonicalDocument.model_validate_json(document_path.read_text(encoding="utf-8"))
        by_id = {element.element_id: element for element in document.elements}
        for context in extractor.extract_all(document):
            contexts.append(context.model_dump(mode="json"))
            element = by_id[context.image_element_id]
            relative = element.metadata.get("asset_path")
            work.append((document_path, document, element, context, document_path.parent / relative))
    write_jsonl(args.output / "image_context_debug.jsonl", contexts)

    selected = work
    if args.pilot:
        # Stable stratification across corpus/order yields varied parsers, pages and image sizes.
        count = min(max(args.pilot, 8), 12, len(work))
        indexes = sorted({round(i * (len(work) - 1) / max(count - 1, 1)) for i in range(count)})
        selected = [work[index] for index in indexes]

    records = []
    selected_ids = {(d.document_id, e.element_id) for _, d, e, _, _ in selected}
    if not args.dry_run:
        policy.require()
        preflight_result = preflight(client)
        (args.output / "preflight.json").write_text(
            json.dumps(preflight_result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        service = CaptionService(client, CaptionCache(args.output / "caption_cache"), policy)
        for index, (_, document, element, context, asset) in enumerate(selected, start=1):
            records.append(
                service.process(
                    document_id=document.document_id,
                    image_element_id=element.element_id,
                    asset_path=asset,
                    context=context,
                    original_caption=element.text.strip() or None,
                    resume=args.resume,
                ).model_dump(mode="json")
            )
            print(
                f"Caption {index}/{len(selected)}: {element.element_id} {records[-1]['status']}",
                flush=True,
            )
            write_jsonl(args.output / "captions.jsonl", records)
    for document_path, document, element, context, asset in work:
        if (document.document_id, element.element_id) in selected_ids and not args.dry_run:
            continue
        records.append(
            CaptionRecord(
                document_id=document.document_id,
                image_element_id=element.element_id,
                asset_path=str(asset),
                image_hash="pending-dry-run",
                context_hash=context.context_hash,
                cache_key="pending",
                provider=client.provider,
                model=client.model,
                status=CaptionStatus.pending,
                context_metadata=context.model_dump(mode="json"),
                review_reasons=["not_selected_in_pilot" if args.pilot else "dry_run"],
            ).model_dump(mode="json")
        )
    records.sort(key=lambda value: (value["document_id"], value["image_element_id"]))
    write_jsonl(args.output / "captions.jsonl", records)
    review_records = [
        record
        for record in records
        if record["status"] in {"needs_review", "error"}
    ]
    write_jsonl(args.output / "error_review_report.jsonl", review_records)
    statuses = {}
    for record in records:
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1
    pilot_report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "discoverable_images": len(work),
        "selected_images": len(selected),
        "recorded_dispositions": len(records),
        "statuses": statuses,
        "review_fields": [
            "visual_correctness",
            "omitted_branches",
            "hallucinated_numbers",
            "reading_order_accuracy",
        ],
        "items": [
            {
                "document_id": record["document_id"],
                "image_element_id": record["image_element_id"],
                "status": record["status"],
                "actual_token_count": record.get("context_metadata", {}).get("total_token_count"),
                "latency_ms": record.get("latency_ms"),
                "failure": record.get("error"),
                "human_visual_review": "pending",
            }
            for record in records
            if (record["document_id"], record["image_element_id"]) in selected_ids
        ],
    }
    (args.output / "pilot_report.json").write_text(
        json.dumps(pilot_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if all(r["status"] != "error" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
