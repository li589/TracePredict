import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
TRACE_WEB_ROOT = REPO_ROOT / "Trace_Web"
ACTUAL_SAMPLES_PATH = TRACE_WEB_ROOT / "public" / "data" / "actual-samples.json"
MODEL_PREDICTIONS_PATH = TRACE_WEB_ROOT / "data" / "model_predictions.json"
MODEL_SAMPLES_PATH = TRACE_WEB_ROOT / "public" / "data" / "model-samples.json"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _normalize_user_id(user_id: str | int) -> str:
    return f"{int(user_id):03d}"


def _load_actual_sample_map() -> dict[str, dict[str, Any]]:
    samples = _load_json(ACTUAL_SAMPLES_PATH, [])
    if not isinstance(samples, list):
        return {}

    result: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        normalized_id = _normalize_user_id(sample.get("userId", 0))
        result[normalized_id] = sample
    return result


def _coerce_point(point: Any, fallback_timestamp: str | None = None, index: int = 0) -> dict[str, Any]:
    if not isinstance(point, dict):
        raise ValueError("Each prediction point must be a JSON object.")

    if "lat" not in point or "lng" not in point:
        raise ValueError("Each prediction point must contain `lat` and `lng`.")

    timestamp = point.get("timestamp") or fallback_timestamp or f"P{index + 1}"
    return {
        "lat": round(float(point["lat"]), 6),
        "lng": round(float(point["lng"]), 6),
        "timestamp": str(timestamp),
    }


def _load_prediction_points(predictions_file: Path, user_id: str) -> list[dict[str, Any]]:
    payload = _load_json(predictions_file, None)
    if payload is None:
        raise FileNotFoundError(f"Prediction file not found: {predictions_file}")

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        if "predictedFuture" in payload:
            return payload["predictedFuture"]

        user_payload = payload.get(user_id)
        if isinstance(user_payload, dict) and "predictedFuture" in user_payload:
            return user_payload["predictedFuture"]

    raise ValueError(
        "Prediction file must be either a point list, "
        "an object with `predictedFuture`, or an object keyed by user id."
    )


def upsert_model_prediction(
    user_id: str | int,
    predicted_future: list[dict[str, Any]],
    *,
    title: str | None = None,
    description: str | None = None,
    prediction_label: str = "模型预测",
    source_file: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    normalized_user_id = _normalize_user_id(user_id)
    actual_sample_map = _load_actual_sample_map()
    actual_sample = actual_sample_map.get(normalized_user_id)
    actual_future = []
    if isinstance(actual_sample, dict):
        actual_future = actual_sample.get("actualFuture") or []

    normalized_points = []
    for index, point in enumerate(predicted_future):
        fallback_timestamp = None
        if index < len(actual_future) and isinstance(actual_future[index], dict):
            fallback_timestamp = actual_future[index].get("timestamp")
        normalized_points.append(_coerce_point(point, fallback_timestamp, index))

    payload = _load_json(MODEL_PREDICTIONS_PATH, {})
    if not isinstance(payload, dict):
        payload = {}

    payload[normalized_user_id] = {
        "title": title or f"模型输出 {normalized_user_id}",
        "description": description or f"从 Python 侧导出的模型预测结果，用户 {normalized_user_id}。",
        "predictionLabel": prediction_label,
        "sourceFile": source_file or "python-export",
        "tags": tags or ["模型输出", "Python 导出"],
        "predictedFuture": normalized_points,
    }

    _write_json(MODEL_PREDICTIONS_PATH, payload)
    return payload[normalized_user_id]


def build_trace_web_model_samples() -> list[dict[str, Any]]:
    actual_samples = _load_json(ACTUAL_SAMPLES_PATH, [])
    prediction_map = _load_json(MODEL_PREDICTIONS_PATH, {})

    if not isinstance(actual_samples, list):
        actual_samples = []
    if not isinstance(prediction_map, dict):
        prediction_map = {}

    exported_samples: list[dict[str, Any]] = []
    for sample in actual_samples:
        if not isinstance(sample, dict):
            continue

        user_id = _normalize_user_id(sample.get("userId", 0))
        override = prediction_map.get(user_id)
        if not isinstance(override, dict):
            continue

        predicted_future = override.get("predictedFuture")
        if not isinstance(predicted_future, list) or not predicted_future:
            continue

        converted = dict(sample)
        converted["id"] = f"model-{user_id}"
        converted["title"] = override.get("title") or f"模型输出 {user_id}"
        converted["description"] = override.get("description") or f"模型输出轨迹，用户 {user_id}。"
        converted["predictedFuture"] = predicted_future
        converted["predictionLabel"] = override.get("predictionLabel") or "模型预测"
        converted["sampleType"] = "actual"
        converted["tags"] = list(
            dict.fromkeys(
                ["模型输出", *(sample.get("tags") or []), *(override.get("tags") or [])]
            )
        )
        if override.get("sourceFile"):
            converted["sourceFile"] = override["sourceFile"]

        exported_samples.append(converted)

    _write_json(MODEL_SAMPLES_PATH, exported_samples)
    return exported_samples


def export_prediction_file(
    *,
    user_id: str | int,
    predictions_file: Path,
    title: str | None,
    description: str | None,
    prediction_label: str,
    source_file: str | None,
    tags: list[str] | None,
) -> list[dict[str, Any]]:
    normalized_user_id = _normalize_user_id(user_id)
    predicted_future = _load_prediction_points(predictions_file, normalized_user_id)
    upsert_model_prediction(
        normalized_user_id,
        predicted_future,
        title=title,
        description=description,
        prediction_label=prediction_label,
        source_file=source_file or str(predictions_file),
        tags=tags,
    )
    return build_trace_web_model_samples()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Python prediction results to Trace_Web model sample files."
    )
    parser.add_argument("--user-id", help="User id, for example 000 or 37.")
    parser.add_argument(
        "--predictions-file",
        help="JSON file containing a point list, an object with `predictedFuture`, or a user-id keyed object.",
    )
    parser.add_argument("--title", help="Optional sample title override.")
    parser.add_argument("--description", help="Optional sample description override.")
    parser.add_argument("--prediction-label", default="模型预测", help="Prediction label shown in the UI.")
    parser.add_argument("--source-file", help="Optional source file path shown in the UI.")
    parser.add_argument("--tag", action="append", dest="tags", help="Repeatable tag.")
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only rebuild Trace_Web/public/data/model-samples.json from existing model_predictions.json.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.build_only:
        samples = build_trace_web_model_samples()
        print(f"Rebuilt {len(samples)} model samples -> {MODEL_SAMPLES_PATH}")
        return

    if not args.user_id or not args.predictions_file:
        parser.error("`--user-id` and `--predictions-file` are required unless using `--build-only`.")

    samples = export_prediction_file(
        user_id=args.user_id,
        predictions_file=Path(args.predictions_file),
        title=args.title,
        description=args.description,
        prediction_label=args.prediction_label,
        source_file=args.source_file,
        tags=args.tags,
    )
    print(f"Exported model prediction for user {args.user_id} -> {MODEL_SAMPLES_PATH}")
    print(f"Current model sample count: {len(samples)}")


if __name__ == "__main__":
    main()
