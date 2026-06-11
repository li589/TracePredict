import json
from pathlib import Path

from generate_actual_samples import OUTPUT_PATH as ACTUAL_OUTPUT_PATH


TRACE_WEB_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = TRACE_WEB_ROOT / "data" / "model_predictions.json"
OUTPUT_PATH = TRACE_WEB_ROOT / "public" / "data" / "model-samples.json"


def load_actual_samples() -> list[dict[str, object]]:
    if not ACTUAL_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing actual sample file: {ACTUAL_OUTPUT_PATH}. "
            "Run `python scripts/generate_actual_samples.py` first."
        )

    with ACTUAL_OUTPUT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_prediction_overrides() -> dict[str, dict[str, object]]:
    if not INPUT_PATH.exists():
        return {}

    with INPUT_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        return payload

    raise ValueError("model_predictions.json must be a JSON object keyed by user id.")


def export_model_samples() -> list[dict[str, object]]:
    actual_samples = load_actual_samples()
    overrides = load_prediction_overrides()
    exported_samples: list[dict[str, object]] = []

    for sample in actual_samples:
        user_id = str(sample["userId"])
        override = overrides.get(user_id)
        if not override:
            continue

        predicted_future = override.get("predictedFuture")
        if not isinstance(predicted_future, list) or not predicted_future:
            continue

        converted = dict(sample)
        converted["id"] = f"model-{user_id}"
        converted["title"] = override.get("title") or f"模型输出 {user_id}"
        converted["description"] = override.get("description") or (
            f"基于真实样本 {user_id} 叠加的模型输出轨迹。"
        )
        converted["predictedFuture"] = predicted_future
        converted["predictionLabel"] = override.get("predictionLabel") or "模型预测"
        converted["sampleType"] = "actual"
        converted["tags"] = list(
            dict.fromkeys(
                [
                    "模型输出",
                    *(sample.get("tags") or []),
                    *(override.get("tags") or []),
                ]
            )
        )
        if override.get("sourceFile"):
            converted["sourceFile"] = override["sourceFile"]

        exported_samples.append(converted)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(exported_samples, handle, ensure_ascii=False, indent=2)

    return exported_samples


def main() -> None:
    samples = export_model_samples()
    print(f"Generated {len(samples)} model samples -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
