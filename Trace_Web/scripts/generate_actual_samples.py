import csv
import json
import math
from datetime import datetime, timedelta
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_WEB_ROOT = REPO_ROOT / "Trace_Web"
CLASSIFICATION_PATH = REPO_ROOT / "Final" / "Cluster" / "json" / "classification_results.json"
TRAJECTORY_DIR = REPO_ROOT / "Final" / "POI_relevance" / "Probability"
OUTPUT_PATH = TRACE_WEB_ROOT / "public" / "data" / "actual-samples.json"
PREFERRED_USERS = ["000", "037", "104", "111"]
OBSERVED_COUNT = 5
FUTURE_COUNT = 4
WINDOW_SIZE = OBSERVED_COUNT + FUTURE_COUNT


def load_cluster_mapping() -> dict[str, str]:
    with CLASSIFICATION_PATH.open("r", encoding="utf-8") as handle:
        raw_clusters = json.load(handle)

    user_to_cluster: dict[str, str] = {}
    for cluster_name, members in raw_clusters.items():
        for member in members:
            user_to_cluster[f"{int(member):03d}"] = f"Cluster {cluster_name}"
    return user_to_cluster


def parse_timestamp(row: dict[str, str]) -> datetime:
    return datetime.strptime(f"{row['date']} {row['time']}", "%Y-%m-%d %H:%M:%S")


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    earth_radius = 6371000
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lng = math.radians(lng2 - lng1)
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng / 2) ** 2
    )
    return 2 * earth_radius * math.asin(math.sqrt(a))


def read_rows(csv_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            try:
                lat = float(raw["latitude"])
                lng = float(raw["longitude"])
            except (KeyError, TypeError, ValueError):
                continue

            rows.append(
                {
                    "lat": lat,
                    "lng": lng,
                    "date": raw.get("date", ""),
                    "time": raw.get("time", ""),
                    "people_id": str(raw.get("people_id", "")).zfill(3),
                    "timestamp": parse_timestamp(raw),
                }
            )
    return rows


def score_window(window: list[dict[str, object]]) -> tuple[float, int]:
    total_distance = 0.0
    max_gap_seconds = 0

    for index in range(1, len(window)):
        previous = window[index - 1]
        current = window[index]
        total_distance += haversine_distance(
            previous["lat"], previous["lng"], current["lat"], current["lng"]
        )
        gap_seconds = int((current["timestamp"] - previous["timestamp"]).total_seconds())
        max_gap_seconds = max(max_gap_seconds, gap_seconds)

    # Prefer longer movement with tighter time continuity.
    score = total_distance - max_gap_seconds * 0.4
    return score, max_gap_seconds


def select_window(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if len(rows) < WINDOW_SIZE:
        return []

    best_window: list[dict[str, object]] = []
    best_score = -float("inf")

    for start in range(0, len(rows) - WINDOW_SIZE + 1):
        window = rows[start : start + WINDOW_SIZE]
        _, max_gap_seconds = score_window(window)
        if max_gap_seconds > 20 * 60:
            continue

        score, _ = score_window(window)
        if score > best_score:
            best_score = score
            best_window = window

    if best_window:
        return best_window

    # Fallback to the first available window if the data is too sparse.
    return rows[:WINDOW_SIZE]


def format_point(row: dict[str, object]) -> dict[str, object]:
    return {
        "lat": round(float(row["lat"]), 6),
        "lng": round(float(row["lng"]), 6),
        "timestamp": row["timestamp"].strftime("%m-%d %H:%M"),
    }


def project_future(observed: list[dict[str, object]], future_count: int) -> list[dict[str, object]]:
    if len(observed) < 2:
        return observed[:future_count]

    recent_steps = []
    for index in range(1, len(observed)):
        previous = observed[index - 1]
        current = observed[index]
        recent_steps.append(
            (
                float(current["lat"]) - float(previous["lat"]),
                float(current["lng"]) - float(previous["lng"]),
            )
        )

    tail = recent_steps[-min(3, len(recent_steps)) :]
    avg_lat_step = sum(step[0] for step in tail) / len(tail)
    avg_lng_step = sum(step[1] for step in tail) / len(tail)

    start_time = observed[-1]["timestamp"]
    previous_time = observed[-2]["timestamp"]
    avg_seconds = max(int((start_time - previous_time).total_seconds()), 60)

    predicted = []
    current_lat = float(observed[-1]["lat"])
    current_lng = float(observed[-1]["lng"])
    current_time = start_time

    for step_index in range(future_count):
        damping = 0.96 ** step_index
        current_lat += avg_lat_step * damping
        current_lng += avg_lng_step * damping
        current_time = current_time + timedelta(seconds=avg_seconds)
        predicted.append(
            {
                "lat": round(current_lat, 6),
                "lng": round(current_lng, 6),
                "timestamp": current_time.strftime("%m-%d %H:%M"),
            }
        )

    return predicted


def build_sample(user_id: str, cluster_mapping: dict[str, str]) -> dict[str, object] | None:
    csv_path = TRAJECTORY_DIR / f"{user_id}.csv"
    if not csv_path.exists():
        return None

    rows = read_rows(csv_path)
    window = select_window(rows)
    if len(window) < WINDOW_SIZE:
        return None

    observed_rows = window[:OBSERVED_COUNT]
    actual_rows = window[OBSERVED_COUNT:]

    observed = [format_point(row) for row in observed_rows]
    actual_future = [format_point(row) for row in actual_rows]
    predicted_future = project_future(observed_rows, len(actual_rows))

    start_time = observed_rows[0]["timestamp"].strftime("%Y-%m-%d %H:%M")
    end_time = actual_rows[-1]["timestamp"].strftime("%Y-%m-%d %H:%M")
    description = (
        f"从真实轨迹文件 {user_id}.csv 抽取的连续轨迹片段，时间范围 {start_time} - {end_time}。"
        " 橙色虚线为前端基线预测，用于展示真实样本接入流程。"
    )

    return {
        "id": f"actual-{user_id}",
        "title": f"真实样本 {user_id}",
        "userId": user_id,
        "cluster": cluster_mapping.get(user_id, "Unknown"),
        "description": description,
        "observed": observed,
        "actualFuture": actual_future,
        "predictedFuture": predicted_future,
        "tags": ["实际样本", "仓库真实 CSV", "基线预测"],
        "sampleType": "actual",
        "predictionLabel": "基线预测",
        "sourceFile": str(csv_path.relative_to(REPO_ROOT)).replace("\\", "/"),
    }


def main() -> None:
    cluster_mapping = load_cluster_mapping()
    samples = []

    for user_id in PREFERRED_USERS:
        sample = build_sample(user_id, cluster_mapping)
        if sample:
            samples.append(sample)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(samples, handle, ensure_ascii=False, indent=2)

    print(f"Generated {len(samples)} actual samples -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
