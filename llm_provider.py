import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib import error, request


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: str | None
    base_url: str | None = None
    temperature: float = 0.2
    max_tokens: int = 512
    timeout: int = 90


DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-3-5-sonnet-latest",
    "mock": "mock-heuristic",
}


def _parse_json_like_records(value: str) -> list[dict[str, Any]]:
    value = value.strip()
    if not value:
        return []

    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except Exception:
        pass

    records = []
    for line in value.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def _extract_section(text: str, start_marker: str, end_marker: str | None = None) -> str:
    if start_marker not in text:
        return ""

    section = text.split(start_marker, 1)[1]
    if end_marker and end_marker in section:
        section = section.split(end_marker, 1)[0]
    return section.strip()


def _extract_latest_user_text(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _extract_system_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(str(message.get("content", "")) for message in messages if message.get("role") == "system").strip()


def _choose_grid_index_with_mock(user_text: str) -> str:
    trajectory_section = _extract_section(user_text, "Trajectory point(data1):", "Current grid data(data2):")
    grid_section = _extract_section(user_text, "Current grid data(data2):", "Current grid total size(data3):")
    trajectory_records = _parse_json_like_records(trajectory_section)
    grid_records = _parse_json_like_records(grid_section)

    if not grid_records:
        return "0"

    last_coordinates = None
    if trajectory_records:
        last_coordinates = trajectory_records[-1].get("coordinates")

    if not isinstance(last_coordinates, list) or len(last_coordinates) < 2:
        first_index = grid_records[0].get("index", 0)
        return str(first_index)

    target_x = float(last_coordinates[0])
    target_y = float(last_coordinates[1])
    best_index = grid_records[0].get("index", 0)
    best_distance = float("inf")

    for grid in grid_records:
        try:
            distance = (float(grid["X"]) - target_x) ** 2 + (float(grid["Y"]) - target_y) ** 2
        except Exception:
            continue
        if distance < best_distance:
            best_distance = distance
            best_index = grid.get("index", best_index)

    return str(best_index)


def _predict_next_point_with_mock(user_text: str) -> str:
    trajectory_match = re.search(r"trajectory point\s+(.*?)\s+in grid", user_text, re.IGNORECASE | re.DOTALL)
    trajectory_section = trajectory_match.group(1).strip() if trajectory_match else ""
    trajectory_records = _parse_json_like_records(trajectory_section)

    if len(trajectory_records) >= 2:
        try:
            last_point = trajectory_records[-1]["coordinates"]
            previous_point = trajectory_records[-2]["coordinates"]
            next_x = float(last_point[0]) + (float(last_point[0]) - float(previous_point[0]))
            next_y = float(last_point[1]) + (float(last_point[1]) - float(previous_point[1]))
            return json.dumps({"coordinates": [round(next_x, 3), round(next_y, 3)]}, ensure_ascii=False)
        except Exception:
            pass

    return json.dumps({"coordinates": [0.0, 0.0]}, ensure_ascii=False)


def _mock_chat_completion(messages: list[dict[str, str]]) -> str:
    user_text = _extract_latest_user_text(messages)
    if "Current grid data(data2):" in user_text:
        return _choose_grid_index_with_mock(user_text)
    if "Predict the next position for trajectory point" in user_text:
        return _predict_next_point_with_mock(user_text)
    return "0"


def resolve_llm_config(provider: str | None = None, model: str | None = None) -> LLMConfig:
    provider = (provider or os.getenv("TRACEPREDICT_LLM_PROVIDER", "")).strip().lower()
    if not provider:
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("GEMINI_API_KEY"):
            provider = "gemini"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            provider = "mock"

    if provider not in DEFAULT_MODELS:
        raise ValueError(f"Unsupported provider: {provider}")

    api_key = None
    base_url = None
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")

    return LLMConfig(
        provider=provider,
        model=model or os.getenv("TRACEPREDICT_LLM_MODEL") or DEFAULT_MODELS[provider],
        api_key=api_key,
        base_url=base_url,
        temperature=float(os.getenv("TRACEPREDICT_LLM_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("TRACEPREDICT_LLM_MAX_TOKENS", "512")),
        timeout=int(os.getenv("TRACEPREDICT_LLM_TIMEOUT", "90")),
    )


def _openai_chat_completion(config: LLMConfig, messages: list[dict[str, str]]) -> str:
    from openai import OpenAI

    client_kwargs = {}
    if config.api_key:
        client_kwargs["api_key"] = config.api_key
    if config.base_url:
        client_kwargs["base_url"] = config.base_url

    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
    )
    return response.choices[0].message.content or ""


def _http_post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int) -> dict[str, Any]:
    request_data = json.dumps(payload).encode("utf-8")
    request_obj = request.Request(url, data=request_data, headers=headers, method="POST")
    try:
        with request.urlopen(request_obj, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def _gemini_chat_completion(config: LLMConfig, messages: list[dict[str, str]]) -> str:
    if not config.api_key:
        raise RuntimeError("GEMINI_API_KEY is missing.")

    system_text = _extract_system_text(messages)
    contents = []
    for message in messages:
        role = message.get("role", "user")
        if role == "system":
            continue
        mapped_role = "model" if role == "assistant" else "user"
        contents.append({"role": mapped_role, "parts": [{"text": str(message.get("content", ""))}]})

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": config.temperature,
            "maxOutputTokens": config.max_tokens,
        },
    }
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.model}:generateContent?key={config.api_key}"
    response = _http_post_json(url, payload, {"Content-Type": "application/json"}, config.timeout)

    candidates = response.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {response}")

    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts).strip()


def _anthropic_chat_completion(config: LLMConfig, messages: list[dict[str, str]]) -> str:
    if not config.api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is missing.")

    system_text = _extract_system_text(messages)
    anthropic_messages = []
    for message in messages:
        role = message.get("role", "user")
        if role == "system":
            continue
        anthropic_messages.append(
            {
                "role": "assistant" if role == "assistant" else "user",
                "content": [{"type": "text", "text": str(message.get("content", ""))}],
            }
        )

    payload = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "messages": anthropic_messages,
    }
    if system_text:
        payload["system"] = system_text

    response = _http_post_json(
        "https://api.anthropic.com/v1/messages",
        payload,
        {
            "Content-Type": "application/json",
            "x-api-key": config.api_key,
            "anthropic-version": "2023-06-01",
        },
        config.timeout,
    )

    content_blocks = response.get("content") or []
    return "".join(block.get("text", "") for block in content_blocks if isinstance(block, dict)).strip()


def chat_completion(messages: list[dict[str, str]], provider: str | None = None, model: str | None = None) -> str:
    config = resolve_llm_config(provider=provider, model=model)
    if config.provider == "mock":
        return _mock_chat_completion(messages)
    if config.provider == "openai":
        return _openai_chat_completion(config, messages)
    if config.provider == "gemini":
        return _gemini_chat_completion(config, messages)
    if config.provider == "anthropic":
        return _anthropic_chat_completion(config, messages)
    raise ValueError(f"Unsupported provider: {config.provider}")


def current_provider_summary(provider: str | None = None, model: str | None = None) -> str:
    config = resolve_llm_config(provider=provider, model=model)
    return f"{config.provider}:{config.model}"
