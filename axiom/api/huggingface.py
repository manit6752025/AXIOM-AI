from __future__ import annotations

from typing import Any


def search_models(query: str, limit: int = 10) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi

    api = HfApi()
    return [
        {
            "id": model.id,
            "downloads": getattr(model, "downloads", None),
            "likes": getattr(model, "likes", None),
            "pipeline_tag": getattr(model, "pipeline_tag", None),
        }
        for model in api.list_models(
            search=query,
            limit=max(1, min(int(limit), 50)),
        )
    ]
