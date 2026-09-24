"""作业查询的稳定键集游标。

游标是不透明 token，内部记录上一页最后一条作业的业务排序键
（``timestamp_start``，作业开始时间）、不可变标识（``id``，用于消除同一秒
并列）、排序方向，以及发起查询时的全部筛选条件。token 带 SHA-256 摘要，
被篡改、损坏，或换了筛选条件/排序方向再次使用时都会抛出 :class:`CursorError`，
接口层据此返回明确的 400 错误。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any

CURSOR_VERSION = "operation-keyset-1"


class CursorError(ValueError):
    """游标缺失、损坏，或与当前查询条件/排序方向不一致。"""


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def encode_cursor(
    *,
    started_at: datetime,
    operation_id: int,
    order: str,
    filters: dict[str, Any],
) -> str:
    """根据一页最后一条记录生成下一页游标。"""
    if started_at.tzinfo is None:
        # 数据库中以 UTC 墙钟时间（naive）存储，缺时区时按 UTC 解释。
        started_at = started_at.replace(tzinfo=timezone.utc)
    payload: dict[str, Any] = {
        "v": CURSOR_VERSION,
        "at": started_at.astimezone(timezone.utc).isoformat(),
        "id": int(operation_id),
        "order": order,
        "q": filters,
    }
    sealed = dict(payload)
    sealed["d"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return base64.urlsafe_b64encode(_canonical(sealed)).decode("ascii").rstrip("=")


def decode_cursor(
    token: str,
    *,
    filters: dict[str, Any],
    order: str,
) -> tuple[datetime, int]:
    """校验并解析游标，返回 (UTC 业务开始时间, 作业ID)。

    filters / order 必须与本次请求一致，否则抛出 :class:`CursorError`。
    """
    try:
        padded = token + "=" * (-len(token) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
        if not isinstance(payload, dict) or "d" not in payload:
            raise CursorError("游标内容不完整")

        digest = payload.pop("d")
        if not isinstance(digest, str) or not hmac.compare_digest(
            hashlib.sha256(_canonical(payload)).hexdigest(), digest
        ):
            raise CursorError("游标已损坏或被篡改")

        if payload.get("v") != CURSOR_VERSION:
            raise CursorError("游标版本不受支持，请从第一页重新开始")
        if payload.get("order") != order:
            raise CursorError(
                f"游标排序方向为 {payload.get('order')}，与本次请求 {order} 不一致"
            )
        if payload.get("q") != filters:
            raise CursorError("游标绑定的筛选条件与本次请求不一致，请从第一页重新开始")

        started_at = datetime.fromisoformat(payload["at"])
        if started_at.tzinfo is None:
            raise CursorError("游标时间缺少时区信息")
        return started_at.astimezone(timezone.utc), int(payload["id"])
    except CursorError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CursorError("游标格式无效或已损坏，请从第一页重新开始") from exc


def to_utc_naive(value: datetime) -> datetime:
    """把写入数据库的业务时间归一化为 naive UTC。

    SQLite 列里存的是 UTC 墙钟文本，键集比较时游标锚点也使用 naive UTC，
    避免不同时区偏移绑定成墙钟字符串后错位。
    """
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value
