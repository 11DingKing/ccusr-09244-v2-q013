"""作业查询的稳定键集游标。

数据分析任务按页读取作业时，页码（offset）分页会在数据持续写入后重复或漏掉
记录。键集游标以业务开始时间 ``timestamp_start`` 为主排序键、不可变的作业主键
``id`` 消除同一秒内的并列，并把筛选条件与排序方向封进带签名的令牌中：

- 继续读取时若客户端修改了筛选条件或排序方向，直接报错而不是返回语义错误的数据；
- 令牌损坏、被篡改或过期（版本不兼容）时同样抛出 :class:`CursorError`；
- 迟到数据插入在游标之前的位置时不会让已发出的页面倒退。
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Mapping

CURSOR_VERSION = 1
SORT_ORDERS = ("asc", "desc")


class CursorError(ValueError):
    """游标损坏、版本不兼容，或与当前筛选条件/排序方向不匹配。"""


def normalize_filters(filters: Mapping[str, Any]) -> dict[str, Any]:
    """把筛选条件整理成键序稳定的字典，作为游标的绑定上下文。"""
    return {key: filters[key] for key in sorted(filters)}


def _normalize_at(at: datetime) -> str:
    if at.tzinfo is None:
        # 历史数据可能写入无时区时间，统一按 UTC 解释。
        at = at.replace(tzinfo=timezone.utc)
    return at.astimezone(timezone.utc).isoformat()


def _serialize(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _seal(payload: dict[str, Any]) -> str:
    payload = dict(payload)
    payload["digest"] = sha256(_serialize(payload)).hexdigest()
    encoded = base64.urlsafe_b64encode(_serialize(payload)).decode()
    return encoded.rstrip("=")


def encode_cursor(
    *,
    at: datetime,
    operation_id: int,
    sort_order: str,
    filters: Mapping[str, Any],
) -> str:
    """根据本页最后一条记录生成下一页游标。"""
    if sort_order not in SORT_ORDERS:
        raise CursorError(f"排序方向无效: {sort_order!r}")
    payload = {
        "v": CURSOR_VERSION,
        "at": _normalize_at(at),
        "id": int(operation_id),
        "dir": sort_order,
        "f": normalize_filters(filters),
    }
    return _seal(payload)


def decode_cursor(
    token: str,
    *,
    sort_order: str,
    filters: Mapping[str, Any],
) -> tuple[datetime, int]:
    """校验并解析游标，返回 ``(timestamp_start, operation_id)``。

    令牌的签名、版本、绑定的筛选条件或排序方向任一不匹配都会抛出
    :class:`CursorError`，调用方应当让客户端从第一页重新开始读取。
    """
    if sort_order not in SORT_ORDERS:
        raise CursorError(f"排序方向无效: {sort_order!r}")
    try:
        padded = token + "=" * (-len(token) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
        digest = payload.pop("digest")
    except CursorError:
        raise
    except Exception as exc:  # base64/json 解码失败、缺字段等
        raise CursorError("游标已损坏，请清空游标后从第一页重新读取") from exc

    if not isinstance(payload, dict) or not isinstance(digest, str):
        raise CursorError("游标已损坏，请清空游标后从第一页重新读取")
    if sha256(_serialize(payload)).hexdigest() != digest:
        raise CursorError("游标签名校验失败，游标可能已被篡改或损坏")
    if payload.get("v") != CURSOR_VERSION:
        raise CursorError("游标版本不兼容，请从第一页重新开始读取")
    if payload.get("dir") != sort_order:
        raise CursorError(
            f"游标绑定的排序方向为 {payload.get('dir')!r}，当前请求为 {sort_order!r}，"
            "修改排序方向后必须从第一页重新开始读取"
        )
    if payload.get("f") != normalize_filters(filters):
        raise CursorError("游标绑定的筛选条件与当前请求不一致，修改筛选条件后必须从第一页重新开始读取")

    try:
        at = datetime.fromisoformat(payload["at"])
        operation_id = int(payload["id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CursorError("游标已损坏，请清空游标后从第一页重新读取") from exc
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    return at.astimezone(timezone.utc), operation_id
