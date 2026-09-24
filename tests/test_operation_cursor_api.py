"""作业查询稳定游标模式的接口验证。

覆盖：同秒并列、反向遍历、组合筛选、读取过程中删除/迟到数据、
重启恢复、游标损坏与条件被修改的明确报错，以及旧页码模式的兼容行为。
"""

from datetime import timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.routers import common, operation


@pytest.fixture()
def ctx(tmp_path):
    db_path = tmp_path / "cursor_test.db"
    url = f"sqlite:///{db_path}"

    def make_engine():
        engine = create_engine(url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=engine)
        return engine, sessionmaker(autocommit=False, autoflush=False, bind=engine)

    engine, Session = make_engine()

    def build_app(session_cls):
        application = FastAPI()
        application.include_router(common.router, prefix="/api/v1")
        application.include_router(operation.router, prefix="/api/v1")

        def override_get_db():
            db = session_cls()
            try:
                yield db
            finally:
                db.close()

        application.dependency_overrides[get_db] = override_get_db
        return application

    client = TestClient(build_app(Session))
    model = client.post("/api/v1/robot-models", json={"name": "机型A", "manufacturer": "厂牌A"}).json()
    scene_a = client.post("/api/v1/scenes", json={"name": "场景A", "category": "制造"}).json()
    scene_b = client.post("/api/v1/scenes", json={"name": "场景B", "category": "零售"}).json()
    skill = client.post("/api/v1/skills", json={"name": "技能A", "category": "抓取"}).json()

    def build_fresh_client():
        """模拟服务重启：全新引擎/会话绑定同一个数据库文件，只凭游标恢复。"""
        _, fresh_session = make_engine()
        return TestClient(build_app(fresh_session))

    yield {
        "client": client,
        "build_fresh_client": build_fresh_client,
        "model_id": model["id"],
        "scene_a": scene_a["id"],
        "scene_b": scene_b["id"],
        "skill_id": skill["id"],
    }

    engine.dispose()


def make_op(client, ctx, start, scene_id=None, serial=None, idx=0):
    end = (datetime_from_iso(start) + timedelta(seconds=30)).isoformat()
    payload = {
        "robot_model_id": ctx["model_id"],
        "scene_id": scene_id or ctx["scene_a"],
        "skill_id": ctx["skill_id"],
        "robot_serial": serial,
        "motion_trajectory": {"points": [idx]},
        "perception_records": {"frames": [idx]},
        "timestamp_start": start,
        "timestamp_end": end,
    }
    resp = client.post("/api/v1/operations", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


def datetime_from_iso(value):
    from datetime import datetime
    return datetime.fromisoformat(value)


def read_all_cursor(client, params, stop_after=None):
    """从第一页开始按游标读完，返回 (keys, pages, responses, next_cursor)。"""
    keys, pages, responses = [], [], []
    cursor = None
    rounds = 0
    while True:
        q = dict(params)
        if cursor:
            q["cursor"] = cursor
        resp = client.get("/api/v1/operations", params=q)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        responses.append(body)
        page_keys = [(item["timestamp_start"], item["id"]) for item in body["items"]]
        keys.extend(page_keys)
        pages.append(page_keys)
        rounds += 1
        if stop_after and rounds >= stop_after:
            return keys, pages, responses, body["next_cursor"]
        if not body["has_next"]:
            assert body["next_cursor"] is None
            return keys, pages, responses, None
        cursor = body["next_cursor"]


def drain(client, params, cursor):
    keys = []
    while cursor:
        body = client.get("/api/v1/operations", params={**params, "cursor": cursor}).json()
        keys.extend((item["timestamp_start"], item["id"]) for item in body["items"])
        cursor = body["next_cursor"]
    return keys


def test_same_second_records_are_stable_and_complete(ctx):
    client = ctx["client"]
    ids = [make_op(client, ctx, "2026-03-01T10:00:00+00:00", idx=i)["id"] for i in range(7)]

    keys, pages, responses, _ = read_all_cursor(
        client, {"mode": "cursor", "sort_order": "asc", "page_size": 3}
    )

    # 同秒记录以不可变 id 消歧，严格递增、无重复、无遗漏。
    assert keys == sorted(keys)
    assert [key[1] for key in keys] == sorted(ids)
    assert len(set(keys)) == 7
    assert pages[0] and len(pages[0]) == 3
    assert responses[0]["mode"] == "cursor"
    assert responses[0]["sort_order"] == "asc"
    assert responses[0]["has_next"] is True
    assert isinstance(responses[0]["next_cursor"], str)


def test_reverse_traversal_desc_and_asc_are_mirror(ctx):
    client = ctx["client"]
    for i in range(6):
        make_op(client, ctx, f"2026-03-01T10:00:{i:02d}+00:00", idx=i)

    asc_keys, _, _, _ = read_all_cursor(
        client, {"mode": "cursor", "sort_order": "asc", "page_size": 2}
    )
    desc_keys, desc_pages, _, _ = read_all_cursor(
        client, {"mode": "cursor", "sort_order": "desc", "page_size": 2}
    )

    assert asc_keys == list(reversed(desc_keys))
    # 反向遍历时页面内与跨页都不能倒退（业务时间为主，id 消除同秒并列）。
    for page in desc_pages:
        assert page == sorted(page, reverse=True)
    for prev, nxt in zip(desc_pages, desc_pages[1:]):
        assert prev[-1] > nxt[0]


def test_combined_filters_bind_cursor(ctx):
    client = ctx["client"]
    # 场景A 4 条，其中 2 条标注成功；场景B 3 条必须被过滤掉。
    a_ids = [
        make_op(client, ctx, f"2026-03-01T09:00:0{i}+00:00", scene_id=ctx["scene_a"], idx=i)["id"]
        for i in range(4)
    ]
    for i in range(3):
        make_op(client, ctx, f"2026-03-01T09:00:0{i}+00:00", scene_id=ctx["scene_b"], idx=10 + i)
    for op_id in a_ids[:2]:
        resp = client.post("/api/v1/annotations", json={
            "operation_data_id": op_id, "is_success": True, "annotator": "tester"
        })
        assert resp.status_code == 200, resp.text

    params = {
        "mode": "cursor", "sort_order": "asc", "page_size": 1,
        "scene_id": ctx["scene_a"], "is_annotated": True,
    }
    keys, _, _, _ = read_all_cursor(client, params)
    assert sorted(key[1] for key in keys) == sorted(a_ids[:2])


def test_continue_after_delete_both_ahead_and_at_cursor(ctx):
    client = ctx["client"]
    ids = [make_op(client, ctx, f"2026-03-02T08:00:0{i}+00:00", idx=i)["id"] for i in range(6)]

    params = {"mode": "cursor", "sort_order": "asc", "page_size": 2}
    _, pages, _, cursor = read_all_cursor(client, params, stop_after=1)
    first_page_ids = {key[1] for key in pages[0]}

    # 删除游标指向的那条（第一页最后一条）和一条更靠后的记录。
    cursor_anchor = pages[0][-1][1]
    later_victim = [i for i in ids if i not in first_page_ids][0]
    for victim in (cursor_anchor, later_victim):
        assert client.delete(f"/api/v1/operations/{victim}").status_code == 200

    # 游标仍可继续：锚点记录被删除不影响键集谓词，已读不重复，剩余不遗漏。
    rest_keys = drain(client, params, cursor)
    rest_ids = {key[1] for key in rest_keys}
    expected = set(ids) - {cursor_anchor, later_victim} - first_page_ids
    assert rest_ids == expected


def test_late_arriving_data_does_not_move_pages_backwards(ctx):
    client = ctx["client"]
    for i in range(4):
        make_op(client, ctx, f"2026-03-03T12:00:{10 + i:02d}+00:00", idx=i)

    params = {"mode": "cursor", "sort_order": "asc", "page_size": 2}
    _, pages, _, cursor = read_all_cursor(client, params, stop_after=1)
    page_one_max = pages[0][-1]

    # 迟到数据：业务开始时间落在游标之前，以及一条在游标之后的正常新数据。
    late_id = make_op(client, ctx, "2026-03-03T12:00:00+00:00", idx=99)["id"]
    fresh_id = make_op(client, ctx, "2026-03-03T12:00:30+00:00", idx=100)["id"]

    rest_keys = drain(client, params, cursor)

    # 第二页起点不得早于第一页终点——已发出的顺序不会倒退。
    assert rest_keys[0] > page_one_max
    rest_ids = {key[1] for key in rest_keys}
    assert late_id not in rest_ids  # 迟到记录在水位线之前，不会插进后续页面
    assert fresh_id in rest_ids    # 新于游标的数据照常读到


def test_restart_resumes_with_stored_cursor(ctx):
    client = ctx["client"]
    ids = [make_op(client, ctx, f"2026-03-04T09:00:0{i}+00:00", idx=i)["id"] for i in range(5)]

    _, _, responses, cursor = read_all_cursor(
        client, {"mode": "cursor", "sort_order": "asc", "page_size": 2}, stop_after=1
    )
    seen = {item["id"] for item in responses[0]["items"]}

    # 服务重启：全新客户端、全新会话，只凭落盘的游标字符串继续。
    restarted = ctx["build_fresh_client"]()
    params = {"mode": "cursor", "sort_order": "asc", "page_size": 2}
    next_cursor = cursor
    while next_cursor:
        resp = restarted.get("/api/v1/operations", params={**params, "cursor": next_cursor})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        seen |= {item["id"] for item in body["items"]}
        next_cursor = body["next_cursor"]
    assert seen == set(ids)


def test_corrupted_cursor_returns_clear_error(ctx):
    client = ctx["client"]
    make_op(client, ctx, "2026-03-05T10:00:00+00:00")
    make_op(client, ctx, "2026-03-05T10:00:01+00:00")

    for bad in ("not-a-cursor", "YWJj===%%%"):
        resp = client.get("/api/v1/operations", params={
            "mode": "cursor", "page_size": 1, "cursor": bad
        })
        assert resp.status_code == 400
        assert "游标" in resp.json()["detail"]

    # 取一个合法游标后篡改其内容，签名必须失效。
    import base64
    import json
    body = client.get("/api/v1/operations", params={"mode": "cursor", "page_size": 1}).json()
    token = body["next_cursor"]
    raw = json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))
    raw["id"] = raw["id"] + 999
    tampered = base64.urlsafe_b64encode(json.dumps(raw).encode()).decode().rstrip("=")
    resp = client.get("/api/v1/operations", params={
        "mode": "cursor", "page_size": 1, "cursor": tampered
    })
    assert resp.status_code == 400
    assert "游标" in resp.json()["detail"]


def test_changing_filters_or_direction_rejects_cursor(ctx):
    client = ctx["client"]
    for i in range(3):
        make_op(client, ctx, f"2026-03-06T10:00:0{i}+00:00", scene_id=ctx["scene_a"], idx=i)
        make_op(client, ctx, f"2026-03-06T10:00:0{i}+00:00", scene_id=ctx["scene_b"], idx=i)

    token = client.get("/api/v1/operations", params={
        "mode": "cursor", "page_size": 1, "sort_order": "asc", "scene_id": ctx["scene_a"]
    }).json()["next_cursor"]

    # 修改筛选条件
    resp = client.get("/api/v1/operations", params={
        "mode": "cursor", "page_size": 1, "sort_order": "asc",
        "scene_id": ctx["scene_b"], "cursor": token
    })
    assert resp.status_code == 400
    assert "筛选条件" in resp.json()["detail"]

    # 修改排序方向
    resp = client.get("/api/v1/operations", params={
        "mode": "cursor", "page_size": 1, "sort_order": "desc",
        "scene_id": ctx["scene_a"], "cursor": token
    })
    assert resp.status_code == 400
    assert "排序方向" in resp.json()["detail"]


def test_legacy_page_mode_keeps_old_behavior(ctx):
    client = ctx["client"]
    for i in range(5):
        make_op(client, ctx, f"2026-03-07T10:00:0{i}+00:00", idx=i)

    first = client.get("/api/v1/operations", params={"page": 1, "page_size": 2}).json()
    assert first["mode"] == "page"
    assert first["page"] == 1
    assert first["page_size"] == 2
    assert first["total"] == 5
    assert first["has_next"] is None
    assert first["next_cursor"] is None
    assert first["sort_order"] is None

    second = client.get("/api/v1/operations", params={"page": 2, "page_size": 2}).json()
    first_ids = {item["id"] for item in first["items"]}
    assert {item["id"] for item in second["items"]}.isdisjoint(first_ids)


def test_timezone_offsets_normalize_to_same_instant(ctx):
    client = ctx["client"]
    # 同一时刻分别用 UTC 与 +08:00 表达，落库字符串必须一致并按 id 消歧。
    id1 = make_op(client, ctx, "2026-03-08T08:00:00+08:00", idx=1)["id"]
    id2 = make_op(client, ctx, "2026-03-08T00:00:00+00:00", idx=2)["id"]
    body = client.get("/api/v1/operations", params={
        "mode": "cursor", "sort_order": "asc", "page_size": 10
    }).json()
    starts = {item["id"]: item["timestamp_start"] for item in body["items"]}
    assert starts[id1] == starts[id2]
    assert [item["id"] for item in body["items"]] == sorted([id1, id2])
