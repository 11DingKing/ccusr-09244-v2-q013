"""作业查询稳定游标分页的接口验证。

覆盖：同秒并列、反向遍历、组合筛选、删除后继续读取、迟到数据不回退、
游标损坏/条件变更报错、重启恢复，以及旧页码分页的兼容行为。
"""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.models import Annotation, OperationData, RobotModel, Scene, Skill
from app.routers import operation as operation_router

API = "/api/v1/operations"
T0 = datetime(2026, 1, 1, 0, 0, 0)


def _make_op(op_id, ts, *, model=1, scene=1, skill=1, serial="SN-1", grade="A"):
    return OperationData(
        id=op_id,
        robot_model_id=model,
        scene_id=scene,
        skill_id=skill,
        robot_serial=serial,
        data_grade=grade,
        motion_trajectory={"wp": []},
        perception_records={"cam": 0},
        timestamp_start=ts,
        timestamp_end=ts + timedelta(seconds=10),
    )


@pytest.fixture()
def db_env(tmp_path):
    db_path = tmp_path / "cursor_test.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add_all([
        RobotModel(id=1, name="RM-A", manufacturer="m"),
        RobotModel(id=2, name="RM-B", manufacturer="m"),
        Scene(id=1, name="场景甲", category="c"),
        Scene(id=2, name="场景乙", category="c"),
        Skill(id=1, name="技能甲", category="c"),
        Skill(id=2, name="技能乙", category="c"),
    ])
    session.commit()

    def client_for():
        app = FastAPI()
        app.include_router(operation_router.router, prefix="/api/v1")

        def _get_db():
            db = Session()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = _get_db
        return TestClient(app)

    yield {
        "session": session,
        "client_for": client_for,
        "db_path": db_path,
        "engine": engine,
    }
    session.close()
    engine.dispose()


def _walk(client, page_size, order="desc", **params):
    """从头遍历游标分页，返回 (id序列, 每页序列, 首页游标之后的最后一个游标)。"""
    seen_pages = []
    cursor = None
    last_cursor = None
    while True:
        q = {"page_size": page_size, "order": order, **params}
        if cursor is not None:
            q["cursor"] = cursor
        resp = client.get(API, params=q)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["mode"] == "cursor"
        assert body["order"] == order
        page_ids = [item["id"] for item in body["items"]]
        seen_pages.append(page_ids)
        if not body["has_next"]:
            assert body["next_cursor"] is None
            break
        assert body["next_cursor"]
        cursor = body["next_cursor"]
        last_cursor = cursor
    return [i for p in seen_pages for i in p], seen_pages, last_cursor


def test_same_second_records_are_tie_broken_by_id(db_env):
    session = db_env["session"]
    # 4 条业务开始时间完全相同（同一秒），依赖不可变 id 消除并列。
    session.add_all([_make_op(101, T0), _make_op(102, T0), _make_op(103, T0), _make_op(104, T0)])
    session.commit()
    client = db_env["client_for"]()

    desc_ids, pages, _ = _walk(client, 2, order="desc")
    assert desc_ids == [104, 103, 102, 101]
    assert pages == [[104, 103], [102, 101]]

    asc_ids, _, _ = _walk(client, 2, order="asc")
    assert asc_ids == [101, 102, 103, 104]
    assert list(reversed(asc_ids)) == desc_ids


def test_reverse_traversal_orders_full_set(db_env):
    session = db_env["session"]
    times = [T0 + timedelta(seconds=i * 5) for i in range(5)]
    # 首尾再加两条与相邻记录同秒的，混合验证。
    session.add_all([
        _make_op(201, times[0]),
        _make_op(202, times[0]),
        _make_op(203, times[1]),
        _make_op(204, times[2]),
        _make_op(205, times[3]),
        _make_op(206, times[4]),
        _make_op(207, times[4]),
    ])
    session.commit()
    client = db_env["client_for"]()

    desc_ids, _, _ = _walk(client, 3, order="desc")
    asc_ids, _, _ = _walk(client, 3, order="asc")
    assert desc_ids == list(reversed(asc_ids)) == [207, 206, 205, 204, 203, 202, 201]


def test_combined_filters_and_cursor_binding(db_env):
    session = db_env["session"]
    for i in range(6):
        session.add(_make_op(300 + i, T0 + timedelta(minutes=i), model=1, scene=1, skill=1))
    for i in range(3):
        session.add(_make_op(400 + i, T0 + timedelta(minutes=i), model=2, scene=2, skill=2))
    # 300..303 有标注（成功），304/305 无标注。
    for op_id, success in [(300, True), (301, True), (302, False), (303, True)]:
        session.add(Annotation(
            operation_data_id=op_id, is_success=success,
            failure_category=None if success else "感知异常",
        ))
    session.commit()
    client = db_env["client_for"]()

    filters = dict(
        robot_model_id=1, scene_id=1, skill_id=1,
        is_annotated=True, is_success=True,
    )
    ids, _, cursor = _walk(client, 2, order="asc", **filters)
    assert ids == [300, 301, 303]

    # 游标继续使用时修改任一筛选条件 -> 明确报错。
    resp = client.get(API, params={
        "page_size": 2, "cursor": cursor, "order": "asc", "scene_id": 2,
    })
    assert resp.status_code == 400
    assert "筛选条件" in resp.json()["detail"]

    # 排序方向与游标不一致 -> 明确报错。
    resp = client.get(API, params={"page_size": 2, "cursor": cursor, "order": "desc", **filters})
    assert resp.status_code == 400
    assert "排序方向" in resp.json()["detail"]

    # 游标损坏 / 被篡改 -> 明确报错。
    resp = client.get(API, params={"page_size": 2, "cursor": "not-a-real-cursor"})
    assert resp.status_code == 400
    tampered = cursor[:-2] + ("aa" if not cursor.endswith("aa") else "bb")
    resp = client.get(API, params={"page_size": 2, "cursor": tampered, **filters})
    assert resp.status_code == 400


def test_continue_after_delete(db_env):
    session = db_env["session"]
    session.add_all([_make_op(500 + i, T0 + timedelta(minutes=i)) for i in range(7)])
    session.commit()
    client = db_env["client_for"]()

    resp = client.get(API, params={"page_size": 3, "order": "asc"})
    first = resp.json()
    assert [i["id"] for i in first["items"]] == [500, 501, 502]
    cursor = first["next_cursor"]
    anchor_id = first["items"][-1]["id"]

    # 删除“下一页”里的一条记录，以及游标锚点本身（上一页最后一条）。
    for doomed in (504, anchor_id):
        session.query(OperationData).filter_by(id=doomed).delete()
    session.commit()

    resp = client.get(API, params={"page_size": 3, "order": "asc", "cursor": cursor})
    assert resp.status_code == 200, resp.text
    second = [i["id"] for i in resp.json()["items"]]
    # 锚点记录不重复出现，已删除的 504 缺席，剩余记录不重不漏。
    assert anchor_id not in second
    assert 504 not in second
    rest_ids = []
    while True:
        rest_ids.extend(second)
        if not resp.json()["has_next"]:
            break
        resp = client.get(API, params={
            "page_size": 3, "order": "asc", "cursor": resp.json()["next_cursor"],
        })
        second = [i["id"] for i in resp.json()["items"]]
    assert rest_ids == [503, 505, 506]
    assert len(set(rest_ids)) == len(rest_ids)


def test_late_arriving_data_never_regresses_emitted_pages(db_env):
    session = db_env["session"]
    # 6 条，desc 遍历：最新 -> 最旧。
    session.add_all([_make_op(600 + i, T0 + timedelta(minutes=i)) for i in range(6)])
    session.commit()
    client = db_env["client_for"]()

    resp = client.get(API, params={"page_size": 2, "order": "desc"})
    page1 = [i["id"] for i in resp.json()["items"]]
    assert page1 == [605, 604]
    cursor = resp.json()["next_cursor"]

    # 迟到的“更新”记录（比首页还新）以及与锚点同秒、id 更大的记录。
    session.add_all([
        _make_op(700, T0 + timedelta(minutes=60)),
        _make_op(701, T0 + timedelta(minutes=4)),
    ])
    session.commit()

    resp = client.get(API, params={"page_size": 2, "order": "desc", "cursor": cursor})
    assert resp.status_code == 200
    page2 = [i["id"] for i in resp.json()["items"]]
    # 已发出的首页顺序不回退：新记录不插入，锚点同秒的并列记录不抢占。
    assert page2 == [603, 602]
    assert not ({700, 701} & set(page2))

    # 继续读完，已发页面不受迟到数据影响。
    seen = page1 + page2
    cur = resp.json()["next_cursor"]
    while cur:
        r = client.get(API, params={"page_size": 2, "order": "desc", "cursor": cur})
        seen.extend(i["id"] for i in r.json()["items"])
        cur = r.json()["next_cursor"]
    assert seen == [605, 604, 603, 602, 601, 600]


def test_resume_with_persisted_cursor_after_restart(db_env):
    session = db_env["session"]
    session.add_all([_make_op(800 + i, T0 + timedelta(minutes=i)) for i in range(7)])
    session.commit()

    client = db_env["client_for"]()
    resp = client.get(API, params={"page_size": 3, "order": "asc"})
    first = [i["id"] for i in resp.json()["items"]]
    saved_cursor = resp.json()["next_cursor"]
    assert first == [800, 801, 802]

    # 模拟任务重启：基于同一个数据库文件新建引擎与客户端（游标已持久化）。
    restart_engine = create_engine(
        f"sqlite:///{db_env['db_path']}", connect_args={"check_same_thread": False}
    )
    RestartSession = sessionmaker(bind=restart_engine)
    app2 = FastAPI()
    app2.include_router(operation_router.router, prefix="/api/v1")

    def _get_db():
        db = RestartSession()
        try:
            yield db
        finally:
            db.close()

    app2.dependency_overrides[get_db] = _get_db
    client2 = TestClient(app2)

    resp = client2.get(API, params={"page_size": 3, "order": "asc", "cursor": saved_cursor})
    assert resp.status_code == 200
    assert [i["id"] for i in resp.json()["items"]] == [803, 804, 805]
    assert resp.json()["has_next"] is True
    resp = client2.get(API, params={
        "page_size": 3, "order": "asc", "cursor": resp.json()["next_cursor"],
    })
    assert [i["id"] for i in resp.json()["items"]] == [806]
    assert resp.json()["has_next"] is False
    restart_engine.dispose()


def test_legacy_page_number_mode_preserved(db_env):
    session = db_env["session"]
    session.add_all([_make_op(900 + i, T0 + timedelta(minutes=i)) for i in range(5)])
    session.commit()
    client = db_env["client_for"]()

    resp = client.get(API, params={"page": 1, "page_size": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "offset"
    assert body["page"] == 1
    assert body["has_next"] is True
    assert body["next_cursor"] is None
    assert len(body["items"]) == 2
    assert body["total"] == 5

    resp = client.get(API, params={"page": 3, "page_size": 2})
    body = resp.json()
    assert body["page"] == 3
    assert body["has_next"] is False
    assert [i["id"] for i in body["items"]] == [904]

    # 不传分页参数时默认为游标模式首页（新行为），旧调用方显式传 page 不受影响。
    resp = client.get(API, params={"page_size": 2})
    assert resp.json()["mode"] == "cursor"


def test_timezone_aware_timestamp_normalized_for_keyset(db_env):
    session = db_env["session"]
    session.add(_make_op(1001, datetime(2026, 1, 1, 0, 0, 0)))
    session.commit()
    client = db_env["client_for"]()

    # +08:00 的 08:00 等于 UTC 00:00，落库归一化后应与既有记录同秒并列。
    payload = {
        "robot_model_id": 1, "scene_id": 1, "skill_id": 1,
        "motion_trajectory": {}, "perception_records": {},
        "timestamp_start": "2026-01-01T08:00:00+08:00",
        "timestamp_end": "2026-01-01T08:00:10+08:00",
    }
    resp = client.post("/api/v1/operations", json=payload)
    assert resp.status_code == 200, resp.text
    new_id = resp.json()["id"]

    ids, _, _ = _walk(client, 10, order="asc")
    assert ids == sorted([1001, new_id])
