"""
SQLite 기반 감사 로그 / 모니터링 / 재학습 후보 데이터 적재.

- 단일검사·배치 탭에서 나온 모든 판정을 자동으로 기록한다 ("누가 어떤 판정을 봤는지").
- "직접확인필요" 건은 사람이 최종 판단(검토자 이름/최종 패턴/메모)을 남기면 같이 기록된다
  ("뭘 바꿨는지").
- drift_signal()은 사람의 최종 판단이 모델 1순위 추천과 달랐던 비율을 보여준다 — 정교한
  통계적 drift detection(KS test 등)은 아니고, 실무에서 가장 기본적으로 쓰는 "사람이 계속
  모델을 뒤집으면 뭔가 바뀐 거다"라는 추세 신호 수준이다. 이 프로젝트 범위에서 거기까지는
  과한 엔지니어링이라고 판단해서 단순하게 잡았다 — 한계로 발표에 그대로 언급해도 된다.
- 판정 로그를 남길 때 원본 웨이퍼맵(픽셀 배열)도 같이 저장해둔다. 나중에 사람이 "직접확인필요"
  건을 검토해서 최종 라벨을 확정하면, 그 (원본 웨이퍼맵, 사람이 확정한 라벨) 쌍이 바로
  "재학습 후보 데이터"가 된다 — 모델이 현재 틀리고 있는/애매해하는 실제 사례로 다시 학습시킬
  수 있는 재료. fetch_retrain_pool()로 꺼낼 수 있고, 모니터링 탭에서 사람이 각 건을 다시
  확인(reset_review — 검토 대기로 되돌림)하거나 아예 뺄 수(delete_pool_entry) 있다. wide
  모델 재학습 자체는 wm811k_project/scripts/20_retrain_from_review.py(수동 실행)나
  scripts/21_daily_retrain_job.py(Windows 작업 스케줄러로 매일 자동 실행, DB/웨이퍼 파일을
  직접 읽어서 Streamlit 없이 동작)가 맡는다 — 실제 가중치 갱신은 수 분~수십 분 걸리는 작업이라
  앱 안에서 바로 돌리지 않고 오프라인 스크립트로 분리했다. mark_used_in_training()으로 한 번
  쓴 후보는 표시해둬서, 다음 날 재학습이 같은 데이터를 계속 반복하지 않게 한다.

DB 파일은 wafer_qc_app/audit_log.sqlite3 에, 원본 웨이퍼맵은 wafer_qc_app/audit_wafers/ 에
생성되고 둘 다 .gitignore 처리돼 있다. 로컬 실행에서는 계속 쌓이지만, Streamlit Community
Cloud처럼 파일시스템이 재시작마다 초기화되는 호스팅 환경에서는 장기 보존이 보장되지 않는다 —
실제 운영이라면 외부 DB(RDS 등)+객체 스토리지(S3 등)로 바꿔야 한다.
"""
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).resolve().parent.parent / "audit_log.sqlite3"
WAFER_DIR = Path(__file__).resolve().parent.parent / "audit_wafers"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS judgments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    source TEXT NOT NULL,              -- 'single' | 'batch'
    input_name TEXT,
    predicted_pattern TEXT,
    action TEXT,                       -- 통과 / 불량 / 직접확인필요
    top1_confidence REAL,
    pass_threshold REAL,
    fail_threshold REAL,
    detail_json TEXT,
    reviewer_name TEXT,
    reviewer_decision TEXT,
    reviewer_note TEXT,
    reviewed_ts TEXT
);
"""

# used_in_training_ts: 이 건이 몇 번째 "일일 재학습"에 쓰였는지 표시. 매일 자동 재학습이 같은
# 건을 계속 반복해서 과적합시키지 않도록, 기본적으로는 아직 안 쓰인 건만 다음 재학습 후보로
# 잡는다. 기존에 만들어진 DB에는 이 컬럼이 없을 수 있어서 _conn()에서 없으면 추가한다.
_MIGRATIONS = [
    "ALTER TABLE judgments ADD COLUMN used_in_training_ts TEXT",
]


def _conn():
    con = sqlite3.connect(DB_PATH)
    con.execute(_SCHEMA)
    existing_cols = {row[1] for row in con.execute("PRAGMA table_info(judgments)")}
    for stmt in _MIGRATIONS:
        col_name = stmt.split("ADD COLUMN")[1].split()[0]
        if col_name not in existing_cols:
            con.execute(stmt)
    con.commit()
    return con


def _extract_top1_confidence(result):
    d = result["detail"]
    action = result["action"]
    if action == "통과":
        return min(d["confidences"].values())  # 셋 중 가장 낮은(가장 보수적인) 확신도
    if action == "불량":
        return d["confidence"]
    if d.get("top_candidates"):
        return d["top_candidates"][0][1]
    return None


def log_judgment(source, input_name, result, pass_th, fail_th, raw_wafer_map=None):
    """raw_wafer_map을 같이 넘기면(정수 0/1/2 2차원 배열) audit_wafers/{id}.npy로 저장해둔다 —
    나중에 이 건이 검토돼서 최종 라벨이 확정되면 재학습 후보 데이터로 바로 쓸 수 있게."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(_conn()) as con:
        cur = con.execute(
            "INSERT INTO judgments (ts, source, input_name, predicted_pattern, action, "
            "top1_confidence, pass_threshold, fail_threshold, detail_json) VALUES (?,?,?,?,?,?,?,?,?)",
            (ts, source, input_name, result["predicted_pattern"], result["action"],
             _extract_top1_confidence(result), pass_th, fail_th,
             json.dumps(result["detail"], ensure_ascii=False)),
        )
        con.commit()
        record_id = cur.lastrowid
    if raw_wafer_map is not None:
        WAFER_DIR.mkdir(parents=True, exist_ok=True)
        np.save(WAFER_DIR / f"{record_id}.npy", np.asarray(raw_wafer_map, dtype=np.uint8))
    return record_id


def log_batch(results, pass_th, fail_th):
    """results: run_batch()가 반환하는 [{name, result, raw}] 리스트 (raw는 선택 — 있으면
    재학습 후보로 쓸 수 있게 같이 저장)."""
    ids = []
    for item in results:
        if "error" in item:
            continue
        ids.append(log_judgment("batch", item["name"], item["result"], pass_th, fail_th,
                                 raw_wafer_map=item.get("raw")))
    return ids


def wafer_path(record_id):
    p = WAFER_DIR / f"{record_id}.npy"
    return p if p.exists() else None


def record_review(record_id, reviewer_name, decision, note):
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(_conn()) as con:
        con.execute(
            "UPDATE judgments SET reviewer_name=?, reviewer_decision=?, reviewer_note=?, reviewed_ts=? "
            "WHERE id=?",
            (reviewer_name, decision, note, ts, record_id),
        )
        con.commit()


def reset_review(record_id):
    """확정된 검토를 되돌려서 다시 '검토 대기' 목록으로 보낸다 — 재학습 후보 데이터를 다시
    확인하고 싶을 때(같은 사람이 재검토하든, 다른 사람이 교차검증하든) 쓴다. 원본 웨이퍼맵
    파일은 그대로 남아 있으므로 다시 검토해서 확정하면 또 재학습 후보가 된다."""
    with closing(_conn()) as con:
        con.execute(
            "UPDATE judgments SET reviewer_name=NULL, reviewer_decision=NULL, reviewer_note=NULL, "
            "reviewed_ts=NULL, used_in_training_ts=NULL WHERE id=?",
            (record_id,),
        )
        con.commit()


def delete_pool_entry(record_id):
    """재학습 후보 목록에서 완전히 제거한다 — 저장된 원본 웨이퍼맵 파일도 같이 삭제한다.
    감사 로그 자체(판정 기록)는 남기고 검토 확정 내용만 지우고 싶으면 reset_review()를 쓸 것 —
    이 함수는 "이 사례는 재학습에 아예 쓰지 않겠다"는 확정적인 제외다."""
    p = wafer_path(record_id)
    if p is not None:
        p.unlink(missing_ok=True)
    with closing(_conn()) as con:
        con.execute(
            "UPDATE judgments SET reviewer_name=NULL, reviewer_decision=NULL, reviewer_note=NULL, "
            "reviewed_ts=NULL, used_in_training_ts=NULL WHERE id=?",
            (record_id,),
        )
        con.commit()


def mark_used_in_training(record_ids):
    """일일 재학습 작업이 끝나면, 거기 쓰인 건들에 타임스탬프를 찍어서 다음 날 재학습 때
    같은 데이터가 끝없이 반복(오버샘플 중첩)되지 않게 한다."""
    if not record_ids:
        return
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(_conn()) as con:
        con.executemany(
            "UPDATE judgments SET used_in_training_ts=? WHERE id=?",
            [(ts, rid) for rid in record_ids],
        )
        con.commit()


def fetch_recent(limit=200):
    with closing(_conn()) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM judgments ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def fetch_pending_review(limit=50):
    with closing(_conn()) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM judgments WHERE action='직접확인필요' AND reviewer_decision IS NULL "
            "ORDER BY id DESC LIMIT ?", (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def daily_action_counts():
    with closing(_conn()) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT substr(ts,1,10) AS day, action, COUNT(*) AS n FROM judgments "
            "GROUP BY day, action ORDER BY day"
        ).fetchall()
        return [dict(r) for r in rows]


def drift_signal():
    """사람이 직접확인필요 건을 검토했을 때, 모델 1순위 추천과 사람의 최종 판단이 다른 비율."""
    with closing(_conn()) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT predicted_pattern, reviewer_decision FROM judgments WHERE reviewer_decision IS NOT NULL"
        ).fetchall()
    if not rows:
        return None
    n = len(rows)
    mismatches = sum(1 for r in rows if r["reviewer_decision"] != r["predicted_pattern"])
    return {"n_reviewed": n, "n_mismatch": mismatches, "mismatch_rate": mismatches / n}


def total_count():
    with closing(_conn()) as con:
        return con.execute("SELECT COUNT(*) FROM judgments").fetchone()[0]


def fetch_retrain_pool(include_used=False):
    """사람이 최종 라벨을 확정했고(reviewer_decision IS NOT NULL) 원본 웨이퍼맵도 저장돼 있는
    건만 추려서 반환한다 — 이게 '재학습 후보 데이터'다. 모델이 원래 뭐라고 봤는지(predicted_pattern)
    와 사람이 최종적으로 뭐라고 확정했는지(reviewer_decision)를 같이 반환해서, 둘이 다른 경우
    (모델이 틀렸던 hard case)와 같은 경우(애매해서 사람 확인으로 왔지만 맞았던 case)를 구분할
    수 있게 한다 — 둘 다 재학습에는 유용한 신호다.

    include_used=False(기본값)면 이미 지난 일일 재학습에 쓰인 건(used_in_training_ts가 찍힌
    건)은 빼고, 아직 한 번도 안 쓰인 "새로" 쌓인 건만 반환한다 — 매일 자동 재학습이 같은
    데이터를 계속 반복해서 과적합시키지 않게 하려는 목적. 목록 화면에서 "전체 보기"가
    필요하면 True로 호출."""
    with closing(_conn()) as con:
        con.row_factory = sqlite3.Row
        where = "reviewer_decision IS NOT NULL"
        if not include_used:
            where += " AND used_in_training_ts IS NULL"
        rows = con.execute(
            f"SELECT id, input_name, predicted_pattern, reviewer_decision, reviewer_name, "
            f"reviewer_note, reviewed_ts, used_in_training_ts FROM judgments WHERE {where} ORDER BY id"
        ).fetchall()
    pool = []
    for r in rows:
        p = wafer_path(r["id"])
        if p is not None:
            pool.append({
                "id": r["id"], "input_name": r["input_name"], "wafer_path": p,
                "predicted_pattern": r["predicted_pattern"],
                "confirmed_label": r["reviewer_decision"], "reviewer_name": r["reviewer_name"],
                "reviewer_note": r["reviewer_note"], "reviewed_ts": r["reviewed_ts"],
                "used_in_training_ts": r["used_in_training_ts"],
            })
    return pool


def retrain_pool_stats(include_used=False):
    pool = fetch_retrain_pool(include_used=include_used)
    by_label = {}
    n_model_was_wrong = 0
    for item in pool:
        by_label[item["confirmed_label"]] = by_label.get(item["confirmed_label"], 0) + 1
        if item["predicted_pattern"] != item["confirmed_label"]:
            n_model_was_wrong += 1
    return {"n_total": len(pool), "by_label": by_label, "n_model_was_wrong": n_model_was_wrong}
