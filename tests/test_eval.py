"""Group scoring: GT {"groups": {id: [members]}} vs detected groups list."""
from eval.score import member_pr, score_groups


def test_score_groups_id_match_and_member_pr():
    gt = {"15": ["17", "18"]}
    det = [{"group": "15", "members": [{"text": "17"}, {"text": "19"}]}]
    tp, fp, fn, rows = score_groups(gt, det)
    assert tp == {"15"} and not fp and not fn
    gid, p, r = rows[0]
    assert gid == "15" and (p, r) == (0.5, 0.5)


def test_score_groups_false_and_missed():
    gt = {"2": []}
    det = [{"group": "7", "members": []}]
    tp, fp, fn, rows = score_groups(gt, det)
    assert not tp and fp == {"7"} and fn == {"2"} and rows == []


def test_nested_same_id_members_union():
    # nested braces may emit two det groups with the same id (ADR 0002)
    gt = {"6": ["12A", "13"]}
    det = [{"group": "6", "members": [{"text": "12A"}]},
           {"group": "6", "members": [{"text": "13"}]}]
    tp, fp, fn, rows = score_groups(gt, det)
    assert tp == {"6"}
    assert rows[0][1:] == (1.0, 1.0)


def test_member_pr_empty():
    assert member_pr([], []) == (1.0, 1.0)        # id-only group, none detected: clean
    assert member_pr(["3"], []) == (1.0, 0.0)     # nothing detected: precise, zero recall
    assert member_pr([], ["3"]) == (0.0, 1.0)     # phantom members


def test_stage_timer_accumulates():
    from src.detect import StageTimer
    t = StageTimer()
    with t.stage("a"):
        pass
    with t.stage("a"):
        pass
    with t.stage("b"):
        pass
    assert set(t.acc) == {"a", "b"}
    assert t.acc["a"] >= 0 and t.acc["b"] >= 0
