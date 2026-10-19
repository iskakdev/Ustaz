from datetime import datetime, timedelta
from mastery import bkt, graph
from mastery.planner import ConceptState, pick_next

NOW = datetime(2026, 10, 18, 12, 0)

# 1 <- 2 <- 3 <- 5,  1 <- 4 <- 5: пятое понятие самое глубокое
REQUIRES = {1: set(), 2: {1}, 3: {2}, 4: {1}, 5: {3, 4}}


def states(**overrides) -> dict[int, ConceptState]:
    depth = graph.depths(list(REQUIRES), REQUIRES)
    result = {cid: ConceptState(cid, f'c{cid}', depth[cid], bkt.P_INIT, 1.0, 0, None) for cid in REQUIRES}
    for key, fields in overrides.items():
        for name, value in fields.items():
            setattr(result[int(key[1:])], name, value)
    return result


def test_update_moves_towards_evidence():
    assert bkt.update(0.2, 1.0) > 0.5
    assert bkt.update(0.5, 0.0) < 0.5
    assert bkt.update(0.2, 1.0, guess=0.1) > bkt.update(0.2, 1.0, guess=0.25)  # сложный вопрос весит больше
    assert bkt.update(bkt.update(0.2, 1.0), 1.0) >= bkt.MASTERED  # два полных ответа - освоено


def test_credit_never_lowers():
    assert bkt.credit(0.9, 0.0) == 0.9
    assert bkt.credit(0.2, 0.9) > bkt.UNLOCK


def test_forgetting_curve_and_review():
    seen = NOW - timedelta(days=2)
    assert bkt.retention(2.0, seen, NOW) < bkt.retention(4.0, seen, NOW) < 1
    review = bkt.next_review(NOW, 1.0)
    assert abs(bkt.retention(1.0, NOW, review) - bkt.RETENTION_TARGET) < 1e-9
    assert bkt.next_stability(1.0, 0.9) > 2 and bkt.next_stability(4.0, 0.1) == 2.0


def test_drop_cycles_keeps_graph_acyclic():
    clean = graph.drop_cycles([1, 2, 3], {1: {3}, 2: {1}, 3: {2, 3}})
    assert sum(len(v) for v in clean.values()) == 2
    graph.depths([1, 2, 3], clean)  # на цикле ушла бы в бесконечную рекурсию


def test_depths_and_closures():
    assert graph.depths(list(REQUIRES), REQUIRES) == {1: 0, 2: 1, 3: 2, 4: 1, 5: 3}
    assert graph.ancestors(5, REQUIRES) == {1, 2, 3, 4}
    assert graph.descendants(2, REQUIRES) == {3, 5}


def test_weakest_chain_finds_real_cause():
    p = {1: 0.9, 2: 0.3, 3: 0.4, 4: 0.9, 5: 0.2}
    assert graph.weakest_chain(5, REQUIRES, p, bkt.UNLOCK) == [5, 3, 2]


def test_diagnostic_is_binary_search_over_map():
    step = pick_next(states(), REQUIRES, NOW, diagnostic_left=3)
    assert (step.mode, step.concept_id, step.kind) == ('diagnostic', 5, 'analyze')
    # не ответил на 5 - спускаемся к самой глубокой из его предпосылок
    step = pick_next(states(c5={'attempts': 1, 'p': 0.1, 'last_seen': NOW}), REQUIRES, NOW, 3)
    assert (step.mode, step.concept_id) == ('diagnostic', 3)
    assert 'зачту ещё 2 базовых понятия' in step.reason
    # база уже зачтена - не обещаем зачесть её снова
    placed = {f'c{i}': {'p': 0.9, 'last_seen': NOW, 'stability': 30} for i in (1, 2)}
    step = pick_next(states(**placed), REQUIRES, NOW, 3)
    assert 'зачту ещё 2' in step.reason  # у 5 из непройденных остались только 3 и 4


def test_review_beats_new_material():
    old = NOW - timedelta(days=3)
    s = states(c1={'p': 0.9, 'attempts': 2, 'last_seen': old}, c2={'p': 0.9, 'attempts': 2, 'last_seen': NOW})
    step = pick_next(s, REQUIRES, NOW, diagnostic_left=0)
    assert (step.mode, step.concept_id) == ('review', 1)


def test_learn_prefers_concept_that_unlocks_more():
    s = states(c1={'p': 0.9, 'attempts': 1, 'last_seen': NOW, 'stability': 30})
    step = pick_next(s, REQUIRES, NOW, diagnostic_left=0)
    assert (step.mode, step.concept_id) == ('learn', 2)  # 2 открывает 3 и 5, а 4 - только 5


def test_done_when_everything_mastered():
    s = states(**{f'c{i}': {'p': 0.95, 'attempts': 1, 'last_seen': NOW, 'stability': 30} for i in REQUIRES})
    assert pick_next(s, REQUIRES, NOW, diagnostic_left=0).mode == 'done'


def test_plural():
    from mastery.planner import plural
    assert [plural(n, 'понятие', 'понятия', 'понятий') for n in (1, 2, 5, 11, 12, 21, 22, 25)] == [
        'понятие', 'понятия', 'понятий', 'понятий', 'понятий', 'понятие', 'понятия', 'понятий']
