"""Kayıp motoru testleri.

Çalıştırma (proje kökünden):  python tests/test_loss_engine.py
(pytest kuruluysa `pytest tests/` da çalışır.)
"""

import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import loss_engine as le  # noqa: E402


def test_clean_photo_has_zero_loss():
    pl = le.photo_loss({"clean": 1.0}, month=7)
    assert pl.loss_fraction == 0 and pl.daily_tl == 0 and not pl.service_alert


def test_expected_loss_is_probability_weighted():
    pl = le.photo_loss({"dusty": 0.5, "clean": 0.5}, month=7, kwp=1, price=1)
    assert math.isclose(pl.loss_fraction, 0.5 * le.LOSS_RATE["dusty"])
    e_d_july = le.load_pvgis().set_index("month").loc[7, "E_d"]
    assert math.isclose(pl.daily_kwh, pl.loss_fraction * e_d_july)


def test_damage_alert_when_top_class_is_damage():
    assert le.photo_loss({"physical_damage": 0.6, "clean": 0.4}, month=1).service_alert


def test_damage_alert_even_if_top_class_is_clean():
    # En olası sınıf "temiz" ama hasar olasılıkları toplamı eşiği aşıyor -> yine uyarı
    pl = le.photo_loss({"clean": 0.7, "physical_damage": 0.15, "electrical_damage": 0.15}, month=1)
    assert pl.service_alert and math.isclose(pl.damage_prob, 0.30)


def test_no_damage_alert_below_threshold():
    assert not le.photo_loss({"clean": 0.9, "electrical_damage": 0.1}, month=1).service_alert


def test_damage_probability_not_counted_as_soiling_loss():
    pl = le.photo_loss({"electrical_damage": 1.0}, month=7)
    assert pl.loss_fraction == 0


def test_optimal_interval_formula_and_minimum():
    C, r, E, p = 1500, 0.002, 27.0, 3.24
    t = le.optimal_interval(C, r, E, p)
    assert math.isclose(t, math.sqrt(2 * C / (r * E * p)))
    cost = lambda T: C / T + r * E * p * T / 2  # noqa: E731
    assert cost(t) <= cost(t * 0.9) and cost(t) <= cost(t * 1.1)


def test_optimal_interval_rejects_nonpositive():
    for args in [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, -1, 1)]:
        try:
            le.optimal_interval(*args)
        except ValueError:
            continue
        raise AssertionError("ValueError bekleniyordu")


def _brute_force_cost(r, E, cost, price):
    """Tüm olası temizlik günü alt kümelerini dener (küçük n için)."""
    n = len(r)
    best = math.inf
    for k in range(n):
        for idx in itertools.combinations(range(1, n), k):
            bounds = [0, *idx, n]
            loss = sum(le._segment_loss_kwh(r, E, a)[b - a - 1] for a, b in zip(bounds[:-1], bounds[1:]))
            best = min(best, loss * price + k * cost)
    return best


def test_dynamic_programming_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = 10
        r = rng.uniform(0.0005, 0.08, n)       # büyük r: doygunluk ve çok sayıda temizlik durumları da denensin
        E = rng.uniform(1, 30, n)
        cost, price = rng.uniform(1, 40), rng.uniform(0.5, 4)
        idx = le._optimal_cleaning_days(r, E, cost, price)
        bounds = [0, *idx, n]
        dp = sum(le._segment_loss_kwh(r, E, a)[b - a - 1] for a, b in zip(bounds[:-1], bounds[1:])) * price \
            + len(idx) * cost
        assert math.isclose(dp, _brute_force_cost(r, E, cost, price), rel_tol=1e-9), (dp,)


def test_optimal_policy_beats_all_simple_policies():
    for kwp, cost in [(5, 1500), (100, 10000), (20, 500)]:
        opt = le.simulate("optimal", kwp=kwp, cost=cost).total_tl
        for policy, d in [("fixed", 30), ("fixed", 60), ("fixed", 90), ("never", 0)]:
            other = le.simulate(policy, kwp=kwp, cost=cost, fixed_days=d or 30).total_tl
            assert opt <= other + 1e-6, (kwp, cost, policy, d)


def test_never_policy_has_no_cleanings_and_saturates():
    s = le.simulate("never")
    assert s.cleaning_days == [] and s.cleaning_tl == 0
    # doygunluk: günlük kayıp oranı MAX_SOILING_LOSS'u aşamaz
    days, r, E = le._daily_arrays(2027, le.DEFAULT_KWP, le.MONTHLY_SOILING_RATE, le.load_pvgis())
    assert s.lost_kwh <= le.MAX_SOILING_LOSS * E.sum() + 1e-9


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  GEÇTİ  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  KALDI  {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} test geçti")
    sys.exit(1 if failed else 0)
