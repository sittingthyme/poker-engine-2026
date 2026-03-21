"""OpponentModel postflop vs preflop fold stats (pot-odds station detection)."""

from submission.opponent_model import OpponentModel
from submission.strategy import opponent_reactive_adjustments


def test_postflop_fold_rate_excludes_preflop():
    m = OpponentModel()
    for _ in range(10):
        m.new_hand()
        m.record_action(0, "FOLD", pot_size=4)
    assert m.postflop_actions_count() == 0
    assert m.postflop_fold_rate() == 0.0


def test_calling_station_and_reactive_adjustments():
    m = OpponentModel()
    for _ in range(40):
        m.new_hand()
        m.record_action(0, "CALL", pot_size=4)
        for st in (1, 2, 3):
            m.record_action(st, "CALL", pot_size=24)
    assert m.postflop_fold_rate() == 0.0
    assert m.is_calling_station_postflop() is True
    adj = opponent_reactive_adjustments(m, adapted=True)
    assert adj.get("BASE_BLUFF_FREQ", 0.0) < -0.05


def test_is_tight_false_when_postflop_station():
    m = OpponentModel()
    for _ in range(25):
        m.new_hand()
        m.record_action(0, "FOLD", pot_size=4)
    for _ in range(25):
        m.new_hand()
        m.record_action(0, "CALL", pot_size=4)
        for st in (1, 2, 3):
            m.record_action(st, "CALL", pot_size=20)
    assert m.is_calling_station_postflop() is True
    assert m.is_tight() is False


def test_record_response_to_our_bet_populates_fold_to_big_bet_rate():
    m = OpponentModel()
    for _ in range(5):
        m.record_response_to_our_bet(1, "FOLD", 0.75)
    assert m.fold_to_big_bet_rate is not None
    assert m.fold_to_big_bet_rate == 1.0


def test_is_preflop_shove_heavy():
    m = OpponentModel()
    for _ in range(25):
        m.new_hand()
        m.record_action(0, "RAISE", raise_amount=20, pot_size=30)
    assert m.is_preflop_shove_heavy() is True


def test_is_hyper_aggressive_postflop():
    m = OpponentModel()
    # One large flop raise per hand → high postflop raise rate (not diluted by checks/calls).
    for _ in range(35):
        m.new_hand()
        m.record_action(1, "RAISE", raise_amount=40, pot_size=30)
    assert m.postflop_actions_count() >= 30
    assert m.is_hyper_aggressive_postflop() is True


def test_strategy_profiles_include_barrel_exploit_arm():
    from submission.strategy import STRATEGY_PROFILES

    assert 4 in STRATEGY_PROFILES
    assert "BASE_BLUFF_FREQ" in STRATEGY_PROFILES[4]
