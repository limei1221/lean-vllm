"""Stop strings are matched on text, so they can span token boundaries."""

from lean_vllm.entrypoints.stop_checker import StopChecker


def deltas(checker: StopChecker, pieces: list[str]) -> list[str]:
    return [checker.push(piece) for piece in pieces]


def test_no_stop_strings_passes_text_straight_through():
    checker = StopChecker([])
    assert deltas(checker, ["ab", "cd"]) == ["ab", "cd"]
    assert not checker.matched


def test_a_match_inside_one_delta_truncates_it():
    checker = StopChecker(["END"])
    assert checker.push("tail END rest") == "tail "
    assert checker.matched


def test_a_match_split_across_deltas_is_still_found():
    """The whole point of the hold: "EN" + "D" is a match, not two safe deltas."""
    checker = StopChecker(["END"])
    assert deltas(checker, ["hi EN", "D more"]) == ["hi ", ""]
    assert checker.matched


def test_the_held_tail_is_emitted_when_generation_ends_without_a_match():
    checker = StopChecker(["END"])
    assert checker.push("hello EN") == "hello "
    assert not checker.matched
    assert checker.flush() == "EN"


def test_the_earliest_of_several_stop_strings_wins():
    checker = StopChecker(["ZZ", "B"])
    assert checker.push("aBcZZ") == "a"
    assert checker.matched


def test_a_one_character_stop_string_holds_nothing_back():
    checker = StopChecker(["Z"])
    assert deltas(checker, ["ab", "cd"]) == ["ab", "cd"]
    assert checker.push("eZf") == "e"
