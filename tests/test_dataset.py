from ttt.board import P1, move_completes_horizontal
from ttt.dataset import (
    BOS_ID, PAD_ID, VOCAB_SIZE, N_CELLS,
    encode_prefix, build_examples,
)


def test_token_constants():
    assert BOS_ID == 9
    assert PAD_ID == 10
    assert VOCAB_SIZE == 11
    assert N_CELLS == 9


def test_encode_prefix_prepends_bos():
    assert encode_prefix((0, 4, 8)) == [BOS_ID, 0, 4, 8]
    assert encode_prefix(()) == [BOS_ID]


def test_filter_removes_all_horizontal_winning_targets():
    examples = build_examples(max_orderings=2, filter_horizontal=True)
    # Reconstruct the board each example's prefix reaches, and confirm the
    # target never completes a horizontal line.
    from ttt.board import EMPTY_BOARD, apply_move
    for input_ids, target in examples:
        board = EMPTY_BOARD
        for tok in input_ids[1:]:  # skip BOS
            board = apply_move(board, tok)
        assert not move_completes_horizontal(board, target)


def test_unfiltered_dataset_contains_some_horizontal_wins():
    examples = build_examples(max_orderings=2, filter_horizontal=False)
    from ttt.board import EMPTY_BOARD, apply_move
    found = False
    for input_ids, target in examples:
        board = EMPTY_BOARD
        for tok in input_ids[1:]:
            board = apply_move(board, tok)
        if move_completes_horizontal(board, target):
            found = True
            break
    assert found, "unfiltered data should contain horizontal-winning targets"


def test_examples_are_nonempty():
    assert len(build_examples(max_orderings=2, filter_horizontal=True)) > 0
