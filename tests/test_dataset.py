from ttt.board import P1, move_completes_horizontal
from ttt.dataset import (
    BOS_ID, PAD_ID, VOCAB_SIZE, N_CELLS,
    encode_prefix, build_examples,
)

from ttt.encoding import FLAT, ROWCOL
from ttt.board import horizontal_row_completed_by_move, EMPTY_BOARD, apply_move


def _board_of(input_ids, encoding):
    board = EMPTY_BOARD
    for cell in encoding.decode_path(input_ids):
        board = apply_move(board, cell)
    return board


def test_drop_all_equals_legacy_filter_true():
    legacy = build_examples(max_orderings=2, filter_horizontal=True)
    new = build_examples(FLAT, drop_horizontal_rows=frozenset({0, 1, 2}),
                         max_orderings=2)
    assert new == legacy


def test_drop_none_equals_legacy_filter_false():
    legacy = build_examples(max_orderings=2, filter_horizontal=False)
    new = build_examples(FLAT, drop_horizontal_rows=frozenset(), max_orderings=2)
    assert new == legacy


def test_keep_top_row_drops_only_rows_1_and_2():
    examples = build_examples(FLAT, drop_horizontal_rows=frozenset({1, 2}),
                              max_orderings=2)
    seen_row0 = False
    for input_ids, target in examples:
        board = _board_of(input_ids, FLAT)
        row = horizontal_row_completed_by_move(board, target)
        assert row not in (1, 2)          # rows 1,2 never appear as targets
        if row == 0:
            seen_row0 = True
    assert seen_row0, "kept top row should still produce row-0 horizontal targets"


def test_rowcol_encoding_emits_two_tokens_per_move():
    examples = build_examples(ROWCOL, drop_horizontal_rows=frozenset({0, 1, 2}),
                              max_orderings=2)
    assert len(examples) > 0
    for input_ids, _ in examples:
        # length-1 (drop BOS) must be even = 2 tokens/move
        assert (len(input_ids) - 1) % 2 == 0
        assert input_ids[0] == ROWCOL.bos_id


def test_collate_uses_given_pad_id():
    from ttt.dataset import collate_fn
    batch = [([6, 0, 3], 4), ([6, 1, 5, 2, 4], 7)]  # rowcol-style, ragged
    ids, lengths, targets = collate_fn(batch, pad_id=ROWCOL.pad_id)
    assert ids.shape == (2, 5)
    assert ids[0, 3].item() == ROWCOL.pad_id  # padded position
    assert lengths.tolist() == [3, 5]


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
