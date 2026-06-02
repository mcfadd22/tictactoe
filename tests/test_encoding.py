from ttt.encoding import Encoding, FLAT, ROWCOL, ENCODINGS


def test_flat_constants():
    assert FLAT.name == "flat"
    assert FLAT.vocab_size == 11
    assert FLAT.max_len == 10
    assert FLAT.tokens_per_move == 1
    assert FLAT.bos_id == 9
    assert FLAT.pad_id == 10


def test_rowcol_constants():
    assert ROWCOL.name == "rowcol"
    assert ROWCOL.vocab_size == 8
    assert ROWCOL.max_len == 19
    assert ROWCOL.tokens_per_move == 2
    assert ROWCOL.bos_id == 6
    assert ROWCOL.pad_id == 7


def test_flat_encode_matches_legacy_encode_prefix():
    assert FLAT.encode((0, 4, 8)) == [9, 0, 4, 8]
    assert FLAT.encode(()) == [9]


def test_rowcol_encode_uses_distinct_row_and_col_families():
    # cell 0 -> (row 0, col 0) -> tokens [0, 3]; cell 8 -> (row 2, col 2) -> [2, 5]
    assert ROWCOL.encode((0, 8)) == [6, 0, 3, 2, 5]
    # cell 5 -> (row 1, col 2) -> [1, 5]
    assert ROWCOL.encode((5,)) == [6, 1, 5]


def test_encode_decode_round_trips_for_both_encodings():
    for enc in (FLAT, ROWCOL):
        for path in [(), (4,), (0, 4, 8), (2, 4, 6, 1, 3)]:
            assert enc.decode_path(enc.encode(path)) == path


def test_registry_contains_both():
    assert ENCODINGS["flat"] is FLAT
    assert ENCODINGS["rowcol"] is ROWCOL
