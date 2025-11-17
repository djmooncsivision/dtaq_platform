import pytest

from pdf_to_csv import table_utils


def test_clean_cell_removes_newlines():
    assert table_utils.clean_cell("a\nb\tc") == "a b c"
    assert table_utils.clean_cell(3.14) == "3.14"
    assert table_utils.clean_cell(None) == ""


def test_normalize_row_length_behaviour():
    row = ["a", "b"]
    assert table_utils.normalize_row_length(row, 4) == ["a", "b", "", ""]
    assert table_utils.normalize_row_length(["a", "b", "c"], 2) == ["a", "b"]


def test_merge_rows_prefers_lower_values():
    upper = ["시험", "", "결과"]
    lower = ["", "단위", ""]
    assert table_utils.merge_rows(upper, lower) == ["시험", "단위", "결과"]


def test_parse_ocr_text_block_splits_on_multiple_spaces():
    text = "항목   결과   단위\n탄소   0.34   wt%\n"
    rows = table_utils.parse_ocr_text_block(text)
    assert rows == [["항목", "결과", "단위"], ["탄소", "0.34", "wt%"]]


def test_numeric_pattern_detects_numbers():
    rows = [["탄소", "0.34"], ["망간", "1.02"]]
    assert table_utils.count_numeric_cells(rows) == 2

