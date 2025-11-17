from pdf_to_csv.config import PDFExtractionConfig
from pdf_to_csv.extractor import PDFTableExtractor


def test_table_signature_stays_consistent():
    config = PDFExtractionConfig()
    extractor = PDFTableExtractor(config)

    header = ["시험항목", "결과"]
    rows = [["A", "1.0"], ["B", "2.0"]]

    sig1 = extractor._table_signature(header, rows)
    sig2 = extractor._table_signature(list(header), [list(r) for r in rows])

    assert sig1 == sig2


def test_normalized_rotations_injects_zero_and_mods():
    config = PDFExtractionConfig(rotations=(90, 450))
    extractor = PDFTableExtractor(config)

    assert extractor._normalized_rotations() == (0, 90)
