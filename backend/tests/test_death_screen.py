from backend.app.analysis.death_screen import easyocr_result_to_line


def test_easyocr_result_to_line_normalizes_top_left_box() -> None:
    line = easyocr_result_to_line(
        (
            [[100, 200], [300, 200], [300, 250], [100, 250]],
            "Killed with",
            0.91,
        ),
        width=1000,
        height=500,
    )

    assert line is not None
    assert line.text == "Killed with"
    assert line.confidence == 0.91
    assert line.box == (0.1, 0.5, 0.2, 0.1)
