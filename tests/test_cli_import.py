from wc_forecast.cli import app


def test_cli_imports() -> None:
    assert app is not None
