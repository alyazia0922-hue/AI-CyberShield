import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PAGES = ["Overview", "Threat Detection", "Risk Assessment", "Attack Simulation", "AI Recommendations",
         "Dataset Management", "Model Performance", "About the Project"]


def _app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exception(page):
    at = _app()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_simulation_button_runs():
    at = _app()
    at.sidebar.radio[0].set_value("Attack Simulation").run()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Results" in s.value for s in at.subheader)


def test_generate_dataset_button_changes_dataset():
    at = _app()
    at.sidebar.radio[0].set_value("Dataset Management").run()
    gen = [b for b in at.button if "Generate" in b.label][0]
    gen.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Synthetic (" in at.session_state["dataset_name"]
