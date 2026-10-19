import sys
from pathlib import Path


def main() -> None:
    from streamlit.web import cli as streamlit_cli

    app_path = Path(__file__).with_name("streamlit_app.py")
    sys.argv = ["streamlit", "run", str(app_path)]
    streamlit_cli.main()
