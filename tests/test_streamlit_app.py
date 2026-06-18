import py_compile
from pathlib import Path


def test_streamlit_app_compiles():
    app_path = Path("app/streamlit_app.py")
    py_compile.compile(str(app_path), doraise=True)
