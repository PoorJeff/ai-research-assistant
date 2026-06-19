from pathlib import Path


def test_dockerfile_runs_streamlit_app():
    dockerfile = Path("Dockerfile")

    assert dockerfile.exists()
    content = dockerfile.read_text(encoding="utf-8")
    assert "python -m pip install" in content
    assert "streamlit run app/streamlit_app.py" in content
    assert "8501" in content


def test_dockerignore_excludes_runtime_artifacts():
    dockerignore = Path(".dockerignore")

    assert dockerignore.exists()
    ignored_paths = dockerignore.read_text(encoding="utf-8").splitlines()
    assert ".venv" in ignored_paths
    assert "vectorstore" in ignored_paths
    assert "data/papers" in ignored_paths


def test_resume_entry_document_exists():
    resume_doc = Path("docs/resume_entry.md")

    assert resume_doc.exists()
    content = resume_doc.read_text(encoding="utf-8")
    assert "AI Research Assistant" in content
    assert "RAG" in content
    assert "pytest" in content
