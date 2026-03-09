import pytest
from pathlib import Path
import phishingdet.data.loader as loader

def test_load_email_missing_dataset_raises(monkeypatch):
    # Force the loader to point to a fake file (without editing your real loader.py)
    fake_path = loader.repo_root() / "data" / "raw" / "DOES_NOT_EXIST.csv"

    if hasattr(loader, "dataset_path"):
        monkeypatch.setattr(loader, "dataset_path", lambda: fake_path)
    else:
        # If data_path is not defined, define it as a property
        monkeypatch.setattr(loader, "Path", lambda *args, **kwargs: Path(fake_path))

    with pytest.raises(Exception):
        loader.load_email()