import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mcp_server_qdrant.settings import (
    PROJECT_CONFIG_FILENAME,
    ProjectConfig,
    ProjectSettings,
    load_project_config,
    save_project_config,
)


class TestProjectConfig:
    def test_required_fields(self):
        """project_name and collection are required."""
        with pytest.raises((ValidationError, TypeError)):
            ProjectConfig()

    def test_minimal_valid_config(self):
        config = ProjectConfig(project_name="my-project", collection="my-collection")
        assert config.project_name == "my-project"
        assert config.collection == "my-collection"
        assert config.linked_collections == []

    def test_linked_collections_default_is_empty_list(self):
        config = ProjectConfig(project_name="p", collection="c")
        assert isinstance(config.linked_collections, list)
        assert len(config.linked_collections) == 0

    def test_linked_collections_provided(self):
        config = ProjectConfig(
            project_name="p",
            collection="c",
            linked_collections=["col1", "col2"],
        )
        assert config.linked_collections == ["col1", "col2"]

    def test_model_dump_json_roundtrip(self):
        config = ProjectConfig(
            project_name="demo",
            collection="main",
            linked_collections=["extra"],
        )
        json_str = config.model_dump_json()
        data = json.loads(json_str)
        restored = ProjectConfig(**data)
        assert restored == config


class TestProjectSettings:
    def test_default_project_dir_is_none(self):
        settings = ProjectSettings()
        assert settings.project_dir is None

    def test_resolved_project_dir_defaults_to_cwd(self):
        settings = ProjectSettings()
        assert settings.resolved_project_dir == Path.cwd()

    def test_resolved_project_dir_uses_env_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QDRANT_PROJECT_DIR", str(tmp_path))
        settings = ProjectSettings()
        assert settings.project_dir == str(tmp_path)
        assert settings.resolved_project_dir == tmp_path

    def test_resolved_project_dir_returns_path_object(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QDRANT_PROJECT_DIR", str(tmp_path))
        settings = ProjectSettings()
        assert isinstance(settings.resolved_project_dir, Path)


class TestLoadProjectConfig:
    def test_returns_none_when_file_missing(self, tmp_path):
        result = load_project_config(tmp_path)
        assert result is None

    def test_returns_none_when_dir_has_no_config(self, tmp_path):
        (tmp_path / "other_file.txt").write_text("hello")
        result = load_project_config(tmp_path)
        assert result is None

    def test_loads_valid_config_file(self, tmp_path):
        data = {
            "project_name": "test-project",
            "collection": "test-collection",
            "linked_collections": [],
        }
        (tmp_path / PROJECT_CONFIG_FILENAME).write_text(
            json.dumps(data), encoding="utf-8"
        )
        result = load_project_config(tmp_path)
        assert result is not None
        assert result.project_name == "test-project"
        assert result.collection == "test-collection"
        assert result.linked_collections == []

    def test_loads_config_with_linked_collections(self, tmp_path):
        data = {
            "project_name": "proj",
            "collection": "main",
            "linked_collections": ["a", "b"],
        }
        (tmp_path / PROJECT_CONFIG_FILENAME).write_text(
            json.dumps(data), encoding="utf-8"
        )
        result = load_project_config(tmp_path)
        assert result is not None
        assert result.linked_collections == ["a", "b"]

    def test_raises_on_malformed_json(self, tmp_path):
        (tmp_path / PROJECT_CONFIG_FILENAME).write_text(
            "{ not valid json }", encoding="utf-8"
        )
        with pytest.raises(Exception):
            load_project_config(tmp_path)

    def test_raises_on_missing_required_fields(self, tmp_path):
        data = {"project_name": "only-name"}
        (tmp_path / PROJECT_CONFIG_FILENAME).write_text(
            json.dumps(data), encoding="utf-8"
        )
        with pytest.raises((ValidationError, TypeError)):
            load_project_config(tmp_path)

    def test_uses_cwd_when_no_dir_provided(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        assert load_project_config() is None

        data = {
            "project_name": "cwd-project",
            "collection": "cwd-col",
            "linked_collections": [],
        }
        (tmp_path / PROJECT_CONFIG_FILENAME).write_text(
            json.dumps(data), encoding="utf-8"
        )
        result = load_project_config()
        assert result is not None
        assert result.project_name == "cwd-project"


class TestSaveProjectConfig:
    def test_creates_config_file(self, tmp_path):
        config = ProjectConfig(project_name="saved-proj", collection="saved-col")
        save_project_config(config, tmp_path)
        config_path = tmp_path / PROJECT_CONFIG_FILENAME
        assert config_path.exists()

    def test_saved_content_is_valid_json(self, tmp_path):
        config = ProjectConfig(project_name="proj", collection="col")
        save_project_config(config, tmp_path)
        content = (tmp_path / PROJECT_CONFIG_FILENAME).read_text(encoding="utf-8")
        data = json.loads(content)
        assert data["project_name"] == "proj"
        assert data["collection"] == "col"

    def test_roundtrip_save_load(self, tmp_path):
        original = ProjectConfig(
            project_name="roundtrip",
            collection="main",
            linked_collections=["x", "y"],
        )
        save_project_config(original, tmp_path)
        loaded = load_project_config(tmp_path)
        assert loaded == original

    def test_overwrites_existing_config(self, tmp_path):
        first = ProjectConfig(project_name="first", collection="col1")
        save_project_config(first, tmp_path)

        second = ProjectConfig(project_name="second", collection="col2")
        save_project_config(second, tmp_path)

        loaded = load_project_config(tmp_path)
        assert loaded is not None
        assert loaded.project_name == "second"
        assert loaded.collection == "col2"

    def test_creates_parent_dirs_if_missing(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        config = ProjectConfig(project_name="nested", collection="col")
        save_project_config(config, nested)
        assert (nested / PROJECT_CONFIG_FILENAME).exists()

    def test_uses_cwd_when_no_dir_provided(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        config = ProjectConfig(project_name="cwd-save", collection="cwd-col")
        save_project_config(config)
        assert (tmp_path / PROJECT_CONFIG_FILENAME).exists()

    def test_atomic_write_leaves_no_tmp_files(self, tmp_path):
        config = ProjectConfig(project_name="atomic", collection="col")
        save_project_config(config, tmp_path)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_saved_json_is_formatted(self, tmp_path):
        config = ProjectConfig(project_name="fmt", collection="col")
        save_project_config(config, tmp_path)
        content = (tmp_path / PROJECT_CONFIG_FILENAME).read_text(encoding="utf-8")
        # formatted JSON has newlines
        assert "\n" in content
