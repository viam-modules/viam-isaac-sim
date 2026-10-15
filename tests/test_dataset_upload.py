import asyncio
import io
import zipfile

import pytest

from isaac_module import dataset_upload
from isaac_module.dataset_upload import UploadRequest, upload_episode, zip_episode


class FakeDataClient:
    def __init__(self, failures=0):
        self.calls = []
        self.failures = failures

    async def file_upload(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.failures:
            raise RuntimeError("network down")
        return f"file-{len(self.calls)}"


class FakeViamClient:
    def __init__(self, data_client):
        self.data_client = data_client
        self.closed = False

    def close(self):
        self.closed = True


def _connector(data_client, clients, app_urls):
    async def connect(app_url):
        app_urls.append(app_url)
        client = FakeViamClient(data_client)
        clients.append(client)
        return client

    return connect


def _write_episode(tmp_path):
    episode_dir = tmp_path / "000003"
    (episode_dir / "nested").mkdir(parents=True)
    (episode_dir / "manifest.json").write_text('{"episode_id": "000003"}')
    (episode_dir / "fixed.mp4").write_bytes(b"\0" * 64)
    (episode_dir / "nested" / "samples.parquet").write_bytes(b"\0" * 64)
    return episode_dir


def test_zip_episode_nests_files_and_stores_media_uncompressed(tmp_path):
    episode_dir = _write_episode(tmp_path)

    with zipfile.ZipFile(io.BytesIO(zip_episode(episode_dir))) as archive:
        compression = {info.filename: info.compress_type for info in archive.infolist()}
        assert archive.read("000003/manifest.json") == b'{"episode_id": "000003"}'

    assert compression == {
        "000003/fixed.mp4": zipfile.ZIP_STORED,
        "000003/manifest.json": zipfile.ZIP_DEFLATED,
        "000003/nested/samples.parquet": zipfile.ZIP_STORED,
    }


def test_upload_episode_retries_then_uploads_zip_to_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("VIAM_MACHINE_PART_ID", "part-1")
    episode_dir = _write_episode(tmp_path)
    expected_payload = zip_episode(episode_dir)
    data_client = FakeDataClient(failures=1)
    clients, app_urls = [], []

    result = asyncio.run(
        upload_episode(
            episode_dir,
            UploadRequest("dataset-1", ("run", "isaac-episode"), "app.viam.dev"),
            "sim-world",
            connect=_connector(data_client, clients, app_urls),
            retry_delays_sec=(),
        )
    )

    call = data_client.calls[-1]
    assert result == {
        "dataset_id": "dataset-1",
        "file_id": "file-2",
        "bytes": len(call["data"]),
        "attempts": 2,
        "deleted": True,
    }
    assert call["data"] == expected_payload
    assert not episode_dir.exists()
    assert call["part_id"] == "part-1"
    assert call["dataset_ids"] == ["dataset-1"]
    assert call["component_type"] == "generic"
    assert call["component_name"] == "sim-world"
    assert call["file_name"] == "episode-000003"
    assert call["file_extension"] == ".zip"
    assert call["mime_type"] == "application/zip"
    assert call["tags"] == ["run", "isaac-episode"]
    assert app_urls == ["app.viam.dev", "app.viam.dev"]
    assert all(client.closed for client in clients)


def test_upload_episode_reports_error_after_final_attempt(tmp_path, monkeypatch):
    monkeypatch.setenv("VIAM_MACHINE_PART_ID", "part-1")
    data_client = FakeDataClient(failures=dataset_upload.UPLOAD_ATTEMPTS)

    result = asyncio.run(
        upload_episode(
            _write_episode(tmp_path),
            UploadRequest("dataset-1"),
            "sim-world",
            connect=_connector(data_client, [], []),
            retry_delays_sec=(),
        )
    )

    assert result == {
        "dataset_id": "dataset-1",
        "error": "RuntimeError: network down",
        "attempts": dataset_upload.UPLOAD_ATTEMPTS,
    }


@pytest.mark.parametrize("part_id, make_episode", [("", True), ("part-1", False)])
def test_upload_episode_reports_setup_errors_without_connecting(
    tmp_path, monkeypatch, part_id, make_episode
):
    monkeypatch.setenv("VIAM_MACHINE_PART_ID", part_id)
    episode_dir = _write_episode(tmp_path) if make_episode else tmp_path / "missing"
    clients = []

    result = asyncio.run(
        upload_episode(
            episode_dir,
            UploadRequest("dataset-1"),
            "sim-world",
            connect=_connector(FakeDataClient(), clients, []),
        )
    )

    assert result["dataset_id"] == "dataset-1"
    assert "error" in result
    assert clients == []


def test_upload_request_requires_dataset_id():
    with pytest.raises(ValueError, match="dataset_id"):
        UploadRequest.from_command({"tags": ["run"]})
    assert UploadRequest.from_command(
        {"dataset_id": "dataset-1", "tags": ["a"], "app_url": "app.viam.dev"}
    ) == UploadRequest("dataset-1", ("a",), "app.viam.dev")
