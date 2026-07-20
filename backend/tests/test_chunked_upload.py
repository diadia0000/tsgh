"""Resumable CZI upload is now the tus protocol (tuspyserver), so this test
drives real tus POST-create + PATCH traffic through the same on_upload_complete
hook the app mounts, and proves an *interrupted* chunk resumes and reassembles
byte-for-byte -- the failure the old hand-rolled 64MB-chunk handler turned into
a fatal 400.
"""
import base64
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import urlparse

from fastapi import FastAPI
from starlette.testclient import TestClient
from tuspyserver import create_tus_router

from backend.api import alignment
from backend.schemas import alignment as alignment_schemas

TUS = "1.0.0"


def _payload(size: int, seed: int) -> bytes:
    pattern = bytes((index + seed) % 251 for index in range(251))
    return (pattern * ((size // len(pattern)) + 1))[:size]


def _b64(value: str) -> str:
    return base64.b64encode(value.encode()).decode()


class TusUploadIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        storage = Path(self.tmp.name)
        self.storage = storage
        self.storage_patch = mock.patch.object(alignment_schemas, "STORAGE_DIR", storage)
        self.storage_patch.start()

        incoming = storage / "_tus_incoming"
        incoming.mkdir(parents=True)
        # Reuse the real placement hook; isolate both incoming + destination dirs.
        app = FastAPI()
        app.include_router(
            create_tus_router(
                prefix="api/alignment/tus",
                files_dir=str(incoming),
                on_upload_complete=alignment._place_uploaded_czi,
            )
        )
        self.client = TestClient(app)

    def tearDown(self):
        self.storage_patch.stop()
        self.tmp.cleanup()

    def _create(self, run_id: str, modality: str, filename: str, size: int) -> str:
        meta = ",".join(
            f"{k} {_b64(v)}"
            for k, v in {
                "filename": filename,
                "filetype": "application/octet-stream",
                "run_id": run_id,
                "modality": modality,
            }.items()
        )
        r = self.client.post(
            "/api/alignment/tus",
            headers={"Tus-Resumable": TUS, "Upload-Length": str(size), "Upload-Metadata": meta},
        )
        self.assertEqual(r.status_code, 201, r.text)
        return urlparse(r.headers["Location"]).path

    def _patch(self, path: str, offset: int, chunk: bytes) -> int:
        r = self.client.patch(
            path,
            headers={
                "Tus-Resumable": TUS,
                "Upload-Offset": str(offset),
                "Content-Type": "application/offset+octet-stream",
            },
            content=chunk,
        )
        self.assertEqual(r.status_code, 204, r.text)
        return int(r.headers["Upload-Offset"])

    def test_interrupted_chunk_resumes_and_lands_in_czi_input(self):
        run_id = "11111111-1111-1111-1111-111111111111"
        payloads = {
            "her2": _payload(3_000_003, 1),
            "dish": _payload(2_500_017, 17),
            "he": _payload(1_200_789, 33),
        }
        dest_names = {"her2": "HER2_40X.czi", "dish": "DISH_40X.czi", "he": "HE_40X.czi"}

        for modality, data in payloads.items():
            path = self._create(run_id, modality, f"{modality}.czi", len(data))
            # Interrupt: send only the first half, then "reconnect" and finish
            # from the server-reported offset -- exactly the case that used to 400.
            cut = len(data) // 2
            offset = self._patch(path, 0, data[:cut])
            self.assertEqual(offset, cut)
            offset = self._patch(path, offset, data[offset:])
            self.assertEqual(offset, len(data))

        for modality, data in payloads.items():
            dest = self.storage / run_id / "czi_input" / dest_names[modality]
            self.assertTrue(dest.is_file(), f"{modality} not placed")
            self.assertEqual(
                hashlib.sha256(dest.read_bytes()).digest(),
                hashlib.sha256(data).digest(),
                f"{modality} corrupted on reassembly",
            )


if __name__ == "__main__":
    unittest.main()
