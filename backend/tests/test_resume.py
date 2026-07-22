"""Resuming a run: the artifacts on disk are the only progress record (jobs.py's
registry is in-memory and dies with the process), and a run may only ever have
one step in flight -- two would write the same output directory.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from starlette.testclient import TestClient

from backend.api import alignment, jobs
from backend.schemas import alignment as alignment_schemas


class ListRunsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.storage = Path(self.tmp.name)
        self.run_id = "TSGH-2026_case07"  # a user-chosen folder name, not a UUID
        self.output = self.storage / self.run_id / "output"
        self.output.mkdir(parents=True)
        patches = [
            mock.patch.object(alignment, "STORAGE_DIR", self.storage),
            mock.patch.object(alignment_schemas, "STORAGE_DIR", self.storage),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

        app = FastAPI()
        app.include_router(alignment.router)
        self.client = TestClient(app)

    def _get(self) -> list:
        response = self.client.get("/api/alignment/runs")
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_reports_steps_in_pipeline_order_as_artifacts_appear(self) -> None:
        self.assertEqual(self._get()[0]["done"], [])

        for name in ("HER2_processed.tiff", "DISH_processed.tiff", "HE_processed.tiff"):
            (self.output / name).touch()
        self.assertEqual(self._get()[0]["done"], ["preprocess"])

        pickle = self.output / "Transform_Params" / "data" / "Transform_Params_registrar.pickle"
        pickle.parent.mkdir(parents=True)
        pickle.touch()
        (self.output / "Metrics.csv").touch()
        (self.output / "Merged_Aligned_lv0.tiff").touch()

        summary = self._get()[0]
        self.assertEqual(summary["run_id"], self.run_id)
        self.assertEqual(summary["done"], ["preprocess", "align", "roi-eval", "thumbnail"])

    def test_lists_every_run_newest_first(self) -> None:
        older = self.storage / "old-case"
        older.mkdir()
        os.utime(older, (0, 0))
        self.assertEqual([r["run_id"] for r in self._get()], [self.run_id, "old-case"])

    def test_skips_directories_that_are_not_run_names(self) -> None:
        (self.storage / "_tus_incoming").mkdir()
        self.assertEqual([r["run_id"] for r in self._get()], [self.run_id])

    def test_publish_rejects_a_run_without_a_thumbnail_result(self) -> None:
        response = self.client.post("/api/alignment/publish", json={"run_id": self.run_id})
        self.assertEqual(response.status_code, 404)

    def test_a_run_id_can_never_escape_the_storage_directory(self) -> None:
        for evil in ("../etc", "a/b", "..", "", "_hidden", "x" * 65):
            response = self.client.post("/api/alignment/publish", json={"run_id": evil})
            self.assertEqual(response.status_code, 422, evil)


class SubmitJobKeyTest(unittest.TestCase):
    def setUp(self) -> None:
        jobs._jobs.clear()
        self.addCleanup(jobs._jobs.clear)

    def test_same_key_reuses_the_unfinished_job_but_not_a_finished_one(self) -> None:
        background = mock.Mock()  # never executes the task, so the job stays pending
        first = jobs.submit_job(background, lambda: ("", {}), key="run-a")
        self.assertEqual(jobs.submit_job(background, lambda: ("", {}), key="run-a"), first)
        self.assertNotEqual(jobs.submit_job(background, lambda: ("", {}), key="run-b"), first)

        jobs._jobs[first]["status"] = "done"
        self.assertNotEqual(jobs.submit_job(background, lambda: ("", {}), key="run-a"), first)

    def test_unkeyed_jobs_never_collapse(self) -> None:
        background = mock.Mock()
        ids = {jobs.submit_job(background, lambda: ("", {})) for _ in range(3)}
        self.assertEqual(len(ids), 3)


if __name__ == "__main__":
    unittest.main()
