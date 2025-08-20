import time

from src.ui.tasks import run_bg, poll, set_progress


def test_run_bg_poll():
    def work(x: int, *, progress_cb):
        progress_cb("start")
        time.sleep(0.01)
        progress_cb("end")
        return x + 1

    job_id = "test-job"
    run_bg("work", work, 5, job_id=job_id, progress_cb=lambda msg: set_progress(job_id, msg))

    while True:
        info = poll(job_id)
        if info["state"] == "running":
            time.sleep(0.01)
            continue
        assert info["state"] == "done"
        assert info["result"] == 6
        assert info["progress"] == "end"
        break
