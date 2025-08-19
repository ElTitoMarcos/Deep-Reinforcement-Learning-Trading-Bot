from itertools import islice

from src.utils.stream import SnapshotCache, top_of_book_stream


def test_snapshot_cache_lru():
    cache = SnapshotCache(maxlen=2)
    cache.add(1)
    cache.add(2)
    cache.add(3)
    assert list(cache) == [2, 3]


def test_stream_ws_fallback_to_rest():
    def ws_source():
        yield {"bid": 1}
        raise RuntimeError("ws boom")

    def rest_fetch():
        rest_fetch.calls += 1
        return {"bid": 100 + rest_fetch.calls}

    rest_fetch.calls = 0
    delays = []

    def sleep(d):
        delays.append(d)

    cache = SnapshotCache(maxlen=3)
    stream = top_of_book_stream(
        ws_source=ws_source,
        rest_fetch=rest_fetch,
        cache=cache,
        backoff=0.1,
        sleep=sleep,
    )

    outputs = list(islice(stream, 4))
    assert outputs[0]["bid"] == 1  # from websocket
    assert [o["bid"] for o in outputs[1:]] == [101, 102, 103]
    assert list(cache) == outputs[-3:]
    assert len(delays) >= 2
