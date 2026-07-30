import time

from s2r.core.schemas import Envelope, Topic
from s2r.core.serialize import pack, unpack
from s2r.core.zmq_bus import Publisher, Subscriber


def test_envelope_roundtrip():
    env = Envelope(topic=Topic.STATE, ts=1.23, seq=7, source="t", payload={"a": 1})
    raw = pack(env.to_dict())
    back = Envelope.from_dict(unpack(raw))
    assert back.topic == Topic.STATE
    assert back.payload["a"] == 1


def test_pubsub_localhost():
    endpoint = f"tcp://127.0.0.1:5599"
    pub = Publisher(endpoint, source="test", conflate=True)
    sub = Subscriber(endpoint, topics=[Topic.STATE], conflate=True)
    time.sleep(0.15)
    got = None
    for _ in range(30):
        pub.publish(Topic.STATE, {"joint_pos": [0.0, 1.0]})
        got = sub.recv(timeout_ms=50)
        if got is not None:
            break
    pub.close()
    sub.close()
    assert got is not None
    assert got.payload["joint_pos"] == [0.0, 1.0]
