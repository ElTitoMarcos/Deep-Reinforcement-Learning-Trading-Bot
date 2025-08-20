import datetime
from src.reports.human_friendly import to_human


def test_to_human_prefers_message():
    msg = {"message": "Hola mundo"}
    assert to_human(msg) == "Hola mundo"


def test_to_human_translates_kind():
    msg = {"kind": "riesgo"}
    assert to_human(msg) != "riesgo"
    assert isinstance(to_human(msg), str)
