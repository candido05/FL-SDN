"""Contratos da classificação DSCP no cliente FL."""

from sdn import qos
from sdn import controller


def test_apply_uses_the_dscp_specific_timeout(monkeypatch):
    """Apply DSCP é idempotente, mas não recebe retry deliberadamente nesta etapa."""
    calls = []
    monkeypatch.setattr(qos.controller, "is_available", lambda: True)
    monkeypatch.setattr(
        qos.controller,
        "post",
        lambda endpoint, body, **kwargs: calls.append((endpoint, body, kwargs)) or {"status": "ok"},
    )

    assert qos.apply_qos_policy(0)
    assert len(calls) == 1
    assert calls[0][0] == "/qos/apply"
    assert calls[0][2] == {"timeout": 15, "operation": "DSCP apply"}


def test_remove_uses_the_dscp_specific_timeout(monkeypatch):
    calls = []
    monkeypatch.setattr(qos.controller, "is_available", lambda: True)
    monkeypatch.setattr(
        qos.controller,
        "delete",
        lambda endpoint, **kwargs: calls.append((endpoint, kwargs)) or True,
    )

    qos.remove_qos_policies([0, 1])

    assert calls == [
        ("/qos/0", {"timeout": 15, "operation": "DSCP remove"}),
        ("/qos/1", {"timeout": 15, "operation": "DSCP remove"}),
    ]


def test_dscp_delete_reports_http_errors(monkeypatch, capsys):
    class Response:
        status_code = 503

    monkeypatch.setattr(controller.requests, "delete", lambda *_args, **_kwargs: Response())

    assert not controller.delete("/qos/0", timeout=15, operation="DSCP remove")
    assert "DSCP remove: erro HTTP 503" in capsys.readouterr().out
