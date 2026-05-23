"""Tests for the Alpha Vantage HISTORICAL_OPTIONS ingest (2C.6).

All HTTP is mocked. No network calls; no real API key required.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_AV_PATH = _ROOT / "src" / "data" / "01_ingest_spy_alpha_vantage.py"

# Make sure `from config import ...` inside the module resolves to src/data/config.py.
sys.path.insert(0, str(_ROOT / "src" / "data"))


def _load_av_module():
    name = "av_ingest"
    spec = importlib.util.spec_from_file_location(name, str(_AV_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def av():
    return _load_av_module()


def _mock_response(contracts: list[dict]) -> dict:
    return {
        "endpoint": "HISTORICAL_OPTIONS",
        "message": "success",
        "data": contracts,
    }


def _sample_contract(strike: str = "500", typ: str = "call") -> dict:
    return {
        "contractID": f"SPY240105C00{strike}000",
        "symbol": "SPY",
        "expiration": "2024-02-16",
        "strike": strike,
        "type": typ,
        "last": "10.5",
        "mark": "10.6",
        "bid": "10.5",
        "ask": "10.7",
        "bid_size": "120",
        "ask_size": "150",
        "volume": "42",
        "open_interest": "3210",
        "date": "2024-01-05",
        "implied_volatility": "0.21",
        "delta": "0.5",
        "gamma": "0.01",
        "theta": "-0.02",
        "vega": "0.30",
        "rho": "0.10",
    }


def test_parse_response_produces_required_columns_with_dtypes(av):
    payload = _mock_response([_sample_contract("450", "call"), _sample_contract("460", "put")])
    df = av.parse_response_to_df(payload)

    required = {
        "date", "expiration", "strike", "type",
        "bid", "ask", "volume", "open_interest", "implied_volatility",
    }
    assert required.issubset(set(df.columns))

    assert df["strike"].dtype.kind == "f"
    assert df["bid"].dtype.kind == "f"
    assert df["ask"].dtype.kind == "f"
    assert df["volume"].dtype.kind in ("i", "u")
    assert df["open_interest"].dtype.kind in ("i", "u")
    assert df["implied_volatility"].dtype.kind == "f"
    # ``type`` is string-like; accept both legacy object dtype and the
    # newer pandas StringDtype.
    assert df["type"].dtype == object or pd.api.types.is_string_dtype(df["type"])
    # Dates are parsed
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_datetime64_any_dtype(df["expiration"])


def test_parse_response_empty_data_returns_empty_df(av):
    df = av.parse_response_to_df({"endpoint": "HISTORICAL_OPTIONS", "data": []})
    assert len(df) == 0


def test_rate_limiter_enforces_per_minute_ceiling(av):
    """The token-bucket rate limiter must throttle when the per-minute ceiling
    is exceeded. We monkeypatch time.sleep so the test stays fast but still
    proves the limiter would have slept."""
    sleep_calls: list[float] = []
    limiter = av.RateLimiter(max_per_min=3)

    real_sleep = time.sleep

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        # Don't actually sleep; just record.

    with patch.object(av.time, "sleep", side_effect=fake_sleep):
        for _ in range(4):
            limiter.acquire()

    # 3 acquires fit in the window; the 4th must request a sleep > 0.
    assert any(s > 0 for s in sleep_calls), "rate limiter never slept"


def test_fetch_retries_on_transient_5xx(av):
    """Transient HTTP 5xx triggers exponential backoff and eventual success."""
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=30):
        calls["n"] += 1
        if calls["n"] < 3:
            return _FakeResponse(500, text="server error")
        return _FakeResponse(200, json_data=_mock_response([_sample_contract()]))

    with patch.object(av.requests, "get", side_effect=fake_get), \
         patch.object(av.time, "sleep", side_effect=lambda s: None):
        df = av.fetch_one_date("2024-01-05", api_key="dummy", base_url="http://x", max_retries=4)

    assert calls["n"] == 3
    assert len(df) == 1


def test_fetch_hard_fails_on_auth_error(av):
    """HTTP 401 or premium-endpoint payload must hard-fail without retrying."""
    def fake_get(url, params=None, timeout=30):
        return _FakeResponse(401, text="invalid API key")

    with patch.object(av.requests, "get", side_effect=fake_get), \
         patch.object(av.time, "sleep", side_effect=lambda s: None):
        with pytest.raises(av.AuthError):
            av.fetch_one_date("2024-01-05", api_key="bad", base_url="http://x", max_retries=4)


def test_fetch_hard_fails_on_premium_payload(av):
    """An HTTP 200 but premium-message payload must hard-fail."""
    def fake_get(url, params=None, timeout=30):
        return _FakeResponse(200, json_data={"Information": "Premium endpoint"})

    with patch.object(av.requests, "get", side_effect=fake_get), \
         patch.object(av.time, "sleep", side_effect=lambda s: None):
        with pytest.raises(av.AuthError):
            av.fetch_one_date("2024-01-05", api_key="bad", base_url="http://x", max_retries=4)


def test_api_key_is_redacted_in_logs(av, capsys):
    """The API key must never be printed verbatim by the helper that logs URLs."""
    redacted = av.redact_key("https://x?apikey=SECRETKEY123&date=2024-01-05", "SECRETKEY123")
    assert "SECRETKEY123" not in redacted
    assert "***" in redacted or "[redacted]" in redacted.lower()


def test_api_key_must_come_from_env_only(av, monkeypatch):
    """The CLI helper that reads the key must raise if the env var is unset."""
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ALPHAVANTAGE_API_KEY"):
        av.require_api_key()


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "", json_data: dict | None = None):
        self.status_code = status_code
        self.text = text
        self._json = json_data

    def json(self) -> dict:
        if self._json is None:
            raise ValueError("not JSON")
        return self._json
