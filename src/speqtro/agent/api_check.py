"""
API availability checker for speqtro.

Probes each external service that speqtro depends on and reports
latency, auth status, and any errors — without consuming quota
(uses the cheapest/lightest endpoint available for each service).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CheckResult:
    name: str                          # Display name, e.g. "Anthropic API"
    key: str                           # Internal key, e.g. "anthropic"
    ok: bool
    latency_ms: Optional[float] = None
    note: str = ""                     # Short status message
    error: str = ""                    # Error detail if not ok


def check_all(config) -> list[CheckResult]:
    """
    Run all API checks concurrently and return results.

    Args:
        config: speqtro Config instance (for api keys / settings).
    """
    import concurrent.futures

    checkers = [
        ("anthropic",  _check_anthropic),
        ("pubchem",    _check_pubchem),
        ("nmrshiftdb", _check_nmrshiftdb),
        ("sdbs",       _check_sdbs),
    ]

    results: list[CheckResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(checkers)) as pool:
        futures = {pool.submit(fn, config): key for key, fn in checkers}
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                key = futures[future]
                results.append(CheckResult(
                    name=key, key=key, ok=False,
                    error=f"Checker raised: {exc}",
                ))

    # Stable display order
    order = {k: i for i, (k, _) in enumerate(checkers)}
    results.sort(key=lambda r: order.get(r.key, 99))
    return results


# ── Individual checkers ────────────────────────────────────────────────────────

def _check_anthropic(config) -> CheckResult:
    """
    Probe the Anthropic API with a minimal 1-token completion.
    Uses claude-haiku (cheapest) to minimise cost (~$0.000025 per check).
    """
    api_key = config.llm_api_key("anthropic")
    if not api_key:
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return CheckResult(
            name="Anthropic API", key="anthropic", ok=False,
            note="No API key configured",
            error="Set ANTHROPIC_API_KEY or run: speqtro config set llm.api_key <key>",
        )

    try:
        import anthropic
    except ImportError:
        return CheckResult(
            name="Anthropic API", key="anthropic", ok=False,
            note="anthropic package not installed",
            error="pip install anthropic",
        )

    t0 = time.monotonic()
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        latency_ms = (time.monotonic() - t0) * 1000
        model_used = getattr(msg, "model", "claude-haiku-4-5-20251001")
        return CheckResult(
            name="Anthropic API", key="anthropic", ok=True,
            latency_ms=latency_ms,
            note=f"Auth OK  ·  model: {model_used}",
        )
    except anthropic.AuthenticationError:
        latency_ms = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="Anthropic API", key="anthropic", ok=False,
            latency_ms=latency_ms,
            note="Authentication failed",
            error="API key is invalid or expired. Run: speqtro config set llm.api_key <key>",
        )
    except anthropic.RateLimitError:
        latency_ms = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="Anthropic API", key="anthropic", ok=False,
            latency_ms=latency_ms,
            note="Rate limited",
            error="You have hit the rate limit. Try again in a moment.",
        )
    except anthropic.APIConnectionError as e:
        latency_ms = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="Anthropic API", key="anthropic", ok=False,
            latency_ms=latency_ms,
            note="Connection error",
            error=str(e),
        )
    except Exception as e:
        latency_ms = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="Anthropic API", key="anthropic", ok=False,
            latency_ms=latency_ms,
            note="Unexpected error",
            error=str(e),
        )


def _check_pubchem(config) -> CheckResult:
    """Probe PubChem REST API with a lightweight status endpoint."""
    import urllib.request
    import urllib.error

    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/property/MolecularFormula/JSON"
    t0 = time.monotonic()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "speqtro-cli/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            latency_ms = (time.monotonic() - t0) * 1000
            if resp.status == 200:
                return CheckResult(
                    name="PubChem", key="pubchem", ok=True,
                    latency_ms=latency_ms, note="Reachable (no key needed)",
                )
            return CheckResult(
                name="PubChem", key="pubchem", ok=False,
                latency_ms=latency_ms,
                note=f"HTTP {resp.status}",
                error=f"Unexpected status {resp.status}",
            )
    except urllib.error.URLError as e:
        latency_ms = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="PubChem", key="pubchem", ok=False,
            latency_ms=latency_ms, note="Unreachable",
            error=str(e.reason) if hasattr(e, "reason") else str(e),
        )
    except Exception as e:
        latency_ms = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="PubChem", key="pubchem", ok=False,
            latency_ms=latency_ms, note="Error", error=str(e),
        )


def _check_nmrshiftdb(config) -> CheckResult:
    """Probe NMRShiftDB2 with a lightweight search query."""
    import urllib.request
    import urllib.error

    url = "https://nmrshiftdb.nmr.uni-koeln.de/portal/media-type/html/user/anon/page/default.psml"
    t0 = time.monotonic()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "speqtro-cli/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            latency_ms = (time.monotonic() - t0) * 1000
            ok = resp.status == 200
            return CheckResult(
                name="NMRShiftDB2", key="nmrshiftdb", ok=ok,
                latency_ms=latency_ms,
                note="Reachable (no key needed)" if ok else f"HTTP {resp.status}",
            )
    except urllib.error.URLError as e:
        latency_ms = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="NMRShiftDB2", key="nmrshiftdb", ok=False,
            latency_ms=latency_ms, note="Unreachable",
            error=str(e.reason) if hasattr(e, "reason") else str(e),
        )
    except Exception as e:
        latency_ms = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="NMRShiftDB2", key="nmrshiftdb", ok=False,
            latency_ms=latency_ms, note="Error", error=str(e),
        )


def _check_sdbs(config) -> CheckResult:
    """Probe SDBS (SDBSWeb) — public spectral database, no key needed."""
    import urllib.request
    import urllib.error

    url = "https://sdbs.db.aist.go.jp/sdbs/cgi-bin/direct_frame_top.cgi"
    t0 = time.monotonic()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "speqtro-cli/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            latency_ms = (time.monotonic() - t0) * 1000
            ok = resp.status == 200
            return CheckResult(
                name="SDBS", key="sdbs", ok=ok,
                latency_ms=latency_ms,
                note="Reachable (no key needed)" if ok else f"HTTP {resp.status}",
            )
    except urllib.error.URLError as e:
        latency_ms = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="SDBS", key="sdbs", ok=False,
            latency_ms=latency_ms, note="Unreachable",
            error=str(e.reason) if hasattr(e, "reason") else str(e),
        )
    except Exception as e:
        latency_ms = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="SDBS", key="sdbs", ok=False,
            latency_ms=latency_ms, note="Error", error=str(e),
        )
