"""Starlette web server for the speqtro GUI."""

import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path

logger = logging.getLogger("speqtro.web.server")

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

_STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Static file
# ---------------------------------------------------------------------------

async def index(request: Request):
    return FileResponse(str(_STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Config / tools
# ---------------------------------------------------------------------------

async def api_config(request: Request):
    import getpass, socket
    from speqtro.agent.config import Config
    cfg = Config.load()
    key = cfg.llm_api_key("anthropic") or ""
    masked = (key[:6] + "…" + key[-4:]) if len(key) > 10 else ("*" * len(key) if key else "")
    try:
        username = getpass.getuser()
    except Exception:
        username = "user"
    try:
        hostname = socket.gethostname().split(".")[0]
    except Exception:
        hostname = "localhost"
    return JSONResponse({
        "model": cfg.get("llm.model"),
        "provider": cfg.get("llm.provider"),
        "api_key_masked": masked,
        "api_key_set": bool(key),
        "username": username,
        "hostname": hostname,
    })


async def api_tools(request: Request):
    from speqtro.tools import registry, ensure_loaded
    ensure_loaded()
    tools = [
        {"name": t.name, "description": t.description, "category": t.category}
        for t in registry.list_tools()
    ]
    return JSONResponse(tools)


# ---------------------------------------------------------------------------
# Chat — SSE streaming
# ---------------------------------------------------------------------------

async def api_chat(request: Request):
    from sse_starlette.sse import EventSourceResponse
    from speqtro.agent.config import Config
    from speqtro.agent.session import Session
    from speqtro.agent.runner import AgentRunner
    from speqtro.web.leaderboard import log_analysis

    body = await request.json()
    query = body.get("query", "").strip()
    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)

    config = Config.load()
    session = Session(config=config, mode="batch")
    runner = AgentRunner(session, headless=True)

    queue: asyncio.Queue = asyncio.Queue()
    t0 = time.time()

    async def run():
        tool_calls_seen: list[str] = []
        try:
            async for event in runner.astream_run(query):
                if event["type"] == "tool":
                    tool_calls_seen.append(event["name"])
                await queue.put(event)
        except Exception as exc:
            await queue.put({"type": "error", "text": str(exc)})
            return
        duration = time.time() - t0
        cost = event.get("cost_usd", 0.0) if event.get("type") == "done" else 0.0
        try:
            log_analysis(
                mode="chat",
                query=query[:500],
                duration_s=round(duration, 2),
                tool_calls=len(tool_calls_seen),
                cost_usd=cost,
            )
        except Exception:
            pass
        await queue.put({"type": "done", "duration_s": round(duration, 2),
                         "cost_usd": cost, "tool_calls": tool_calls_seen})

    asyncio.create_task(run())

    async def event_gen():
        while True:
            item = await queue.get()
            yield {"data": json.dumps(item)}
            if item["type"] in ("done", "error"):
                break

    return EventSourceResponse(event_gen())


# ---------------------------------------------------------------------------
# Verify — shared helpers
# ---------------------------------------------------------------------------

def _cas_to_smiles(cas: str):
    """Resolve a CAS number to SMILES via PubChem REST API."""
    import urllib.request
    cas = cas.strip()
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{cas}/property/IsomericSMILES/JSON"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        return data["PropertyTable"]["Properties"][0]["IsomericSMILES"]
    except Exception:
        return None


def _resolve_structure(raw: str):
    """
    Accept either a SMILES string or a CAS registry number.
    CAS numbers match the pattern d{1,7}-dd-d.
    Returns (smiles_str, error_str).
    """
    import re
    raw = raw.strip()
    if re.match(r'^\d{1,7}-\d{2}-\d$', raw):
        smiles = _cas_to_smiles(raw)
        if not smiles:
            return None, f"Could not resolve CAS '{raw}' via PubChem."
        return smiles, None
    return raw, None


def _parse_peak_str(raw: str, nucleus: str) -> list:
    peaks = []
    for p in str(raw).split(","):
        p = p.strip()
        if p:
            try:
                peaks.append({"shift": float(p), "nucleus": nucleus})
            except ValueError:
                pass
    return peaks


async def _parse_upload(upload, hint: str = "auto") -> dict | None:
    """Save an UploadFile to a temp path and parse it with autodetect."""
    if upload is None or not getattr(upload, "filename", None):
        return None
    suffix = Path(upload.filename).suffix.lower() or ".tmp"
    content = await upload.read()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        from speqtro.input.autodetect import parse_any
        return parse_any(tmp_path, spectrum_hint=hint)
    except Exception as e:
        logger.warning("Failed to parse upload '%s' (hint=%s): %s", upload.filename, hint, e)
        return None
    finally:
        if tmp_path:
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _peaks_from_parsed(parsed: dict) -> list:
    """Extract peaks from a parse_any() result into verify_product format."""
    if not parsed:
        return []
    nucleus = parsed.get("nucleus", "1H")
    return [
        {
            "shift": float(p.get("shift", p.get("ppm", 0.0))),
            "nucleus": nucleus,
            "integral": p.get("integral"),
            "multiplicity": p.get("multiplicity"),
        }
        for p in parsed.get("peaks", [])
    ]


# ---------------------------------------------------------------------------
# Verify — JSON endpoint (manual peaks + optional CAS)
# ---------------------------------------------------------------------------

async def api_verify(request: Request):
    from speqtro.modes.verify import verify_product
    from speqtro.web.leaderboard import log_analysis

    body = await request.json()
    raw = body.get("smiles", "").strip()
    if not raw:
        return JSONResponse({"error": "Provide a SMILES string or CAS number."}, status_code=400)

    smiles, err = _resolve_structure(raw)
    if err:
        return JSONResponse({"error": err}, status_code=400)

    solvent = body.get("solvent", "CDCl3")
    observed_peaks = (
        _parse_peak_str(body.get("h1", ""), "1H")
        + _parse_peak_str(body.get("c13", ""), "13C")
    )

    t0 = time.time()
    result = verify_product(
        smiles=smiles,
        observed_peaks=observed_peaks,
        solvent=solvent,
        sm_smiles=body.get("sm_smiles") or None,
    )
    duration = time.time() - t0

    try:
        log_analysis(
            mode="verify",
            query=smiles,
            compound=result.get("compound_name"),
            confidence=result.get("confidence_percent"),
            verdict=result.get("verdict"),
            duration_s=round(duration, 2),
            tool_calls=0,
            cost_usd=0.0,
        )
    except Exception:
        pass

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Verify — multipart upload endpoint (NMR / IR / MS files)
# ---------------------------------------------------------------------------

async def api_verify_upload(request: Request):
    from speqtro.modes.verify import verify_product
    from speqtro.web.leaderboard import log_analysis

    form = await request.form()

    raw = (form.get("smiles") or "").strip()
    if not raw:
        return JSONResponse({"error": "Provide a SMILES string or CAS number."}, status_code=400)

    smiles, err = _resolve_structure(raw)
    if err:
        return JSONResponse({"error": err}, status_code=400)

    solvent = form.get("solvent", "CDCl3")
    sm_smiles = form.get("sm_smiles") or None

    # Parse all uploaded files (sent as repeated 'spec_file' fields).
    # parse_any() auto-detects NMR / IR / MS from content / extension.
    uploads = form.getlist("spec_file")
    parsed_files = await asyncio.gather(*[_parse_upload(u, hint="auto") for u in uploads])

    observed_peaks: list = []
    parsed_ms = None

    for parsed in parsed_files:
        if not parsed:
            continue
        stype = parsed.get("spectrum_type", "nmr")
        if stype == "nmr":
            observed_peaks.extend(_peaks_from_parsed(parsed))
        elif stype == "ms" and parsed_ms is None:
            parsed_ms = parsed  # keep first MS result for metadata

    # Merge any manually typed peaks too
    observed_peaks.extend(_parse_peak_str(form.get("h1", ""), "1H"))
    observed_peaks.extend(_parse_peak_str(form.get("c13", ""), "13C"))

    if not observed_peaks:
        return JSONResponse(
            {"error": "No NMR peaks found. Upload a NMR file (.jdx, .csv, .txt) or enter peaks manually."},
            status_code=400,
        )

    t0 = time.time()
    result = verify_product(
        smiles=smiles,
        observed_peaks=observed_peaks,
        solvent=solvent,
        sm_smiles=sm_smiles,
    )
    duration = time.time() - t0

    if parsed_ms:
        result["ms_upload"] = {
            "n_peaks": len(parsed_ms.get("peaks", [])),
            "precursor_mz": parsed_ms.get("precursor_mz"),
        }

    try:
        log_analysis(
            mode="verify",
            query=smiles,
            compound=result.get("compound_name"),
            confidence=result.get("confidence_percent"),
            verdict=result.get("verdict"),
            duration_s=round(duration, 2),
            tool_calls=0,
            cost_usd=0.0,
        )
    except Exception:
        pass

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Predict endpoints
# ---------------------------------------------------------------------------

# Priority-ordered tool names per mode: ML tools first, rule-based fallbacks last.
_PREDICT_TOOL_PRIORITY = {
    "predict_h1":  ["nmr.predict_h1_nmrshiftdb",  "nmr.predict_h1_shifts"],
    "predict_c13": ["nmr.predict_c13_nmrshiftdb", "nmr.predict_c13_shifts"],
    "predict_ir":  ["ir.predict_absorptions"],
    "predict_ms":  ["ms.predict_msms_iceberg",     "ms.fragment_predict"],
}


def _normalize_predict_result(result: dict, mode: str) -> dict:
    """Remap tool output keys to the frontend-expected schema."""
    out = dict(result)

    if mode in ("predict_h1", "predict_c13"):
        raw_preds = result.get("predictions", [])
        # NMRshiftDB format: {atom_id, min, mean, max}
        # Empirical format:  {estimated_shift_ppm, range_ppm, num_H, environment}
        if raw_preds and "mean" in raw_preds[0]:
            out["shifts"] = [
                {
                    "shift": p.get("mean"),
                    "range": f"{p['min']}–{p['max']}",
                    "count": 1,
                    "environment": f"atom {p.get('atom_id', '?')}",
                }
                for p in raw_preds
            ]
        else:
            out["shifts"] = [
                {
                    "shift": p.get("estimated_shift_ppm"),
                    "range": p.get("range_ppm"),
                    "count": p.get("num_H", 1),
                    "environment": p.get("environment", ""),
                }
                for p in raw_preds
            ]

    elif mode == "predict_ir":
        out["bands"] = [
            {
                "wavenumber": b.get("wavenumber_range"),
                "intensity": b.get("intensity", ""),
                "assignment": b.get("functional_group", ""),
            }
            for b in result.get("bands", [])
        ]

    elif mode == "predict_ms":
        # ICEBERG format: predicted_peaks [{mz, intensity}]
        if result.get("predicted_peaks") is not None:
            out["fragments"] = [
                {
                    "mz": p.get("mz"),
                    "neutral_loss": f"{p['intensity']:.3f}",
                    "description": "predicted fragment ion",
                }
                for p in result.get("predicted_peaks", [])
            ]
        else:
            # Rule-based format
            losses = [
                {
                    "mz": f.get("fragment_mz"),
                    "neutral_loss": f.get("neutral_loss"),
                    "description": f.get("functional_group", ""),
                }
                for f in result.get("predicted_neutral_losses", [])
            ]
            chars = [
                {
                    "mz": f.get("mz"),
                    "neutral_loss": "—",
                    "description": f"{f.get('formula', '')} — {f.get('description', '')}",
                }
                for f in result.get("characteristic_fragments", [])
            ]
            out["fragments"] = losses + chars

    return out


async def _predict(request: Request, mode: str):
    from speqtro.tools import registry, ensure_loaded
    from speqtro.web.leaderboard import log_analysis

    body = await request.json()
    raw = body.get("smiles", "").strip()
    if not raw:
        return JSONResponse({"error": "SMILES or CAS number is required"}, status_code=400)

    smiles, err = _resolve_structure(raw)
    if err:
        return JSONResponse({"error": err}, status_code=400)

    ensure_loaded()

    tool_names = _PREDICT_TOOL_PRIORITY.get(mode, [])
    result = None
    used_tool = None
    last_error = "No prediction tools available for this mode."

    t0 = time.time()
    for tool_name in tool_names:
        tool = registry.get_tool(tool_name)
        if tool is None:
            continue
        try:
            res = tool.run(smiles=smiles)
        except Exception as exc:
            last_error = str(exc)
            logger.warning("Tool %s raised: %s", tool_name, exc)
            continue
        # Tool-level errors come back as {"error": "..."} or {"summary": "Error: ..."}
        if res.get("error") or res.get("summary", "").startswith("Error:"):
            last_error = res.get("error") or res.get("summary")
            continue
        result = res
        used_tool = tool_name
        break

    duration = time.time() - t0

    if result is None:
        return JSONResponse({"error": last_error}, status_code=400)

    result = _normalize_predict_result(result, mode)
    result["_tool_used"] = used_tool

    try:
        log_analysis(mode=mode, query=smiles, duration_s=round(duration, 2))
    except Exception:
        pass

    return JSONResponse(result)


async def api_predict_h1(request: Request):
    return await _predict(request, "predict_h1")


async def api_predict_c13(request: Request):
    return await _predict(request, "predict_c13")


async def api_predict_ir(request: Request):
    return await _predict(request, "predict_ir")


async def api_predict_ms(request: Request):
    return await _predict(request, "predict_ms")


# ---------------------------------------------------------------------------
# History / stats
# ---------------------------------------------------------------------------

async def api_history(request: Request):
    from speqtro.web.leaderboard import get_history
    limit = int(request.query_params.get("limit", 50))
    return JSONResponse(get_history(limit=limit))


async def api_stats(request: Request):
    from speqtro.web.leaderboard import get_stats
    return JSONResponse(get_stats())


# ---------------------------------------------------------------------------
# Doctor
# ---------------------------------------------------------------------------

async def api_doctor(request: Request):
    from speqtro.agent.config import Config
    from speqtro.tools import registry, ensure_loaded

    config = Config.load()
    ensure_loaded()

    tools = [t.name for t in registry.list_tools()]

    issues = config.llm_preflight_issue() if hasattr(config, "llm_preflight_issue") else None

    return JSONResponse({
        "api_key_set": bool(config.llm_api_key("anthropic")),
        "model": config.get("llm.model"),
        "tools_loaded": len(tools),
        "tool_names": tools,
        "issue": issues,
    })


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

routes = [
    Route("/", index),
    Mount("/static", app=StaticFiles(directory=str(_STATIC_DIR)), name="static"),
    Route("/api/config", api_config),
    Route("/api/tools", api_tools),
    Route("/api/chat", api_chat, methods=["POST"]),
    Route("/api/verify", api_verify, methods=["POST"]),
    Route("/api/verify/upload", api_verify_upload, methods=["POST"]),
    Route("/api/predict/h1", api_predict_h1, methods=["POST"]),
    Route("/api/predict/c13", api_predict_c13, methods=["POST"]),
    Route("/api/predict/ir", api_predict_ir, methods=["POST"]),
    Route("/api/predict/ms", api_predict_ms, methods=["POST"]),
    Route("/api/history", api_history),
    Route("/api/stats", api_stats),
    Route("/api/doctor", api_doctor),
]

app = Starlette(routes=routes)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
