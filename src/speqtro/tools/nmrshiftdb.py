"""NMRshiftDB2 spectral database search tools.

NMRshiftDB2 (University of Cologne) provides ~54,000 peer-reviewed NMR spectra
with assigned structures under an open CC license.

Web service base: https://nmrshiftdb.nmr.uni-koeln.de
REST API docs:    https://nmrshiftdb.nmr.uni-koeln.de/nmrshiftdb/

Configuration:
  SPEQ_NMRSHIFTDB_URL  — override base URL (e.g. for a local mirror)
  SPEQ_NMRSHIFTDB_KEY  — API key if your instance requires one (sent as X-Api-Key header)
"""

import logging
import os
import xml.etree.ElementTree as ET
from typing import Optional

import httpx

from speqtro.tools import registry

logger = logging.getLogger("speqtro.tools.nmrshiftdb")

_DEFAULT_BASE = "https://nmrshiftdb.nmr.uni-koeln.de"


def _base_url() -> str:
    return os.environ.get("SPEQ_NMRSHIFTDB_URL", _DEFAULT_BASE).rstrip("/")


def _headers() -> dict:
    h = {"User-Agent": "speqtro-cli/1.0", "Accept": "application/xml, text/xml, */*"}
    key = os.environ.get("SPEQ_NMRSHIFTDB_KEY", "")
    if key:
        h["X-Api-Key"] = key
    return h


def _get(url: str, params: dict = None, timeout: int = 20) -> Optional[httpx.Response]:
    """GET with shared headers; returns Response or None on error."""
    try:
        resp = httpx.get(url, params=params, headers=_headers(), timeout=timeout, follow_redirects=True)
        return resp
    except Exception as e:
        logger.warning("NMRshiftDB2 request failed (%s): %s", url, e)
        return None


# ── XML helpers ───────────────────────────────────────────────────────────────

def _text(el: ET.Element, tag: str, default: str = "") -> str:
    child = el.find(tag)
    return child.text.strip() if child is not None and child.text else default


def _parse_spectrum_results(xml_text: str) -> list[dict]:
    """
    Parse NMRshiftDB2 spectrum-search XML response.

    Expected shape (simplified):
      <results>
        <result similarity="0.94">
          <spectrumid>12345</spectrumid>
          <molfile>...</molfile>   or <smiles>...</smiles>
          <name>Caffeine</name>
          <formula>C8H10N4O2</formula>
          <peaks>
            <peak shift="28.3" intensity="1"/>
            ...
          </peaks>
        </result>
        ...
      </results>
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning("XML parse error: %s", e)
        return []

    results = []
    # Support both <results><result> and bare <result> at root
    result_els = root.findall(".//result")
    if not result_els and root.tag == "result":
        result_els = [root]

    for el in result_els:
        similarity = float(el.get("similarity", el.get("score", "0")) or 0)
        spectrum_id = _text(el, "spectrumid") or _text(el, "spectrum_id") or el.get("id", "")
        compound_id = _text(el, "compoundid") or _text(el, "compound_id") or ""
        name = _text(el, "name") or _text(el, "compound_name") or "Unknown"
        formula = _text(el, "formula") or _text(el, "molecular_formula") or ""
        smiles = _text(el, "smiles") or _text(el, "SMILES") or ""
        inchikey = _text(el, "inchikey") or _text(el, "InChIKey") or ""

        # Collect peaks from the result
        peaks = []
        for p in el.findall(".//peak"):
            shift = p.get("shift") or p.get("ppm")
            intensity = p.get("intensity") or p.get("integral") or "1"
            if shift:
                try:
                    peaks.append({"shift": float(shift), "intensity": float(intensity)})
                except ValueError:
                    pass

        entry: dict = {
            "similarity": round(similarity, 4),
            "spectrum_id": spectrum_id,
            "compound_id": compound_id,
            "name": name,
            "formula": formula,
        }
        if smiles:
            entry["smiles"] = smiles
        if inchikey:
            entry["inchikey"] = inchikey
        if peaks:
            entry["matched_peaks"] = sorted(peaks, key=lambda x: x["shift"], reverse=True)
        if spectrum_id:
            entry["url"] = f"{_base_url()}/portal/media-type/html/user/anon/page/default.psml/js_pane/P-Spectrum/spectrum/{spectrum_id}"

        results.append(entry)

    return results


def _parse_compound_response(xml_text: str) -> dict:
    """Parse a compound/structure lookup XML response."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {}

    compound: dict = {
        "name": _text(root, "name") or _text(root, "compound_name") or "",
        "formula": _text(root, "formula") or "",
        "smiles": _text(root, "smiles") or _text(root, "SMILES") or "",
        "inchikey": _text(root, "inchikey") or "",
        "spectra": [],
    }

    for sp in root.findall(".//spectrum"):
        nucleus = sp.get("nucleus") or _text(sp, "nucleus") or ""
        sp_id = sp.get("id") or _text(sp, "id") or ""
        peaks = []
        for p in sp.findall(".//peak"):
            shift = p.get("shift") or p.get("ppm")
            if shift:
                try:
                    peaks.append(float(shift))
                except ValueError:
                    pass
        if nucleus or peaks:
            compound["spectra"].append({"nucleus": nucleus, "spectrum_id": sp_id, "shifts_ppm": sorted(peaks, reverse=True)})

    return compound


# ── Tools ─────────────────────────────────────────────────────────────────────

@registry.register(
    name="nmr.nmrshiftdb_search",
    description=(
        "Search NMRshiftDB2 (~54k peer-reviewed spectra) for compounds matching "
        "observed ¹³C or ¹H NMR peak positions. Returns ranked candidates with "
        "compound names, SMILES, similarity scores, and spectrum IDs. "
        "Ideal for validating ChefNMR candidates or identifying unknowns from spectra."
    ),
    category="nmr",
    parameters={
        "type": "object",
        "properties": {
            "c13_shifts": {
                "type": "string",
                "description": (
                    "Comma-separated ¹³C chemical shifts in ppm to search, e.g. '20.5, 128.3, 168.1'. "
                    "Provide c13_shifts or h1_shifts or both."
                ),
            },
            "h1_shifts": {
                "type": "string",
                "description": (
                    "Comma-separated ¹H chemical shifts in ppm to search, e.g. '7.26, 3.82, 2.31'. "
                    "Provide h1_shifts or c13_shifts or both."
                ),
            },
            "tolerance_ppm": {
                "type": "number",
                "description": "Matching tolerance in ppm (default: 2.0 for ¹³C, 0.1 for ¹H).",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 10).",
            },
        },
    },
)
def nmrshiftdb_search(
    c13_shifts: str = "",
    h1_shifts: str = "",
    tolerance_ppm: Optional[float] = None,
    max_results: int = 10,
    **kwargs,
) -> dict:
    """Search NMRshiftDB2 by ¹³C and/or ¹H peak positions."""
    if not c13_shifts and not h1_shifts:
        return {"error": "Provide at least one of c13_shifts or h1_shifts."}

    all_results: list[dict] = []
    errors: list[str] = []

    # ── ¹³C search ──────────────────────────────────────────────────────────
    if c13_shifts:
        try:
            shifts_c = [float(s.strip()) for s in c13_shifts.split(",") if s.strip()]
        except ValueError as e:
            return {"error": f"Could not parse c13_shifts: {e}"}

        tol_c = tolerance_ppm if tolerance_ppm is not None else 2.0
        shifts_csv = ",".join(f"{s:.4f}" for s in shifts_c)

        url = f"{_base_url()}/nmrshiftdb/webservice/spectrumSearch/C13/{shifts_csv}"
        params = {"threshold": str(tol_c), "mf": "", "numResults": str(max_results)}
        resp = _get(url, params=params)

        if resp is None:
            errors.append("¹³C search: HTTP request failed")
        elif resp.status_code != 200:
            errors.append(f"¹³C search: HTTP {resp.status_code} from {url}")
            logger.debug("¹³C response body: %s", resp.text[:500])
        else:
            parsed = _parse_spectrum_results(resp.text)
            for r in parsed:
                r["search_nucleus"] = "13C"
            all_results.extend(parsed)

    # ── ¹H search ───────────────────────────────────────────────────────────
    if h1_shifts:
        try:
            shifts_h = [float(s.strip()) for s in h1_shifts.split(",") if s.strip()]
        except ValueError as e:
            return {"error": f"Could not parse h1_shifts: {e}"}

        tol_h = tolerance_ppm if tolerance_ppm is not None else 0.1
        shifts_csv = ",".join(f"{s:.4f}" for s in shifts_h)

        url = f"{_base_url()}/nmrshiftdb/webservice/spectrumSearch/H1/{shifts_csv}"
        params = {"threshold": str(tol_h), "mf": "", "numResults": str(max_results)}
        resp = _get(url, params=params)

        if resp is None:
            errors.append("¹H search: HTTP request failed")
        elif resp.status_code != 200:
            errors.append(f"¹H search: HTTP {resp.status_code} from {url}")
            logger.debug("¹H response body: %s", resp.text[:500])
        else:
            parsed = _parse_spectrum_results(resp.text)
            for r in parsed:
                r["search_nucleus"] = "1H"
            all_results.extend(parsed)

    # ── Merge & rank ─────────────────────────────────────────────────────────
    # If both nuclei were searched, deduplicate by compound_id/name and
    # average the similarity scores.
    if c13_shifts and h1_shifts and all_results:
        merged: dict[str, dict] = {}
        for r in all_results:
            key = r.get("compound_id") or r.get("inchikey") or r["name"]
            if key in merged:
                existing = merged[key]
                existing["similarity"] = round((existing["similarity"] + r["similarity"]) / 2, 4)
                existing["search_nucleus"] = "13C+1H"
                if r.get("smiles") and not existing.get("smiles"):
                    existing["smiles"] = r["smiles"]
            else:
                merged[key] = dict(r)
        all_results = list(merged.values())

    all_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    all_results = all_results[:max_results]

    if not all_results and not errors:
        return {
            "summary": "No matching compounds found in NMRshiftDB2.",
            "results": [],
            "n_results": 0,
        }

    # ── Build summary ─────────────────────────────────────────────────────────
    lines = [f"NMRshiftDB2 search — {len(all_results)} result(s):"]
    for i, r in enumerate(all_results, 1):
        nucleus_tag = r.get("search_nucleus", "")
        sim_str = f"{r['similarity']:.2%}" if r.get("similarity") else "N/A"
        smiles_str = f"  SMILES: {r['smiles']}" if r.get("smiles") else ""
        lines.append(
            f"\n  [{i}] {r['name']} (sim={sim_str}, {nucleus_tag})"
            f"\n      formula={r.get('formula', '')}  id={r.get('spectrum_id', '')}"
        )
        if smiles_str:
            lines.append(f"     {smiles_str}")
        if r.get("url"):
            lines.append(f"      url: {r['url']}")

    return {
        "summary": "\n".join(lines),
        "results": all_results,
        "n_results": len(all_results),
        "errors": errors if errors else None,
        "source": "NMRshiftDB2",
    }


@registry.register(
    name="nmr.nmrshiftdb_fetch_compound",
    description=(
        "Fetch all NMR spectra stored in NMRshiftDB2 for a known compound, "
        "identified by name, SMILES, InChIKey, or NMRshiftDB2 compound ID. "
        "Returns all available ¹H and ¹³C peak lists for that compound."
    ),
    category="nmr",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Compound name, SMILES, InChIKey, or NMRshiftDB2 compound ID.",
            },
            "query_type": {
                "type": "string",
                "description": "One of: 'name' (default), 'smiles', 'inchikey', 'id'.",
            },
        },
        "required": ["query"],
    },
)
def nmrshiftdb_fetch_compound(
    query: str = "",
    query_type: str = "name",
    **kwargs,
) -> dict:
    """Fetch NMR spectra from NMRshiftDB2 for a specific compound."""
    if not query:
        return {"error": "Provide a query (name, SMILES, InChIKey, or compound ID)."}

    qt = (query_type or "name").strip().lower()

    # Map query type to NMRshiftDB2 web service endpoint
    endpoint_map = {
        "name": f"{_base_url()}/nmrshiftdb/webservice/compounds/name/{httpx.URL(query)}",
        "smiles": f"{_base_url()}/nmrshiftdb/webservice/compounds/smiles",
        "inchikey": f"{_base_url()}/nmrshiftdb/webservice/compounds/inchikey/{query}",
        "id": f"{_base_url()}/nmrshiftdb/webservice/compounds/{query}",
    }

    if qt not in endpoint_map:
        return {"error": f"Unknown query_type '{query_type}'. Use: name, smiles, inchikey, id."}

    if qt == "smiles":
        # SMILES needs URL-safe encoding as a query param
        import urllib.parse
        url = f"{_base_url()}/nmrshiftdb/webservice/compounds/smiles"
        params = {"smiles": query}
        resp = _get(url, params=params)
    elif qt == "name":
        import urllib.parse
        url = f"{_base_url()}/nmrshiftdb/webservice/compounds/name/{urllib.parse.quote(query, safe='')}"
        resp = _get(url)
    else:
        resp = _get(endpoint_map[qt])

    if resp is None:
        return {"error": "HTTP request to NMRshiftDB2 failed."}
    if resp.status_code == 404:
        return {"summary": f"No compound found in NMRshiftDB2 for '{query}'.", "results": []}
    if resp.status_code != 200:
        return {"error": f"NMRshiftDB2 returned HTTP {resp.status_code}."}

    compound = _parse_compound_response(resp.text)
    if not compound.get("name") and not compound.get("spectra"):
        return {"summary": f"No data parsed from NMRshiftDB2 response for '{query}'.", "raw_preview": resp.text[:400]}

    lines = [
        f"NMRshiftDB2 compound: {compound.get('name', query)}",
        f"  Formula: {compound.get('formula', 'N/A')}",
        f"  SMILES:  {compound.get('smiles', 'N/A')}",
        f"  InChIKey: {compound.get('inchikey', 'N/A')}",
        f"  Spectra: {len(compound.get('spectra', []))} stored",
    ]
    for sp in compound.get("spectra", []):
        shifts_str = ", ".join(f"{s:.2f}" for s in sp.get("shifts_ppm", [])[:12])
        if len(sp.get("shifts_ppm", [])) > 12:
            shifts_str += f" ... (+{len(sp['shifts_ppm']) - 12} more)"
        lines.append(f"\n  [{sp.get('nucleus', '?')} spectrum {sp.get('spectrum_id', '')}]")
        lines.append(f"    Peaks: {shifts_str}")

    return {
        "summary": "\n".join(lines),
        "compound": compound,
        "source": "NMRshiftDB2",
    }
