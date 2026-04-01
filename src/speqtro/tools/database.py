"""Chemical database lookup tools."""

import json
import logging

import httpx

from speqtro.tools import registry

logger = logging.getLogger("speqtro.tools.database")

_PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def _pubchem_get(url: str) -> dict | None:
    try:
        resp = httpx.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception as e:
        logger.warning("PubChem request failed: %s", e)
        return None


@registry.register(
    name="database.pubchem_search",
    description=(
        "Search PubChem for a compound by name, SMILES, InChI, or InChIKey. "
        "Returns CID, IUPAC name, molecular formula, canonical SMILES, exact mass, and known spectra links."
    ),
    category="database",
    parameters={
        "query": "Compound name, SMILES string, InChI, or InChIKey to search for",
        "query_type": "Type of query: 'name' (default), 'smiles', 'inchi', 'inchikey', 'formula'",
    },
)
def pubchem_search(query: str = "", query_type: str = "name", **kwargs) -> dict:
    """Search PubChem PUG REST API for compound information."""
    if not query:
        return {"summary": "Error: no query provided"}

    qt = query_type.strip().lower() if query_type else "name"
    # Map to PubChem namespace
    namespace_map = {
        "name": "name",
        "smiles": "smiles",
        "inchi": "inchi",
        "inchikey": "inchikey",
        "formula": "formula",
    }
    namespace = namespace_map.get(qt, "name")

    # Step 1: get CID
    import urllib.parse
    encoded_query = urllib.parse.quote(query, safe="")
    url = f"{_PUBCHEM_BASE}/compound/{namespace}/{encoded_query}/cids/JSON"
    data = _pubchem_get(url)

    if not data or "IdentifierList" not in data:
        # Try name search as fallback
        if namespace != "name":
            url2 = f"{_PUBCHEM_BASE}/compound/name/{encoded_query}/cids/JSON"
            data = _pubchem_get(url2)
        if not data or "IdentifierList" not in data:
            return {"summary": f"No PubChem results found for '{query}'"}

    cids = data["IdentifierList"].get("CID", [])
    if not cids:
        return {"summary": f"No PubChem CIDs found for '{query}'"}

    cid = cids[0]

    # Step 2: get properties
    props_url = (
        f"{_PUBCHEM_BASE}/compound/cid/{cid}/property/"
        "IUPACName,MolecularFormula,MolecularWeight,ExactMass,"
        "CanonicalSMILES,IsomericSMILES,InChI,InChIKey/JSON"
    )
    props_data = _pubchem_get(props_url)

    if not props_data:
        return {
            "summary": f"Found CID {cid} for '{query}' but could not fetch properties.",
            "cid": cid,
        }

    props = props_data.get("PropertyTable", {}).get("Properties", [{}])[0]

    iupac = props.get("IUPACName", "")
    formula = props.get("MolecularFormula", "")
    mw = props.get("MolecularWeight", "")
    exact = props.get("ExactMass", "")
    smiles = props.get("CanonicalSMILES", "")
    inchikey = props.get("InChIKey", "")

    # Step 3: check for spectra
    spectra_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/categories/compound/{cid}/JSON"
    spectra_info = "Check PubChem for spectra: " + f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}#section=Spectra"

    summary_lines = [
        f"PubChem CID: {cid}",
        f"IUPAC Name: {iupac}",
        f"Formula: {formula}",
        f"Molecular Weight: {mw} Da",
        f"Exact Mass: {exact} Da",
        f"SMILES: {smiles}",
        f"InChIKey: {inchikey}",
        spectra_info,
    ]

    return {
        "summary": "\n".join(l for l in summary_lines if l.split(": ", 1)[-1]),
        "cid": cid,
        "iupac_name": iupac,
        "formula": formula,
        "molecular_weight": mw,
        "exact_mass": exact,
        "smiles": smiles,
        "inchikey": inchikey,
        "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
        "spectra_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}#section=Spectra",
    }
