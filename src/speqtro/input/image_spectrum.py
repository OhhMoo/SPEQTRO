"""
Spectrum image parser using Claude's vision capabilities.

Extracts NMR peaks from spectrum screenshots, scanned PDFs, or images.
This is a best-effort extraction — shifts may have ±0.1-0.2 ppm uncertainty.

Requires ANTHROPIC_API_KEY to be set (uses the same key as the main agent).
"""

import base64
import json
import re
from pathlib import Path


# Claude model used for vision extraction (use Sonnet for cost/speed balance)
_VISION_MODEL = "claude-sonnet-4-6"

_EXTRACTION_PROMPT = """\
This is an NMR spectrum image. Extract all visible peaks as a JSON array.

For each peak, provide:
- "shift": chemical shift in ppm (read carefully from the x-axis; NMR axes have ppm decreasing left to right)
- "multiplicity": s/d/t/q/m/dd/dt/br s etc. (null if not labeled)
- "integral": approximate number of protons if integration curve is shown (null if not shown)
- "coupling_hz": J coupling constant in Hz if visible/labeled (null if not shown)

Important notes:
- The x-axis runs right to left (0 ppm on the right, higher ppm on the left)
- Read the axis scale carefully — do not guess
- Include solvent peaks if visible (e.g., CDCl3 at 7.26 ppm)
- If there is a peak table or labeled annotations in the image, prefer those over visual estimation

Return ONLY a valid JSON object with this structure:
{
  "nucleus": "1H" or "13C" etc. (guess from axis range and peak pattern),
  "solvent": "CDCl3" etc. if visible in image (or null),
  "frequency_mhz": spectrometer frequency if shown (or null),
  "peaks": [
    {"shift": 7.26, "multiplicity": "s", "integral": null, "coupling_hz": null},
    ...
  ]
}

Return ONLY the JSON, no other text."""


def parse(image_path: Path) -> dict:
    """
    Extract NMR peak data from a spectrum image using Claude vision.

    Args:
        image_path: Path to .png, .jpg, .jpeg, .pdf, or .tiff file.

    Returns:
        Normalized spectrum dict with extraction_confidence = "approximate".

    Raises:
        ImportError: If anthropic package is not installed.
        ValueError: If Claude returns unparseable JSON.
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError(
            "anthropic package is required for image spectrum parsing. "
            "Install it with: pip install anthropic"
        )

    image_path = Path(image_path)

    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode()

    suffix = image_path.suffix.lower()
    media_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".pdf": "application/pdf",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    media_type = media_type_map.get(suffix, "image/png")

    client = Anthropic()
    message = client.messages.create(
        model=_VISION_MODEL,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": _EXTRACTION_PROMPT,
                },
            ],
        }],
    )

    raw_text = message.content[0].text.strip()

    # Strip markdown code fences if present
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Claude returned non-JSON response for image extraction: {e}\n"
            f"Response: {raw_text[:500]}"
        )

    raw_peaks = result.get("peaks", [])
    peaks = []
    for p in raw_peaks:
        if "shift" not in p:
            continue
        coupling = p.get("coupling_hz")
        if isinstance(coupling, (int, float)):
            coupling = [float(coupling)]
        peaks.append({
            "shift": round(float(p["shift"]), 4),
            "intensity": None,
            "integral": p.get("integral"),
            "multiplicity": p.get("multiplicity"),
            "coupling_hz": coupling,
        })

    return {
        "peaks": sorted(peaks, key=lambda p: -p["shift"]),
        "nucleus": result.get("nucleus") or "1H",
        "solvent": result.get("solvent") or "unknown",
        "frequency_mhz": result.get("frequency_mhz"),
        "source_format": "image_extraction",
        "raw_spectrum": None,
        "ppm_scale": None,
        "extraction_confidence": "approximate",
        "_warning": (
            "Peaks extracted from image via vision LLM. "
            "Shifts may have ±0.1-0.2 ppm uncertainty. "
            "For high-confidence analysis, use instrument data files."
        ),
    }
