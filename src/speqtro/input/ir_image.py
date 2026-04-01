"""
IR spectrum image parser using Claude's vision capabilities.

Extracts IR absorption peaks from spectrum screenshots, scanned PDFs, or images.
Returns (wavenumbers, intensities) arrays suitable for SSIN and a simplified peak list.

Limitations:
  - Shifts may have ±5–20 cm⁻¹ uncertainty depending on image resolution
  - Intensities are approximated from visual peak heights
  - Strongly prefer JDX files from instruments when available
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path


_VISION_MODEL = "claude-sonnet-4-6"

_IR_EXTRACTION_PROMPT = """\
This is an IR (infrared) spectrum image. Extract the absorption data as JSON.

IR spectrum conventions:
- X-axis: wavenumber in cm⁻¹, typically running from HIGH values (~4000) on the LEFT \
  to LOW values (~400-500) on the RIGHT.
- Y-axis is one of:
    * Transmittance (%T): absorptions appear as DOWNWARD peaks (troughs)
    * Absorbance (A): absorptions appear as UPWARD peaks

Key absorption regions to look for:
  3200–3550 cm⁻¹   O–H stretch (broad)
  3300–3500 cm⁻¹   N–H stretch
  2850–3000 cm⁻¹   C–H stretch (alkyl)
  3000–3100 cm⁻¹   C–H stretch (aromatic/vinyl)
  2100–2260 cm⁻¹   C≡N or C≡C stretch
  1650–1850 cm⁻¹   C=O stretch (strong, sharp)
  1500–1680 cm⁻¹   C=C / C=N
  1000–1300 cm⁻¹   C–O / C–N
  Below 1500 cm⁻¹  Fingerprint region

Instructions:
1. Read the axis labels carefully.
2. Identify all significant absorption bands (absorptions, not background).
3. If %T axis: report TROUGHS as absorptions (invert in your mind — dips = absorptions).
4. If the spectrum has labeled wavenumber annotations, use them for precision.
5. Estimate relative intensity 0–1 where 1 = strongest absorption in the spectrum.

Return ONLY a valid JSON object:
{
  "y_units": "transmittance" or "absorbance" or "unknown",
  "x_range": [min_wavenumber, max_wavenumber],
  "peaks": [
    {"wavenumber": 1715.0, "intensity": 0.95},
    {"wavenumber": 3300.0, "intensity": 0.60},
    ...
  ]
}

Sort peaks by intensity descending. Include all significant bands (typically 5–20 peaks).
Return ONLY the JSON, no other text."""


def parse(image_path: Path) -> dict:
    """
    Extract IR spectrum data from an image using Claude vision.

    Args:
        image_path: Path to .png, .jpg, .jpeg, .pdf, or .tiff file.

    Returns:
        {
            "wavenumbers":  list[float],   # cm⁻¹ of detected peaks
            "intensities":  list[float],   # normalised 0–1 per peak
            "peaks":        list[dict],    # [{"wavenumber": float, "intensity": float}, ...]
            "y_units":      str,
            "source_format": "ir_image_extraction",
            "extraction_confidence": "approximate",
            "_warning": str,
        }

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
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode()

    media_type_map = {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".pdf":  "application/pdf",
        ".tiff": "image/tiff",
        ".tif":  "image/tiff",
    }
    media_type = media_type_map.get(image_path.suffix.lower(), "image/png")

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
                {"type": "text", "text": _IR_EXTRACTION_PROMPT},
            ],
        }],
    )

    raw_text = message.content[0].text.strip()
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Claude returned non-JSON response for IR image extraction: {e}\n"
            f"Response: {raw_text[:500]}"
        )

    raw_peaks = result.get("peaks", [])
    peaks: list[dict] = []
    for p in raw_peaks:
        try:
            wn = float(p["wavenumber"])
            it = float(p.get("intensity", 0.5))
            if 200 <= wn <= 5000:
                peaks.append({
                    "wavenumber": round(wn, 1),
                    "intensity": round(min(1.0, max(0.0, it)), 4),
                })
        except (KeyError, ValueError, TypeError):
            continue

    # Sort descending by intensity for display; ascending by wavenumber for arrays
    wavenumbers = [p["wavenumber"] for p in sorted(peaks, key=lambda x: x["wavenumber"])]
    intensities = [p["intensity"] for p in sorted(peaks, key=lambda x: x["wavenumber"])]

    return {
        "wavenumbers": wavenumbers,
        "intensities": intensities,
        "peaks": peaks,
        "y_units": result.get("y_units", "unknown"),
        "x_range": result.get("x_range"),
        "source_format": "ir_image_extraction",
        "extraction_confidence": "approximate",
        "_warning": (
            "IR peaks extracted from image via vision LLM. "
            "Wavenumber accuracy: ±5–20 cm⁻¹ depending on image resolution. "
            "For high-confidence SSIN analysis, use the original JDX file."
        ),
    }
