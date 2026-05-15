"""
label_remap.py
--------------
HiRISE v3 landmark class → risk score remapping.

Blueprint §5.3 (HiRISE Map-Proj-v3 — Cross-Domain Evaluation):
  The eight real classes each map to a specific risk score that reflects
  the physical danger to a Mars rover.  "edge_case" does NOT exist in
  the actual dataset — never reference it.

Exports:
    HIRISE_V3_RISK_MAP   — dict {class_name: risk_score}
    HIRISE_V3_CLASS_IDS  — dict {class_name: integer_id} matching classmap CSV
    remap_label_id       — int class id → risk score (float)
    remap_label_name     — class name string → risk score (float)
    build_risk_label_tensor  — batch of integer label ids → float risk tensor
"""

import torch

# ---------------------------------------------------------------------------
# Risk map — blueprint §5.3 Table
# ---------------------------------------------------------------------------

HIRISE_V3_RISK_MAP: dict[str, float] = {
    "other":          0.15,  # Generic flat terrain
    "crater":         0.90,  # Rim and interior: entrapment and slope hazard
    "dark_dune":      0.85,  # Fine sand: high slip and entrapment
    "slope_streak":   0.80,  # Mass movement indicator: unstable terrain
    "bright_dune":    0.50,  # Variable composition: moderate slip
    "impact_ejecta":  0.55,  # Scattered rock: passable but rough
    "swiss_cheese":   0.85,  # CO2 sublimation pits: structurally unsafe
    "spider":         0.45,  # Dendritic erosion: rough but often passable
}

# Integer class IDs as used in the Wagstaff et al. dataset
# (0-indexed, matching the label text file ordering).
HIRISE_V3_CLASS_IDS: dict[str, int] = {
    "other":         0,
    "crater":        1,
    "dark_dune":     2,
    "slope_streak":  3,
    "bright_dune":   4,
    "impact_ejecta": 5,
    "swiss_cheese":  6,
    "spider":        7,
}

# Reverse map: id → name
_ID_TO_NAME: dict[int, str] = {v: k for k, v in HIRISE_V3_CLASS_IDS.items()}

# Lookup tensor for fast batch remapping (index with class id → risk score)
_RISK_LUT: torch.Tensor = torch.zeros(len(HIRISE_V3_CLASS_IDS), dtype=torch.float32)
for _name, _id in HIRISE_V3_CLASS_IDS.items():
    _RISK_LUT[_id] = HIRISE_V3_RISK_MAP[_name]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def remap_label_id(class_id: int) -> float:
    """Map integer class id to risk score.

    Parameters
    ----------
    class_id : int
        Integer in [0, 7] matching HIRISE_V3_CLASS_IDS.

    Returns
    -------
    float risk score in [0, 1]

    Raises
    ------
    KeyError if class_id is not in the dataset.
    """
    name = _ID_TO_NAME.get(class_id)
    if name is None:
        raise KeyError(
            f"Unknown HiRISE v3 class id {class_id!r}. "
            f"Valid ids: {list(_ID_TO_NAME.keys())}"
        )
    return HIRISE_V3_RISK_MAP[name]


def remap_label_name(class_name: str) -> float:
    """Map class name string to risk score.

    Parameters
    ----------
    class_name : str
        One of the eight real class names (lower-case, underscores).

    Returns
    -------
    float risk score in [0, 1]

    Raises
    ------
    KeyError if class_name is not in HIRISE_V3_RISK_MAP.
    """
    if class_name not in HIRISE_V3_RISK_MAP:
        raise KeyError(
            f"Unknown HiRISE v3 class name {class_name!r}. "
            f"Valid names: {list(HIRISE_V3_RISK_MAP.keys())}"
        )
    return HIRISE_V3_RISK_MAP[class_name]


def build_risk_label_tensor(class_ids: torch.Tensor) -> torch.Tensor:
    """Convert a batch of integer class ids to risk score tensor.

    Uses a pre-built look-up table for O(1) batch remapping without Python
    loops.

    Parameters
    ----------
    class_ids : torch.Tensor, shape (N,), dtype int64 or int32
        Integer class ids in [0, 7].

    Returns
    -------
    torch.Tensor, shape (N,), dtype float32
        Risk scores in [0, 1].

    Example
    -------
    >>> ids = torch.tensor([1, 0, 2])  # crater, other, dark_dune
    >>> build_risk_label_tensor(ids)
    tensor([0.9000, 0.1500, 0.8500])
    """
    return _RISK_LUT[class_ids.long()]


def get_class_names() -> list[str]:
    """Return ordered list of class names matching integer IDs 0–7."""
    return [_ID_TO_NAME[i] for i in range(len(_ID_TO_NAME))]


def is_hazardous(class_id: int, threshold: float = 0.7) -> bool:
    """Return True if the class risk score exceeds threshold."""
    return remap_label_id(class_id) > threshold
