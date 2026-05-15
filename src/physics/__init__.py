"""
src/physics/__init__.py
-----------------------
Public API for Stage 2 Physics Feature Engine.

Quick import:
    from src.physics import PhysicsFeatureEngine, build_physics_engine_from_config
"""

from src.physics.combine import PhysicsFeatureEngine, build_physics_engine_from_config
from src.physics.discontinuity import DiscontinuityProxy
from src.physics.roughness import RoughnessProxy
from src.physics.slope import SlopeProxy

__all__ = [
    "PhysicsFeatureEngine",
    "build_physics_engine_from_config",
    "SlopeProxy",
    "RoughnessProxy",
    "DiscontinuityProxy",
]
