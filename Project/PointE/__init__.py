"""
Point-E integration module for Handsketch3D
"""

from .sketch_preprocessor import (
    SketchPreprocessor,
    QuickDrawPreprocessor,
    preprocess_for_pointe
)

try:
    from .pointe_inference import (
        PointEGenerator,
        image_to_pointcloud,
        POINT_E_AVAILABLE
    )
except ImportError:
    POINT_E_AVAILABLE = False
    PointEGenerator = None
    image_to_pointcloud = None

__all__ = [
    'SketchPreprocessor',
    'QuickDrawPreprocessor',
    'preprocess_for_pointe',
    'PointEGenerator',
    'image_to_pointcloud',
    'POINT_E_AVAILABLE'
]
