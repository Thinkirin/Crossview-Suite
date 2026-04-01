"""CrossViewer core modules."""

from .model import CrossViewerModel
from .modules import ART, OCVA, MaskPooling
from .losses import InfoNCELoss, TripletLoss
from .vision_encoder import Qwen3VLVisionEncoder

__all__ = [
    'CrossViewerModel',
    'ART',
    'OCVA',
    'MaskPooling',
    'InfoNCELoss',
    'TripletLoss',
    'Qwen3VLVisionEncoder',
]

__version__ = '0.1.0'
