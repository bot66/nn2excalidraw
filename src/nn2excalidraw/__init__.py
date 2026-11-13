"""Turn PyTorch modules into deterministic Excalidraw architecture diagrams."""

from nn2excalidraw.api import Diagram, diagram_from_path, diagram_from_source, parse_input_shape
from nn2excalidraw.errors import NN2ExcalidrawError

__all__ = [
    "Diagram",
    "NN2ExcalidrawError",
    "diagram_from_path",
    "diagram_from_source",
    "parse_input_shape",
]
__version__ = "0.1.0"
