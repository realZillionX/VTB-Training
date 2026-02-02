"""Puzzle generation and evaluation toolkit."""

__all__ = [
    "AbstractPuzzleGenerator",
    "AbstractPuzzleEvaluator",
    "JigsawGenerator",
    "JigsawEvaluator",
    "JigsawPuzzleRecord",
    "JigsawEvaluationResult",
    "SudokuGenerator",
    "SudokuEvaluator",
    "SudokuPuzzleRecord",
    "SudokuEvaluationResult",
    "MirrorGenerator",
    "MirrorEvaluator",
    "MirrorPuzzleRecord",
    "MirrorEvaluationResult",
    "MirrorCellEvaluation",
    "ArcPuzzleGenerator",
    "ArcPuzzleEvaluator",
    "ArcPuzzleRecord",
    "ArcEvaluationResult",
    "ArcCellEvaluation",
    "MazeGenerator",
    "MazeEvaluator",
    "MazeLabyrinthGenerator",
    "MazeLabyrinthEvaluator",
    "MazeHexagonGenerator",
    "MazeHexagonEvaluator",
    "MazePuzzleRecord",
    "MazeEvaluationResult",
    "PieceEvaluation",
    "CellEvaluation",
    "RayGenerator",
    "RayEvaluator",
    "RayPuzzleRecord",
    "RayEvaluationResult",
    "RayIntersectionGenerator",
    "RayIntersectionEvaluator",
    "RayIntersectionPuzzleRecord",
    "RayIntersectionEvaluationResult",
    "MidpointGenerator",
    "MidpointEvaluator",
    "Segment",
    "ArcConnectGenerator",
    "ArcConnectEvaluator",
    "ArcConnectPuzzleRecord",
    "ArcConnectEvaluationResult",
    "CircleCountGenerator",
    "CircleCountEvaluator",
    "CircleCountPuzzleRecord",
    "CircleCountEvaluationResult",
    "transcribe_video",
]

from .base import AbstractPuzzleEvaluator, AbstractPuzzleGenerator
from .visual.jigsaw import (
    JigsawGenerator,
    JigsawEvaluator,
    JigsawPuzzleRecord,
    JigsawEvaluationResult,
    PieceEvaluation,
)
from .visual.sudoku import (
    SudokuGenerator,
    SudokuEvaluator,
    SudokuPuzzleRecord,
    SudokuEvaluationResult,
    CellEvaluation,
)
from .visual.mirror import (
    MirrorGenerator,
    MirrorEvaluator,
    MirrorPuzzleRecord,
    MirrorEvaluationResult,
    MirrorCellEvaluation,
)
from .visual.arcagi import (
    ArcPuzzleGenerator,
    ArcPuzzleEvaluator,
    ArcPuzzleRecord,
    ArcEvaluationResult,
    ArcCellEvaluation,
)
from .maze.maze_square import (
    MazeGenerator,
    MazeEvaluator,
    MazePuzzleRecord,
    MazeEvaluationResult,
)
from .maze.maze_labyrinth import (
    MazeLabyrinthGenerator,
    MazeLabyrinthEvaluator,
)
from .maze.maze_hexagon import (
    MazeHexagonGenerator,
    MazeHexagonEvaluator,
)
from .visual.rects import (
    RectsGenerator,
    RectsEvaluator,
    RectsPuzzleRecord,
    RectsEvaluationResult,
)

# Ray-and-mirrors (speak option via NATO)
from .eyeballing.ray import (
    RayGenerator,
    RayEvaluator,
    RayPuzzleRecord,
    RayEvaluationResult,
)

from .eyeballing.ray_intersection import (
    RayIntersectionGenerator,
    RayIntersectionEvaluator,
)


# Arc connection (speak option via NATO)
from .eyeballing.arc_connect import (
    ArcConnectGenerator,
    ArcConnectEvaluator,
    ArcConnectPuzzleRecord,
)

from .visual.circle_count import (
    CircleCountGenerator,
    CircleCountEvaluator,
    CircleCountPuzzleRecord,
    CircleCountEvaluationResult,
)

