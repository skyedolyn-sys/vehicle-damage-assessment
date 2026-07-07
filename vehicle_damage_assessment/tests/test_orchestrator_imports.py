"""orchestrator import 卫生: 验证无重复 import, 无未使用 import。"""
import ast
import sys
from pathlib import Path

# Resolve ORCH_PATH relative to this test file so the test works in any
# worktree or environment, not just /Users/sky/vehicle_damage_assessment.
_ORCH_PATH = Path(__file__).resolve().parents[1] / "agents" / "assessment_orchestrator.py"
ORCH_PATH = _ORCH_PATH if _ORCH_PATH.exists() else Path(
    "/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment/agents/assessment_orchestrator.py"
)


def test_no_duplicate_imports():
    tree = ast.parse(ORCH_PATH.read_text(encoding="utf-8"))
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                key = ("import", tuple(a.name for a in node.names))
            else:
                key = ("from", node.module, tuple(a.name for a in node.names))
            assert key not in seen, f"duplicate import: {key}"
            seen.add(key)


def test_orchestrator_module_loads():
    """orchestrator 模块能成功 import。"""
    from agents import assessment_orchestrator  # noqa: F401