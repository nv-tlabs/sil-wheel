"""Doc-accuracy guard: the SKILL.md examples must be valid + reference real API.

Extracts every ```python block from SKILL.md, compiles it (syntax check), and
verifies every `client.<method>(...)` / `WheelClient.<method>(...)` it documents
actually exists on the SDK. Prevents the docs from drifting away from the code
(a documented-but-missing method is the worst footgun for an agent reading the
skill). No network, no server.

    pytest tests/test_skill_examples.py -q
"""
import re
from pathlib import Path

from sil_wheel_agent import WheelClient

SKILL = Path(__file__).resolve().parent.parent / "SKILL.md"
_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_CALL_RE = re.compile(r"\b(?:client|dev)\.(\w+)\(|\bWheelClient\.(\w+)\(")


def _python_blocks() -> list[str]:
    return _BLOCK_RE.findall(SKILL.read_text(encoding="utf-8"))


def test_skill_has_python_examples():
    assert len(_python_blocks()) >= 5


def test_skill_examples_are_valid_python():
    for i, block in enumerate(_python_blocks()):
        try:
            compile(block, f"<SKILL.md block {i}>", "exec")
        except SyntaxError as e:
            raise AssertionError(f"SKILL.md python block {i} has a syntax error: {e}\n{block}")


def test_skill_documents_only_real_methods():
    text = SKILL.read_text(encoding="utf-8")
    referenced = {m for pair in _CALL_RE.findall(text) for m in pair if m}
    missing = sorted(name for name in referenced if not hasattr(WheelClient, name))
    assert not missing, f"SKILL.md references SDK methods that do not exist: {missing}"
