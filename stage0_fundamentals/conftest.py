"""pytest 根 conftest：让 `import stage0.xxx` 在 tests/ 里可用（把工程根加入 sys.path）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
