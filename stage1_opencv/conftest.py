"""pytest 根 conftest：把工程根加入 sys.path，使 `import stage1.xxx` 可用。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
