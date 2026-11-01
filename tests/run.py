"""零依赖测试 runner：`python tests/run.py`。装了 pytest 的话直接 pytest 也行。"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from importlib import import_module  # noqa: E402

MODULES = [import_module("tests." + path.stem)
           for path in sorted(Path(__file__).resolve().parent.glob("test_*.py"))]
ALL_TESTS = [fn for module in MODULES for fn in module.ALL_TESTS]


def main() -> int:
    failed = 0
    for fn in ALL_TESTS:
        try:
            fn()
            print(f"  通过  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  失败  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(ALL_TESTS) - failed}/{len(ALL_TESTS)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
