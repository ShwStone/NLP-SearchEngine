"""零依赖测试 runner：`python tests/run.py`。装了 pytest 的话直接 pytest 也行。"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests import (test_contracts, test_hybrid, test_qdrant, test_offline_delivery,
                   test_scale_support)  # noqa: E402

ALL_TESTS = (test_contracts.ALL_TESTS + test_hybrid.ALL_TESTS +
             test_qdrant.ALL_TESTS + test_offline_delivery.ALL_TESTS +
             test_scale_support.ALL_TESTS)


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
