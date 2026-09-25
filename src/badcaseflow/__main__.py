"""支持使用 `python -m badcaseflow` 启动命令行入口。"""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
