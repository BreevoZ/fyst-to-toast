"""只读 HDF5 全屏浏览器；Tab 文件树，Enter 完整值，m metadata，q 恢复 terminal.

python h5reader.py data/fyst_daisy_003_000.h5
python h5reader.py file.h5 --tree
python h5reader.py file.h5 --node /shared/boresight_azel
python h5reader.py file.h5 --view text --rows 5
"""

import argparse
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
from _hdf5_table import attributes, table_view
from astropy.table import Table
from ecsv_reader import GridViewer, detail_text, one_line, pad, paint, scalar, shorten


class ToTree(Exception):
    """Return from a table to the surrounding file browser."""


class H5Grid(GridViewer):
    """Reuse ECSV navigation and cell inspection on lazily loaded HDF5 data."""

    def draw(self):
        """Label the source accurately and expose the file-tree shortcut."""
        visible, page = super().draw()
        height, width = self.screen.getmaxyx()
        if height >= 12 and width >= 40:
            self.screen.move(0, 0)
            self.screen.clrtoeol()
            self.put(
                0,
                0,
                f"HDF5  {self.path.name} · {self.table.title} · {len(self.table):,} rows",
                self.curses.A_BOLD,
            )
            self.screen.move(height - 1, 0)
            self.screen.clrtoeol()
            self.put(
                height - 1,
                0,
                "↑↓←→ / Space/b / g / 搜索 / Enter 全文 · Tab 文件树 · q 退出",
                self.curses.A_REVERSE,
            )
            self.screen.refresh()
        return visible, page

    def handle_key(self, key):
        """Reserve Tab for the HDF5 tree; arrows still move between columns."""
        if key == "\t":
            raise ToTree
        return False


class Browser:
    """Navigate groups and datasets without loading complete arrays."""

    def __init__(self, screen, curses, file, path, args, color):
        self.screen, self.curses, self.file = screen, curses, file
        self.path, self.args, self.color = path, args, color
        self.expanded = {"/", "/shared"}
        self.row, self.top = 0, 0
        self.utility = GridViewer(screen, curses, Table(), path, args, color)
        self.notice = "Enter 打开 / 展开 · m attributes · d detector · / 找路径 · ? 帮助"

    def entries(self):
        """Traverse hard links, leaving soft/external/cyclic links unopened."""
        result = (
            [("trajectory", "view", "time / Az / El / scan / flags", 0)]
            if "shared/times" in self.file
            else []
        )

        def visit(group, depth, ancestors):
            result.append(
                (group.name, "group", f"{len(group)} entries; {len(group.attrs)} attrs", depth)
            )
            if group.name not in self.expanded:
                return
            for name in group:
                path = group.name.rstrip("/") + "/" + name
                link = group.get(name, getlink=True)
                if isinstance(link, (h5py.SoftLink, h5py.ExternalLink)):
                    result.append((path, "link", str(link), depth + 1))
                    continue
                obj = group[name]
                if isinstance(obj, h5py.Group):
                    address = h5py.h5o.get_info(obj.id).addr
                    if address in ancestors:
                        result.append((path, "link", "cyclic hard link", depth + 1))
                    else:
                        visit(obj, depth + 1, ancestors | {address})
                else:
                    dtype = f"record[{len(obj.dtype.names)}]" if obj.dtype.names else str(obj.dtype)
                    result.append((path, "dataset", f"shape={obj.shape}  {dtype}", depth + 1))

        visit(self.file, 0, {h5py.h5o.get_info(self.file.id).addr})
        return result

    def open_table(self, node):
        """Run a grid in the same alternate screen; Tab returns to the tree."""
        try:
            table = table_view(self.file, node, detector=self.args.detector)
            columns = self.args.columns if node == self.args.node else None
            if columns and any(name not in table.colnames for name in columns):
                raise ValueError(f"列名不存在；可选：{', '.join(table.colnames)}")
            args = SimpleNamespace(**vars(self.args))
            args.columns = columns
            args.start = min(args.start, max(0, len(table) - 1))
            H5Grid(self.screen, self.curses, table, self.path, args, self.color).run()
            return "exit"
        except ToTree:
            return "tree"
        except (KeyError, ValueError, TypeError, OSError) as error:
            self.utility.popup("无法显示这个节点", str(error))
            return "tree"

    def run(self):
        """Start on trajectory columns, then allow exploration of the native schema."""
        if not self.args.tree and self.open_table(self.args.node) == "exit":
            return
        while True:
            entries = self.entries()
            height, width = self.screen.getmaxyx()
            page = max(1, height - 6)
            self.row = max(0, min(self.row, len(entries) - 1))
            self.top = max(0, min(self.top, self.row))
            if self.row >= self.top + page:
                self.top = self.row - page + 1
            self.screen.erase()
            put = self.utility.put
            put(0, 0, f"HDF5 tree · {self.path.name} · read-only", self.curses.A_BOLD)
            put(1, 0, f"detector: {self.args.detector} [d]   Tab: trajectory   m: attributes")
            for offset, (path, kind, info, depth) in enumerate(entries[self.top : self.top + page]):
                name = path.rsplit("/", 1)[-1] or "/"
                marker = "▼" if path in self.expanded else "▶" if kind == "group" else "·"
                style = self.curses.A_REVERSE if self.top + offset == self.row else 0
                put(offset + 3, 0, f"{'  ' * depth}{marker} {name}  {info}", style)
            selected, kind, _, _ = entries[self.row]
            put(height - 3, 0, selected, self.curses.A_BOLD)
            put(height - 2, 0, self.notice)
            put(
                height - 1,
                0,
                "↑↓ / Space/b 翻页 · → 展开 · ← 收起 · Enter 打开 · q 退出",
                self.curses.A_REVERSE,
            )
            self.screen.refresh()
            key = self.screen.get_wch()
            if key in ("q", "\x1b"):
                return
            if key == "\x0c":
                self.screen.clearok(True)
            elif key in (self.curses.KEY_DOWN, "j"):
                self.row += 1
            elif key in (self.curses.KEY_UP, "k"):
                self.row -= 1
            elif key in (self.curses.KEY_NPAGE, " "):
                self.row += page
            elif key in (self.curses.KEY_PPAGE, "b"):
                self.row -= page
            elif key in (self.curses.KEY_HOME, "H"):
                self.row = 0
            elif key in (self.curses.KEY_END, "G"):
                self.row = len(entries) - 1
            elif key == self.curses.KEY_RIGHT and kind == "group":
                self.expanded.add(selected)
            elif key == self.curses.KEY_LEFT:
                self.expanded.discard(selected)
            elif key in ("\n", "\r", self.curses.KEY_ENTER):
                if kind == "group" and selected != "/shared":
                    self.expanded.symmetric_difference_update({selected})
                elif kind != "link" and self.open_table(selected) == "exit":
                    return
            elif key == "\t" and "shared/times" in self.file:
                if self.open_table("trajectory") == "exit":
                    return
            elif key == "m" and kind != "link":
                meta = attributes(self.file if selected == "trajectory" else self.file[selected])
                self.utility.popup(f"Attributes · {selected}", detail_text("attributes", meta))
            elif key == "/":
                query = self.utility.prompt("查找当前展开的路径：")
                if query:
                    found = next(
                        (
                            i
                            for i in range(self.row + 1, self.row + len(entries) + 1)
                            if query.casefold() in entries[i % len(entries)][0].casefold()
                        ),
                        None,
                    )
                    if found is not None:
                        self.row = found % len(entries)
                    else:
                        self.notice = f"没找到 {query!r}；可以先展开 group"
            elif key == "d":
                value = self.utility.prompt("Detector index（0-based）：")
                if value is not None:
                    try:
                        self.args.detector = max(0, int(value))
                    except ValueError:
                        self.notice = "请输入整数"
            elif key == "?":
                self.utility.popup(
                    "HDF5 reader",
                    "默认表格为 trajectory 的 degree 视图。\n"
                    "Tab 文件树；选择 shared 可以看全部原始列。\n"
                    "选择 dataset 后 Enter 打开；向量展开成分量列。\n"
                    "detdata 按 detector 切片，d 选择 detector 下标。\n"
                    "m 查看节点 attributes，包含完整 JSON。\n"
                    "q / Esc / Ctrl+C 退出，恢复原 terminal。\n"
                    "每列按 256 行缓存，不会一次读取整个大 dataset。",
                )


def main(argv=None):
    """Open only in read mode and restore the terminal after every exit path."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--view", choices=["grid", "text"], default="grid")
    parser.add_argument("--node", default="trajectory", help="trajectory 或真实 dataset/group 路径")
    parser.add_argument("--tree", action="store_true")
    parser.add_argument("--detector", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--columns", help="显示列，逗号分隔")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--precision", type=int, default=6)
    parser.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    args = parser.parse_args(argv)
    if (
        args.start < 0
        or args.detector < 0
        or (args.rows is not None and args.rows < 0)
        or not 1 <= args.precision <= 17
    ):
        parser.error("start / detector / rows 不可为负；precision 应为 1–17")
    args.columns = args.columns.split(",") if args.columns else None
    args.all, args.row, args.grid_rows = False, None, args.rows
    args.cell_width, args.start_at, args.details = 28, "data", False
    color = args.color == "always" or (
        args.color == "auto" and sys.stdout.isatty() and "NO_COLOR" not in os.environ
    )
    try:
        with h5py.File(args.path, "r") as file:
            if args.node == "trajectory" and "shared/times" not in file:
                args.tree = True
            if args.view == "grid" and sys.stdin.isatty() and sys.stdout.isatty():
                import curses
                import termios

                settings = termios.tcgetattr(sys.stdin.fileno())
                curses.set_escdelay(50)

                def interrupted(signum, frame):
                    raise KeyboardInterrupt

                previous = signal.signal(signal.SIGTERM, interrupted)
                try:
                    curses.wrapper(
                        lambda screen: Browser(screen, curses, file, args.path, args, color).run()
                    )
                finally:
                    termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, settings)
                    signal.signal(signal.SIGTERM, previous)
            elif args.tree:
                print(f"HDF5 · {args.path.name} · read-only")
                file.visititems(
                    lambda name, obj: print(f"/{name}  {getattr(obj, 'shape', 'group')}")
                )
                print(detail_text("Root attributes", attributes(file)))
            else:
                table = table_view(file, args.node, args.detector)
                names = args.columns or table.colnames
                if any(name not in table.colnames for name in names):
                    raise ValueError(f"可选列：{', '.join(table.colnames)}")
                print(f"HDF5 · {args.path.name} · {table.title} · {len(table):,} rows")
                print("  ".join(pad(shorten(name, 24), 24) for name in names))
                print(
                    "  ".join(pad(f"[{table[name].unit or 'not declared'}]", 24) for name in names)
                )
                for row in range(
                    args.start,
                    min(len(table), args.start + (50 if args.rows is None else args.rows)),
                ):
                    values = [
                        one_line(scalar(table[name][row], args.precision, args.full))
                        for name in names
                    ]
                    print(
                        "  ".join(
                            paint(
                                pad(value if args.full else shorten(value, 24), 24),
                                ("8ab4f8", "81c995", "fdd663")[i % 3],
                                color,
                            )
                            for i, value in enumerate(values)
                        )
                    )
    except KeyboardInterrupt:
        return 130
    except (OSError, KeyError, ValueError, TypeError) as error:
        parser.exit(2, f"HDF5 reader: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
