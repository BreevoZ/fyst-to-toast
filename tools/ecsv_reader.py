"""Browse ECSV in a full-screen colored table, without leaving rows in scrollback.

Examples
--------
python ecsv_reader.py schedule.ecsv
python ecsv_reader.py samples.ecsv --view text --pager never
python ecsv_reader.py schedule.ecsv --row 0 --details
python ecsv_reader.py samples.ecsv --start 20 --rows 10
python ecsv_reader.py schedule.ecsv --html preview.html

Arrow keys select cells; Space / b or PgDn / PgUp turn pages. Enter opens a
complete cell; m shows metadata; u shows column units; g jumps to a row.
The table keeps its headers visible and q restores the previous terminal screen.
Non-interactive output and --view text retain the original aligned CSV format.
This is a display only; the source ECSV is never modified.
"""

import argparse
import html
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import unicodedata
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import Table

PALETTE = (
    "8ab4f8",
    "81c995",
    "fdd663",
    "f28b82",
    "c58af9",
    "78d9ec",
    "fcad70",
    "e89bd4",
    "c6d87f",
    "93c9b2",
    "b4b7fa",
    "e4c49f",
    "95cee3",
    "bcd3a4",
    "efaad1",
    "c7d0d9",
    "d7bff2",
    "85ddb9",
    "f7ba93",
    "dfdba3",
)
MUTED, KEY, TITLE = "a9b4c4", "8ab4f8", "fdd663"
ANSI = re.compile(r"\x1b\[([0-9;]*)m")


def paint(text, color, enabled, bold=False):
    """Apply true-color ANSI after padding, so escape codes do not affect alignment."""
    if not enabled:
        return text
    rgb = ";".join(str(int(color[i : i + 2], 16)) for i in (0, 2, 4))
    return f"\033[{('1;' if bold else '')}38;2;{rgb}m{text}\033[0m"


def display_width(text):
    """Count visible terminal cells, including full-width and combining characters."""
    return sum(
        0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in "WF" else 1
        for c in text
    )


def one_line(text):
    """Keep embedded newlines and control codes from breaking a CSV display row."""
    return "".join(
        {"\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(c, f"\\x{ord(c):02x}")
        if ord(c) < 32 or ord(c) == 127
        else c
        for c in str(text)
    )


def shorten(text, width):
    """Shorten only display text, marking every shortened cell explicitly."""
    if display_width(text) <= width:
        return text
    result = ""
    for c in text:
        if display_width(result + c) > width - 3:
            break
        result += c
    return result + "..."


def pad(text, width, right=False):
    """Align cells before adding color codes."""
    spaces = " " * max(0, width - display_width(text))
    return spaces + text if right else text + spaces


def scalar(value, precision, full):
    """Format values without replacing tiny nonzero values with zero."""
    if np.ma.is_masked(value):
        return "--"
    if isinstance(value, u.Quantity):
        return f"{scalar(value.value, precision, full)} {value.unit}"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return repr(value) if full else f"{value:.{precision}g}"
    return "null" if value is None else str(value)


def structured(value):
    """Decode metadata JSON without requiring every text field to be valid JSON."""
    if np.ma.is_masked(value) and np.ndim(value) == 0:
        return value
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            return json.loads(value)
        except ValueError:
            return value
    if isinstance(value, np.ndarray) and not isinstance(value, u.Quantity):
        return value.tolist()
    return value


class Header:
    """Render readable metadata and schema above the CSV body."""

    def __init__(self, args, color):
        self.args, self.color, self.lines = args, color, []

    def text(self, text="", color=MUTED, bold=False, indent=0):
        """Wrap descriptive text without wrapping or splitting the data table."""
        prefix = " " * min(indent, self.args.width // 2)
        lines = textwrap.wrap(
            str(text),
            width=self.args.width,
            initial_indent=prefix,
            subsequent_indent=prefix,
            break_on_hyphens=False,
        ) or [""]
        self.lines.extend(paint(line, color, self.color, bold) for line in lines)

    def heading(self, text):
        """Emphasize a section heading."""
        self.text()
        self.text(text, TITLE, True)
        self.text("─" * min(len(text), self.args.width), TITLE)

    def tree(self, name, value, depth=0, label_width=0):
        """Print nested metadata as aligned key/value pairs with hanging indentation."""
        value = structured(value)
        indent = min(2 * depth, self.args.width // 2)
        if (
            isinstance(value, list)
            and len(value) <= 6
            and all(not isinstance(v, (list, dict)) for v in value)
        ):
            value = (
                "[" + ", ".join(scalar(v, self.args.precision, self.args.full) for v in value) + "]"
            )
        if isinstance(value, (dict, list)) and value:
            self.text(f"{name}:", KEY, indent=indent)
            items = list(value.items() if isinstance(value, dict) else enumerate(value))
            width = min(40, max(len(str(k)) for k, _ in items))
            for key, item in items:
                self.tree(str(key), item, depth + 1, width)
            return
        prefix = " " * indent + f"{name:<{label_width}} : "
        if len(prefix) > self.args.width - 10:
            self.text(f"{name}:", KEY, indent=indent)
            prefix = " " * min(indent + 2, self.args.width // 2)
        text = scalar(value, self.args.precision, self.args.full)
        lines = textwrap.wrap(
            text,
            width=self.args.width,
            initial_indent=prefix,
            subsequent_indent=" " * min(len(prefix), self.args.width // 2),
            break_on_hyphens=False,
        ) or [prefix.rstrip()]
        for i, line in enumerate(lines):
            if i == 0:
                self.lines.append(
                    paint(line[: len(prefix)], KEY, self.color)
                    + paint(line[len(prefix) :], MUTED, self.color)
                )
            else:
                self.lines.append(paint(line, MUTED, self.color))


def csv_cell(text):
    """Quote CSV delimiters and quotes, after making a cell a single visual line."""
    return '"' + text.replace('"', '""') + '"' if any(c in text for c in ',"') else text


def render(table, path, args, color=False):
    """Return the upper metadata/schema, one CSV table, and navigation hints."""
    header = Header(args, color)
    header.text(f"ECSV · {path.name}", TITLE, True)
    header.text(f"File: {path}")
    header.text(
        f"{len(table):,} rows · {len(table.colnames)} columns · {len(table.meta)} metadata fields"
    )
    if args.metadata:
        header.heading("Metadata")
        if not table.meta:
            header.text("(none)")
        width = min(40, max(map(len, table.meta), default=0))
        for key, value in table.meta.items():
            header.tree(key, value, label_width=width)
    if args.schema:
        header.heading("Columns & units")
        name_width = max(map(len, table.colnames), default=6)
        header.text(f"{'#':>3}  {'Column':<{name_width}}  {'Type':<12}  Unit", bold=True)
        for i, name in enumerate(table.colnames):
            col = table[name]
            unit = str(col.unit) if col.unit is not None else "— (not declared)"
            header.text(
                f"{i:>3}  {name:<{name_width}}  {str(col.dtype):<12}  {unit}",
                PALETTE[i % len(PALETTE)],
            )
    if args.details:
        row = args.row if args.row is not None else 0
        header.heading(f"Row {row} details (zero-based)")
        for name in args.columns or table.colnames:
            header.tree(name, table[name][row])
    indices = (
        [args.row]
        if args.row is not None
        else list(
            range(args.start, len(table) if args.all else min(len(table), args.start + args.rows))
        )
    )
    header.heading(
        "CSV data" + (f" · rows {indices[0]}–{indices[-1]} (zero-based)" if indices else "")
    )
    precision_note = (
        "Original numeric precision."
        if args.full
        else f"Numbers: {args.precision} significant digits (display only)."
    )
    header.text(precision_note + " Timestamp strings and units are unchanged.")
    header.text("Each column keeps its color. Scroll horizontally for wide tables.")
    header.text("Pager: g = metadata/units at top · arrows = scroll · q = quit")
    names = args.columns or table.colnames
    columns, shortened = [], False
    for name in names:
        values = []
        for index in indices:
            raw = one_line(scalar(table[name][index], args.precision, args.full))
            keep_time = name in ("start_time", "stop_time", "timestamp_utc")
            shown = raw if args.full or keep_time else shorten(raw, args.cell_width)
            shortened |= shown != raw
            values.append(csv_cell(shown))
        label = csv_cell(one_line(name))
        width = max(display_width(label), *(display_width(v) for v in values), 1)
        shade = PALETTE[table.colnames.index(name) % len(PALETTE)]
        columns.append((label, values, width, shade, table[name].dtype.kind in "biufc"))
    lines = [", ".join(paint(pad(c[0], c[2]), c[3], color, True) for c in columns)]
    for i in range(len(indices)):
        lines.append(", ".join(paint(pad(c[1][i], c[2], c[4]), c[3], color) for c in columns))
    footer = Header(args, color)
    footer.text()
    footer.text(f"Showing {len(indices):,} of {len(table):,} rows.")
    if shortened:
        footer.text(
            "'...' marks shortened cells. Use --row N --details or --full for complete values."
        )
    if indices and indices[-1] + 1 < len(table) and args.row is None:
        footer.text(f"Next page: --start {indices[-1] + 1} --rows {args.rows} | All rows: --all")
    footer.text(
        "Terminal pager: left/right = horizontal scroll · up/down = vertical scroll · q = quit"
    )
    return "\n".join(header.lines) + "\n", "\n".join(lines) + "\n", "\n".join(footer.lines) + "\n"


def ansi_html(text):
    """Translate this viewer's ANSI colors to safe HTML spans."""
    result, end, opened = [], 0, False
    for match in ANSI.finditer(text):
        result.append(html.escape(text[end : match.start()]))
        if opened:
            result.append("</span>")
            opened = False
        codes = match.group(1).split(";")
        if "38" in codes:
            start = codes.index("38")
            rgb = ",".join(codes[start + 2 : start + 5])
            bold = "font-weight:600;" if codes[0] == "1" else ""
            result.append(f'<span style="{bold}color:rgb({rgb})">')
            opened = True
        end = match.end()
    result.append(html.escape(text[end:]))
    if opened:
        result.append("</span>")
    return "".join(result)


def write_html(path, parts, title):
    """Save the identical aligned table with horizontal scrolling in a browser."""
    path.parent.mkdir(parents=True, exist_ok=True)
    top, body, footer = (ansi_html(part) for part in parts)
    path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)} · ECSV reader</title><style>"
        "*{box-sizing:border-box}body{margin:0;background:#10151f;color:#ccd4e0;"
        "font:14px/1.6 system-ui}main{padding:28px 32px;max-width:100%}"
        "pre{margin:0;font:13px/1.7 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;"
        "font-variant-ligatures:none;tab-size:4}header{border-bottom:1px solid #30394a;"
        "padding-bottom:16px;margin-bottom:18px}header h1{font-size:21px;margin:0 0 6px}"
        "header p{margin:0;color:#a9b4c4}.info{margin-bottom:18px;border:1px solid #344156;"
        "border-radius:6px;background:#151c28}.info summary{padding:9px 14px;cursor:pointer;"
        "color:#8ab4f8;font-weight:600}.metadata{overflow:auto;max-height:28vh;padding:0 14px 12px;"
        "scrollbar-color:#64748b #151c28}h2{margin:14px 0 8px;font-size:16px;color:#fdd663}"
        ".data{border:1px solid #344156;border-radius:6px;background:#151c28;"
        "overflow:auto;max-height:52vh;padding:16px;scrollbar-color:#64748b #151c28}"
        ".data pre{width:max-content;min-width:100%}.hint{margin-top:16px;white-space:pre-wrap}"
        "</style><main><header>"
        f"<h1>{html.escape(title)}</h1>"
        "<p>One color per column · Scroll the table horizontally</p>"
        '</header><details class="info" open>'
        "<summary>Metadata &amp; units · scroll inside, or click to collapse</summary>"
        f'<pre class="metadata" tabindex="0">{top}</pre></details>'
        "<h2>CSV data</h2>"
        f'<div class="data" tabindex="0" aria-label="Scrollable ECSV data"><pre>{body}</pre></div>'
        f'<pre class="hint">{footer}</pre></main></html>'
    )


def wrap_cells(text, width):
    """Wrap a detail view by terminal cell width, including Chinese characters."""
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        current, used = "", 0
        for char in one_line(paragraph):
            size = display_width(char)
            if used + size > width and current:
                lines.append(current)
                current, used = "", 0
            current += char
            used += size
        lines.append(current)
    return lines


def detail_text(name, value):
    """Expand JSON for inspection, while preserving full numeric precision."""
    value = structured(value)
    if isinstance(value, (dict, list)):
        content = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    else:
        content = scalar(value, 17, True)
    return f"{name}\n\n{content}"


class GridViewer:
    """Keep one continuous table, fixed headers and a selected-cell inspector."""

    def __init__(self, screen, curses, table, path, args, color):
        self.screen, self.curses = screen, curses
        self.table, self.path, self.args = table, path, args
        self.names = args.columns or table.colnames
        end = len(table) if args.all or args.grid_rows is None else args.start + args.grid_rows
        self.rows = [args.row] if args.row is not None else range(args.start, min(len(table), end))
        self.row = self.col = self.top = self.left = 0
        self.query = ""
        self.notice = "Enter full value · m metadata · u columns · / search · g go to row · ? help"
        self.colors = []
        screen.keypad(True)
        curses.curs_set(0)
        if color and curses.has_colors():
            curses.start_color()
            try:
                curses.use_default_colors()
                background = -1
            except curses.error:
                background = curses.COLOR_BLACK
            for i, rgb in enumerate(PALETTE[: curses.COLOR_PAIRS - 1]):
                if curses.COLORS >= 256:
                    r, g, b = (round(int(rgb[j : j + 2], 16) / 255 * 5) for j in (0, 2, 4))
                    foreground = 16 + 36 * r + 6 * g + b
                else:
                    foreground = (4, 2, 3, 1, 5, 6, 7)[i % 7] % curses.COLORS
                curses.init_pair(i + 1, foreground, background)
                self.colors.append(curses.color_pair(i + 1))
        self.widths = []
        for name in self.names:
            values = (self.cell(r, name) for r in list(self.rows[:200]))
            longest = max([display_width(name), *(display_width(v) for v in values)], default=8)
            self.widths.append(max(10, min(args.cell_width, 28, longest)))
        try:
            curses.mousemask(curses.ALL_MOUSE_EVENTS)
        except curses.error:
            pass

    def cell(self, row, name):
        """Format a grid cell; units live in the fixed header instead of every row."""
        value = self.table[name][row]
        if isinstance(value, u.Quantity):
            value = value.value
        return one_line(scalar(value, self.args.precision, self.args.full))

    def shade(self, col):
        """Keep the original column color even after filtering or scrolling."""
        if not self.colors:
            return 0
        original = self.table.colnames.index(self.names[col])
        return self.colors[original % len(self.colors)]

    def put(self, y, x, text, style=0, width=None):
        """Draw only inside the viewport and never interpret data as terminal controls."""
        height, columns = self.screen.getmaxyx()
        room = columns - x - 1 if width is None else min(width, columns - x - 1)
        if 0 <= y < height and room >= 3:
            try:
                self.screen.addstr(y, x, shorten(one_line(text), room), style)
            except self.curses.error:
                pass  # A resize may occur between getmaxyx and addstr.

    def layout(self, width, gutter):
        """Fit whole columns; the selected column is always visible."""
        if self.col < self.left:
            self.left = self.col
        while True:
            visible, x = [], gutter
            for i in range(self.left, len(self.names)):
                size = min(self.widths[i], max(8, width - gutter - 3))
                if x + size + 2 >= width:
                    break
                visible.append((i, x, size))
                x += size + 3
            if not visible or self.col <= visible[-1][0]:
                return visible
            self.left += 1

    def draw(self):
        """Render a bounded screen; headers and the selected value never scroll away."""
        height, width = self.screen.getmaxyx()
        self.screen.erase()
        if height < 12 or width < 40:
            self.put(0, 0, "Window too small; enlarge it or press q to quit")
            self.screen.refresh()
            return [], 1
        page = height - 9
        self.top = max(0, min(self.top, max(0, len(self.rows) - page)))
        if self.row < self.top:
            self.top = self.row
        if self.row >= self.top + page:
            self.top = self.row - page + 1
        gutter = max(6, len(str(max(len(self.table) - 1, 0))) + 2)
        visible = self.layout(width, gutter)
        self.put(
            0,
            0,
            f"ECSV  {self.path.name}    {len(self.table):,} rows × {len(self.names)} cols",
            self.curses.A_BOLD,
        )
        start = self.table.meta.get("start_utc", self.table.meta.get("timeline_start_time", ""))
        summary = f"Metadata: {len(self.table.meta)} fields [m]"
        if start:
            summary += f"  ·  start: {start}"
        self.put(1, 0, summary)
        self.put(2, 0, "row", self.curses.A_BOLD, gutter - 1)
        for col, x, size in visible:
            name = self.names[col]
            style = self.shade(col) | self.curses.A_BOLD
            if col == self.col:
                style |= self.curses.A_REVERSE
            self.put(2, x, pad(shorten(name, size), size), style, size)
            unit = str(self.table[name].unit) if self.table[name].unit is not None else "—"
            self.put(3, x, f"[{unit}]", self.shade(col), size)
        self.put(4, 0, "─" * (width - 1))
        for offset, position in enumerate(range(self.top, min(len(self.rows), self.top + page))):
            y, row = 5 + offset, self.rows[position]
            self.put(y, 0, str(row), self.curses.A_BOLD if position == self.row else 0, gutter - 1)
            for col, x, size in visible:
                numeric = self.table[self.names[col]].dtype.kind in "biufc"
                text = pad(shorten(self.cell(row, self.names[col]), size), size, numeric)
                style = self.shade(col)
                if position == self.row and col == self.col:
                    style |= self.curses.A_REVERSE
                self.put(y, x, text, style, size)
                self.put(y, x + size, " │ ", width=3)
        if not self.rows:
            self.put(5, gutter, "(empty table)")
        if self.rows and self.names:
            name, row = self.names[self.col], self.rows[self.row]
            unit = self.table[name].unit or "not declared"
            info = f"row {row} · col {self.col + 1}/{len(self.names)} · {name} [{unit}]"
            self.put(height - 4, 0, info, self.curses.A_BOLD)
            self.put(height - 3, 0, scalar(self.table[name][row], 17, True), self.shade(self.col))
        self.put(height - 2, 0, self.notice)
        self.put(
            height - 1,
            0,
            "↑↓ rows  ←→ columns  Space/b pages  [ ] horizontal pages  q quit",
            self.curses.A_REVERSE,
        )
        self.screen.refresh()
        return visible, page

    def popup(self, title, text):
        """Inspect complete metadata or JSON in a separate, paged screen."""
        offset = 0
        while True:
            height, width = self.screen.getmaxyx()
            lines = wrap_cells(text, max(8, width - 3))
            page = max(1, height - 3)
            offset = max(0, min(offset, max(0, len(lines) - page)))
            self.screen.erase()
            self.put(0, 0, title, self.curses.A_BOLD)
            for i, line in enumerate(lines[offset : offset + page]):
                self.put(i + 1, 0, line)
            self.put(
                height - 1,
                0,
                f"{offset + 1}/{len(lines)}  ↑↓ / Space/b pages · Home/End · q/Esc back",
                self.curses.A_REVERSE,
            )
            self.screen.refresh()
            key = self.screen.get_wch()
            if key in ("q", "\x1b", "\n", "\r"):
                return
            if key == "\x0c":
                self.screen.clearok(True)
            if key in (self.curses.KEY_NPAGE, " "):
                offset += page
            elif key in (self.curses.KEY_PPAGE, "b"):
                offset -= page
            elif key in (self.curses.KEY_DOWN, "j"):
                offset += 1
            elif key in (self.curses.KEY_UP, "k"):
                offset -= 1
            elif key in (self.curses.KEY_HOME, "g"):
                offset = 0
            elif key in (self.curses.KEY_END, "G"):
                offset = len(lines) - page

    def prompt(self, label):
        """Read a short query without echoing it into shell history or scrollback."""
        value = ""
        while True:
            height, width = self.screen.getmaxyx()
            self.screen.move(height - 1, 0)
            self.screen.clrtoeol()
            self.put(height - 1, 0, label + value + "▏", self.curses.A_REVERSE, width - 1)
            self.screen.refresh()
            key = self.screen.get_wch()
            if key in ("\n", "\r", self.curses.KEY_ENTER):
                return value
            if key == "\x1b":
                return None
            if key in (self.curses.KEY_BACKSPACE, "\x7f", "\b"):
                value = value[:-1]
            elif isinstance(key, str) and key.isprintable():
                value += key

    def find_next(self):
        """Search full values in the selected column, wrapping through loaded rows."""
        if not self.query or not self.rows or not self.names:
            return
        name = self.names[self.col]
        for step in range(1, len(self.rows) + 1):
            position = (self.row + step) % len(self.rows)
            value = scalar(self.table[name][self.rows[position]], 17, True)
            if self.query.casefold() in value.casefold():
                self.row = position
                self.notice = f"Found {self.query!r} · column {name} · n next"
                return
        self.notice = f"Column {name} has no match for {self.query!r}"

    def handle_key(self, key):
        """Allow other read-only table viewers to add navigation without duplicating this grid."""
        return False

    def run(self):
        """Navigate by cells and pages while preserving the underlying table."""
        if self.args.start_at == "top":
            self.popup("Metadata", detail_text("File metadata", self.table.meta))
        if self.args.details and self.rows:
            self.popup(
                "Row details",
                "\n\n".join(
                    detail_text(name, self.table[name][self.rows[0]]) for name in self.names
                ),
            )
        while True:
            visible, page = self.draw()
            key = self.screen.get_wch()
            if self.handle_key(key):
                continue
            if key in ("q", "\x1b"):
                return
            if key == "\x0c":
                self.screen.clearok(True)
            elif key in (self.curses.KEY_DOWN, "j"):
                self.row += 1
            elif key in (self.curses.KEY_UP, "k"):
                self.row -= 1
            elif key in (self.curses.KEY_RIGHT, "l", "\t"):
                self.col += 1
            elif key in (self.curses.KEY_LEFT, "h", self.curses.KEY_BTAB):
                self.col -= 1
            elif key in (self.curses.KEY_NPAGE, " "):
                self.row += page
                self.top += page
            elif key in (self.curses.KEY_PPAGE, "b"):
                self.row -= page
                self.top -= page
            elif key == "]" and visible:
                self.col = min(len(self.names) - 1, visible[-1][0] + 1)
                self.left = self.col
            elif key == "[":
                self.col = self.left = max(0, self.left - max(1, len(visible)))
            elif key in (self.curses.KEY_HOME, "H"):
                self.row = self.top = 0
            elif key in (self.curses.KEY_END, "G"):
                self.row = len(self.rows) - 1
            elif key == "g" and self.rows:
                value = self.prompt("Go to row (0-based, Esc cancels): ")
                if value is not None:
                    try:
                        self.row = self.rows.index(int(value))
                    except ValueError:
                        self.notice = "Row is outside the displayed range"
            elif key == "/":
                query = self.prompt("Search current column (Esc cancels): ")
                if query is not None:
                    self.query = query
                    self.find_next()
            elif key == "n":
                self.find_next()
            elif key in ("\n", "\r", self.curses.KEY_ENTER) and self.rows and self.names:
                name = self.names[self.col]
                self.popup(
                    f"row {self.rows[self.row]} · {name} · full value",
                    detail_text(name, self.table[name][self.rows[self.row]]),
                )
            elif key == "m":
                self.popup("Metadata", detail_text("File metadata", self.table.meta))
            elif key == "u":
                text = "\n".join(
                    f"{i}: {name}   [{self.table[name].unit or 'not declared'}]"
                    f"   {self.table[name].dtype}"
                    for i, name in enumerate(self.names)
                )
                self.popup("Columns & units", text)
            elif key == "?":
                self.popup(
                    "Keyboard help",
                    "Arrow keys: select row / column\nSpace or PgDn: next page\n"
                    "b or PgUp: previous page (Mac: fn+↑/↓)\n[ / ]: horizontal pages\n"
                    "Home / H: first row; End / G: last row\ng: enter row number (0-based)\n"
                    "/: search full values in current column; n: next match\nEnter: full cell value\n"
                    "m: metadata; u: column names / types / units\nq/Esc in popups: return to table\n"
                    "q/Esc or Ctrl+C in table: quit and restore terminal\n"
                    "... marks a truncated display; values are unchanged. See the full value below or press Enter.",
                )
            elif key == self.curses.KEY_MOUSE:
                try:
                    _, x, y, _, state = self.curses.getmouse()
                    if state & self.curses.BUTTON4_PRESSED:
                        self.row -= max(3, page // 2)
                    elif state & getattr(self.curses, "BUTTON5_PRESSED", 0):
                        self.row += max(3, page // 2)
                    elif state & (self.curses.BUTTON1_CLICKED | self.curses.BUTTON1_PRESSED):
                        if 5 <= y < 5 + page:
                            self.row = self.top + y - 5
                            for col, left, size in visible:
                                if left <= x < left + size:
                                    self.col = col
                except self.curses.error:
                    pass
            self.row = max(0, min(self.row, len(self.rows) - 1))
            self.col = max(0, min(self.col, len(self.names) - 1))


def run_grid(table, path, args, color):
    """Use the alternate screen and restore terminal state even on interruption."""
    import curses
    import termios

    descriptor = sys.stdin.fileno()
    terminal_settings = termios.tcgetattr(descriptor)
    curses.set_escdelay(50)  # Esc should close a view promptly, not wait a full second.

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        curses.wrapper(lambda screen: GridViewer(screen, curses, table, path, args, color).run())
    finally:
        termios.tcsetattr(descriptor, termios.TCSANOW, terminal_settings)
        signal.signal(signal.SIGTERM, previous)


def pager_start_line(header):
    """Locate the last data heading after any metadata, including ANSI-colored text."""
    lines = ANSI.sub("", header).splitlines()
    return next((i + 1 for i in reversed(range(len(lines))) if lines[i].startswith("CSV data")), 1)


def main(argv=None):
    """Read one ECSV and display a rainbow CSV table or an optional HTML preview."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", type=Path, help="ECSV file (never modified)")
    parser.add_argument("--rows", type=int, help="limit rows (grid default: all; text default: 50)")
    parser.add_argument(
        "--view",
        choices=("grid", "text"),
        default="grid",
        help="interactive grid (default) or legacy text/less",
    )
    parser.add_argument("--start", type=int, default=0, help="first row, zero-based")
    parser.add_argument("--all", action="store_true", help="show all rows from --start")
    parser.add_argument("--row", type=int, help="show one row in the same CSV table")
    parser.add_argument("--details", action="store_true", help="expand --row fields above the CSV")
    parser.add_argument("--columns", help="comma-separated original column names")
    parser.add_argument(
        "--metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="text view: metadata above the CSV (default: on); grid uses m",
    )
    parser.add_argument(
        "--schema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="text view: schema above CSV (default: on); grid always shows unit headers",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="ANSI colors (auto: terminal only; respects NO_COLOR)",
    )
    parser.add_argument(
        "--pager",
        choices=("auto", "always", "never"),
        default="auto",
        help="interactive display (auto: terminal); never prints plain text; --view text uses less",
    )
    parser.add_argument(
        "--start-at",
        choices=("data", "top"),
        default="data",
        help="where the terminal pager opens (default: data; g returns to metadata)",
    )
    parser.add_argument("--html", type=Path, help="also save a self-contained color preview")
    parser.add_argument(
        "--full",
        action="store_true",
        help="original numeric precision; grid still fits cells to screen (Enter for full value)",
    )
    parser.add_argument(
        "--precision", type=int, default=6, help="display significant digits (1-17)"
    )
    parser.add_argument(
        "--cell-width",
        type=int,
        default=40,
        help="max text cell width (default: 40); grid uses compact widths up to 28",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=max(60, min(shutil.get_terminal_size((110, 24)).columns, 140)),
        help="metadata text width; the CSV table stays on one line",
    )
    args = parser.parse_args(argv)
    args.grid_rows = args.rows
    args.rows = 50 if args.rows is None else args.rows
    if (
        args.rows < 1
        or args.start < 0
        or args.width < 60
        or args.cell_width < 8
        or not 1 <= args.precision <= 17
    ):
        parser.error(
            "Require rows >= 1, start >= 0, width >= 60, cell-width >= 8 and precision 1-17"
        )
    if args.html and args.html.resolve() == args.path.resolve():
        parser.error("HTML output must be different from the input ECSV")
    try:
        table = Table.read(args.path, format="ascii.ecsv")
    except Exception as error:
        parser.error(f"Cannot read {args.path}: {error}")
    if args.row is not None and not 0 <= args.row < len(table):
        parser.error(f"--row must be between 0 and {len(table) - 1}")
    if args.row is None and len(table) and args.start >= len(table):
        parser.error(f"--start must be less than {len(table)}")
    if args.details and (not len(table) or (args.row is None and len(table) != 1)):
        parser.error("Select --row N to expand details of a multirow file")
    if args.columns is not None:
        args.columns = list(dict.fromkeys(n.strip() for n in args.columns.split(",") if n.strip()))
        if not args.columns or set(args.columns) - set(table.colnames):
            parser.error(f"Unknown or empty selection. Columns: {', '.join(table.colnames)}")
    use_color = args.color == "always" or (
        args.color == "auto" and sys.stdout.isatty() and "NO_COLOR" not in os.environ
    )
    interactive = sys.stdout.isatty() and sys.stdin.isatty()
    if interactive and args.view == "grid" and args.pager != "never":
        if args.html:
            write_html(args.html, render(table, args.path, args, True), args.path.name)
        return run_grid(table, args.path, args, use_color) or 0
    parts = render(table, args.path, args, use_color)
    if args.html:
        write_html(args.html, render(table, args.path, args, True), args.path.name)
    text = "\n".join(parts)
    less = shutil.which("less")
    use_pager = args.pager == "always" or (
        args.pager == "auto" and sys.stdout.isatty() and sys.stdin.isatty()
    )
    if use_pager and less:
        line = pager_start_line(parts[0]) if args.start_at == "data" else 1
        subprocess.run(
            [less, "-R", "-S", f"+{line}"],
            input=text,
            text=True,
            check=False,
            env={**os.environ, "LESS": ""},  # User LESS=-X must not leave the table in scrollback.
        )
    else:
        print(text, end="")
    if args.html:
        print(f"HTML preview: {args.html}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(130)
