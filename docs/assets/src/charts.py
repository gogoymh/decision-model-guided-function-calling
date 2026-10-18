"""README charts: SVG in an HTML card, rendered to PNG for light and dark themes.

Usage: python docs/assets/src/charts.py docs/assets/src/data.json docs/assets
Needs Playwright with Chromium and the Inter font. The numbers in data.json must equal
the filled paper results (Tables 1-2 and Figure 2); colors are a validated categorical palette.
"""
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  base="#c3c2b7", blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", gray="#c9c7c0", link="#c3c2b7",
                  border="rgba(11,11,11,0.10)"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 base="#383835", blue="#3987e5", orange="#d95926", aqua="#199e70", gray="#4b4a46", link="#4b4a46",
                 border="rgba(255,255,255,0.10)"),
}
W = 880


def page(t, title, subtitle, legend, svg):
    items = "".join(f'<span><i class="sw" style="background:{color}"></i>{label}</span>' for label, color in legend)
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;background:transparent}}
.card{{width:{W}px;box-sizing:border-box;padding:24px 28px 18px;border-radius:16px;background:{t['surface']};
border:1px solid {t['border']};font-family:Inter,system-ui,sans-serif}}
.t{{font:600 19px Inter;color:{t['ink']};letter-spacing:-.01em}}
.s{{font:400 13.5px Inter;color:{t['ink2']};margin-top:5px}}
.lg{{display:flex;gap:20px;margin-top:12px;font:500 13px Inter;color:{t['ink2']}}}
.sw{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px;vertical-align:-1px}}
svg{{display:block;margin-top:10px}} svg text{{font-family:Inter}}
</style></head><body><div class="card" id="card"><div class="t">{title}</div><div class="s">{subtitle}</div>
{'<div class="lg">' + items + '</div>' if legend else ''}{svg}</div></body></html>"""


def text(x, y, s, color, size=12.5, weight=400, anchor="start"):
    return (f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" dominant-baseline="middle">{s}</text>')


def categories(t, rows):
    """Dots per category: Direct and JBFC with gpt-5-nano (joined), and Direct with gpt-5.4-mini."""
    width, left, x0, x1 = W - 56, 0, 168, 540
    cols = {"nano": (620, t["orange"]), "JBFC": (700, t["blue"]), "mini": (780, t["aqua"])}
    row_h, top = 36, 26
    height = top + row_h * len(rows) + 30

    def sx(v):
        return x0 + (x1 - x0) * v / 100

    out = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    for v in (0, 25, 50, 75, 100):
        out.append(f'<line x1="{sx(v)}" y1="{top - 8}" x2="{sx(v)}" y2="{top + row_h * len(rows)}" stroke="{t["grid"]}" stroke-width="1"/>')
        out.append(text(sx(v), top + row_h * len(rows) + 16, f"{v}%", t["muted"], 11, anchor="middle"))
    for name, (x, color) in cols.items():
        out.append(f'<circle cx="{x - len(name) * 7 - 10}" cy="{top - 14}" r="4" fill="{color}"/>')
        out.append(text(x, top - 14, name, t["muted"], 11, 500, "end"))
    for i, (label, direct, jbfc, mini) in enumerate(rows):
        y = top + row_h * i + row_h / 2
        total = label == "Total"
        if total:
            out.append(f'<line x1="0" y1="{y - row_h / 2}" x2="{width}" y2="{y - row_h / 2}" stroke="{t["base"]}" stroke-width="1"/>')
        out.append(text(left, y, label, t["ink"] if total else t["ink2"], 13.5, 600 if total else 450))
        out.append(f'<line x1="{sx(direct)}" y1="{y}" x2="{sx(jbfc)}" y2="{y}" stroke="{t["link"]}" stroke-width="2" stroke-linecap="round"/>')
        for value, color in ((direct, t["orange"]), (jbfc, t["blue"]), (mini, t["aqua"])):
            out.append(f'<circle cx="{sx(value)}" cy="{y}" r="6.5" fill="{color}" stroke="{t["surface"]}" stroke-width="2"/>')
        best = max(direct, jbfc, mini)
        for value, (x, _) in zip((direct, jbfc, mini), cols.values()):
            top_value = value == best
            out.append(text(x, y, f"{value:.1f}", t["ink"] if top_value else t["ink2"], 13, 650 if top_value else 450, "end"))
    out.append("</svg>")
    return "".join(out)


def methods(t, rows):
    width, x0, x1 = W - 56, 250, 600
    cols = {"API s": 700, "USD / 1k": 790}
    row_h, top, bar = 36, 26, 18
    height = top + row_h * len(rows) + 30

    def sx(v):
        return x0 + (x1 - x0) * v / 100

    out = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    for v in (0, 25, 50, 75, 100):
        out.append(f'<line x1="{sx(v)}" y1="{top - 8}" x2="{sx(v)}" y2="{top + row_h * len(rows)}" stroke="{t["grid"]}" stroke-width="1"/>')
        out.append(text(sx(v), top + row_h * len(rows) + 16, f"{v}%", t["muted"], 11, anchor="middle"))
    for name, x in cols.items():
        out.append(text(x, top - 14, name, t["muted"], 11, 500, "end"))
    for i, (label, note, accuracy, seconds, usd) in enumerate(rows):
        y = top + row_h * i + row_h / 2
        star = label == "JBFC"
        out.append(text(0, y, label, t["ink"] if star else t["ink2"], 13.5, 650 if star else 450))
        if note:
            out.append(text(66, y, note, t["muted"], 11.5))
        end, r = sx(accuracy), 4
        color = t["blue"] if star else t["gray"]
        out.append(f'<path d="M{x0},{y - bar / 2} H{end - r} Q{end},{y - bar / 2} {end},{y - bar / 2 + r} '
                   f'V{y + bar / 2 - r} Q{end},{y + bar / 2} {end - r},{y + bar / 2} H{x0} Z" fill="{color}"/>')
        out.append(text(end + 8, y, f"{accuracy:.1f}%", t["ink"] if star else t["ink2"], 13, 650 if star else 450))
        out.append(text(cols["API s"], y, f"{seconds:.2f}", t["ink2"], 12.5, 450, "end"))
        out.append(text(cols["USD / 1k"], y, f"${usd:.3f}", t["ink2"], 12.5, 450, "end"))
    out.append("</svg>")
    return "".join(out)


def limits(t, rows):
    width, gap = W - 56, 48
    panel = (width - gap) / 2
    top, plot_h = 34, 190
    height = top + plot_h + 40
    out = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    specs = [("Call accuracy (%)", 1, (70, 80, (70, 75, 80)), t["blue"], lambda v: f"{v:.1f}"),
             ("Cost, USD per 1,000 requests", 2, (0, 0.8, (0, 0.2, 0.4, 0.6, 0.8)), t["orange"], lambda v: f"${v:.3f}")]
    for k, (label, index, (lo, hi, ticks), color, fmt) in enumerate(specs):
        ox = k * (panel + gap)
        px0, px1 = ox + 40, ox + panel - 18

        def sx(r, px0=px0, px1=px1):
            return px0 + (px1 - px0) * r / 10

        def sy(v, lo=lo, hi=hi):
            return top + plot_h - plot_h * (v - lo) / (hi - lo)

        out.append(text(ox, 10, label, t["ink2"], 12.5, 600))
        for v in ticks:
            out.append(f'<line x1="{px0}" y1="{sy(v)}" x2="{px1}" y2="{sy(v)}" stroke="{t["grid"]}" stroke-width="1"/>')
            out.append(text(px0 - 8, sy(v), f"{v:g}" if index == 1 else f"{v:.1f}", t["muted"], 11, anchor="end"))
        for r, *_ in rows:
            out.append(text(sx(r), top + plot_h + 16, f"r={r}", t["muted"], 11, anchor="middle"))
        points = [(sx(row[0]), sy(row[index])) for row in rows]
        out.append(f'<polyline points="{" ".join(f"{x},{y}" for x, y in points)}" fill="none" stroke="{color}" '
                   f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for (x, y), row in zip(points, rows):
            out.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}" stroke="{t["surface"]}" stroke-width="2"/>')
        for j in (0, 1, len(rows) - 1):  # first, default (r=2) and last; the line rises, so label below early points
            x, y = points[j]
            value = fmt(rows[j][index]) + (" (default)" if rows[j][0] == 2 else "")
            if j == len(rows) - 1:
                out.append(text(x - 2, y - 16, value, t["ink"], 12.5, 600, "end"))
            else:
                out.append(text(x + (4 if j == 0 else 8), y + 19, value, t["ink"], 12.5, 600, "start"))
    out.append("</svg>")
    return "".join(out)


def main(data_path, out_dir):
    data = json.loads(Path(data_path).read_text())
    out_dir = Path(out_dir).resolve()
    jobs = []
    for mode, t in THEMES.items():
        jobs.append((f"chart-categories-{mode}.png", page(t, "Accuracy by BFCL category",
                     "All 3,051 examples · one run per generator · JBFC makes at most 2 repairs · bold: highest in the row",
                     [("nano: Direct, gpt-5-nano", t["orange"]), ("JBFC: gpt-5-nano + Jev", t["blue"]),
                      ("mini: Direct, gpt-5.4-mini", t["aqua"])], categories(t, data["categories"]))))
        jobs.append((f"chart-methods-{mode}.png", page(t, "Call accuracy by method",
                     "gpt-5-nano · same first proposal for every method · at most 2 repairs · time and cost per request at right",
                     [], methods(t, data["methods"]))))
        jobs.append((f"chart-limits-{mode}.png", page(t, "JBFC by repair limit r",
                     "Call accuracy and cost · one run extended from r=2, so every limit shares its first attempts",
                     [], limits(t, data["limits"]))))
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, html in jobs:
            path = out_dir / (name + ".html")
            path.write_text(html)
            pg = browser.new_page(viewport={"width": W + 20, "height": 900}, device_scale_factor=2)
            pg.goto(f"file://{path}")
            pg.wait_for_timeout(150)
            pg.locator("#card").screenshot(path=str(out_dir / name), omit_background=True)
            pg.close()
            print(name)
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
