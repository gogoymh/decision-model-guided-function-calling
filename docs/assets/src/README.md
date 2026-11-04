# README artwork sources

The DM-Call hero is native HTML/SVG (`hero.html`). The six theme-specific charts use
`data.json`, which holds the current paper's numbers: BFCL category accuracy of the Table 1
GPT runs (Direct and DM-Call with Jev, mean of three runs), Table 3 (archived `gpt-5-nano`
runs replayed; time and cost are retained references, and Self-refine's cost is unknown)
and Figure 2 (accuracy and cost by repair limit).

The figure under "How it works" is Figure 1 of the paper. `dm-call-figure1.pdf` is the
manuscript's figure file; its third panel title reads "DM-Call (Ours)", built from the
figure's own glyph outlines, and its hidden search text says the same. Export the PNG with:

```bash
pdftoppm -png -singlefile -scale-to-x 1760 -scale-to-y -1 docs/assets/src/dm-call-figure1.pdf docs/assets/dm-call-figure1
```

To export chart HTML for inspection through a browser without launching Chromium:

```bash
python3 docs/assets/src/charts.py docs/assets/src/data.json work/artwork --html-only
```

The standalone renderer uses Playwright and Chromium, at twice the logical dimensions:

```bash
uv run --with playwright python docs/assets/src/render.py docs/assets/src/hero.html docs/assets/dm-call-hero.png 1280 460
uv run --with playwright python docs/assets/src/charts.py docs/assets/src/data.json docs/assets
```

The committed PNGs (2026-11-04) were rendered at 1x, the README's display size, with the
same viewports and Inter loaded from Google Fonts, because Inter is not installed on the
rendering machine. Without Inter the font stack falls back to the system sans-serif font.

If a compatible Chromium is already installed, set
`PLAYWRIGHT_CHROMIUM_EXECUTABLE` to its executable path. Otherwise install
Playwright's Chromium with `uv run --with playwright playwright install chromium`.

Rejected candidates are never executed during repair; the library's final dispatch
executes an accepted call. The hero contains no benchmark performance claims.
