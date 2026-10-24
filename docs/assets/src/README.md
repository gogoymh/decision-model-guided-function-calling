# README artwork sources

The DMGFC hero and framework are native HTML/SVG. The six theme-specific charts use
`data.json`, whose measurements preserve the original BFCL GPT runs. The current
paper's six-row benchmark summary also includes Claude and exploratory API-Bank;
those additional results are shown separately in the root README.

To export chart HTML for inspection through a browser without launching Chromium:

```bash
python3 docs/assets/src/charts.py docs/assets/src/data.json work/artwork --html-only
```

The renamed assets were captured and verified through the Codex in-app browser.
The existing standalone renderer remains available to contributors using Playwright
and Chromium, at twice the logical dimensions:

```bash
uv run --with playwright python docs/assets/src/render.py docs/assets/src/hero.html docs/assets/dmgfc-hero.png 1280 460
uv run --with playwright python docs/assets/src/render.py docs/assets/src/framework.html docs/assets/dmgfc-framework.png 1440 820
uv run --with playwright python docs/assets/src/charts.py docs/assets/src/data.json docs/assets
```

If a compatible Chromium is already installed, set
`PLAYWRIGHT_CHROMIUM_EXECUTABLE` to its executable path. Otherwise install
Playwright's Chromium with `uv run --with playwright playwright install chromium`.
The font stack falls back to the system sans-serif font when Inter is unavailable.

The framework's probabilities are illustrative, as its caption states. Rejected
candidates are never executed during repair; the library's final dispatch executes
an accepted call. The hero contains no benchmark performance claims.
