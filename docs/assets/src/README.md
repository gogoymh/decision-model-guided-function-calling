# README artwork sources

The hero and framework are native HTML/SVG. The six theme-specific charts use
`data.json`, whose measurements preserve the original BFCL GPT runs. The current
paper's six-row benchmark summary also includes Claude and exploratory API-Bank;
those additional results are shown separately in the root README.

Render at twice the logical dimensions with Playwright and Chromium:

```bash
uv run --with playwright python docs/assets/src/render.py docs/assets/src/hero.html docs/assets/dmfc-hero.png 1280 460
uv run --with playwright python docs/assets/src/render.py docs/assets/src/framework.html docs/assets/dmfc-framework.png 1440 820
uv run --with playwright python docs/assets/src/charts.py docs/assets/src/data.json docs/assets
```

If a compatible Chromium is already installed, set
`PLAYWRIGHT_CHROMIUM_EXECUTABLE` to its executable path. Otherwise install
Playwright's Chromium with `uv run --with playwright playwright install chromium`.
The font stack falls back to the system sans-serif font when Inter is unavailable.

The framework's probabilities are illustrative, as its caption states. Rejected
candidates are never executed during repair; the library's final dispatch executes
an accepted call. The hero contains no benchmark performance claims.
