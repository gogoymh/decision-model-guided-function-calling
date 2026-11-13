# README images

- `dm-call-hero.png` is `hero.html` rendered at twice its logical size (2560 x 920). The page sets the
  title in [Pretendard](https://github.com/orioncactus/pretendard) (SIL Open Font License, loaded from
  jsDelivr) over `hero-background.jpg`, an AI-generated image supplied by the authors. Render it with:

  ```bash
  PLAYWRIGHT_CHROMIUM_EXECUTABLE="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    uv run --with playwright python docs/assets/src/render.py docs/assets/src/hero.html docs/assets/dm-call-hero.png 1280 460
  ```

  `render.py` waits for the fonts and stops if Pretendard did not load.
- `star.gif` is `star.html` rendered at twice its logical size (1280 x 320, 4 s, 25 frames per second).
  The page draws each frame with `render(t)`; `render_gif.py` captures the frames and builds the GIF with ffmpeg:

  ```bash
  PLAYWRIGHT_CHROMIUM_EXECUTABLE="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    uv run --with playwright python docs/assets/src/render_gif.py docs/assets/src/star.html docs/assets/star.gif 640 160 4.0 25
  ```

- `dm-call-figure1.png` is Figure 1 of the paper, scaled to 1,120 px from the paper's figure file.
- `dm-call-figure2.png` is Figure 2 of the paper, rendered at 1,600 px from the paper's figure file
  (`pdftoppm -png -singlefile -scale-to-x 1600 -scale-to-y -1 repair-limit.pdf dm-call-figure2`).

The hero carries no result numbers; results appear only in the README's Results section.
