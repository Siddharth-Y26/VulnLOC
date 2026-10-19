---
title: VulnLoc
emoji: 🔎
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# VulnLoc — line-level vulnerability localization (demo)

Paste a C or C++ function. **Findings** ranks its lines with the released model; the
**Detailed report** runs every stage of the pipeline on the paste.

- **It ranks lines; it cannot say a function is safe.** The model was trained only on
  functions known to be vulnerable.
- **Its verdict tracks a function's length, not whether it is vulnerable** — on validation
  it raises a concern about 93.5% of harmless functions against 85.0% of vulnerable ones
  (run `6c942e6c…`, an experiment).
  Use it to order the lines of a function already suspected, never as a detector.
- A reason is given only where a known weakness pattern matches the text of a line. A
  pattern is a text match, never the model's reasoning.

The first request after the Space wakes waits for three models to load (about a minute on
the free CPU).
