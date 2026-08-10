# Design.md

Visual design system for the project's user-facing surfaces: the demo UI, API documentation, and research figures.

> `[ASSUMPTION]` The proposal deck contains no written design specification. The palette and type choices below are **derived from the deck's existing visual language** (near-white gradient background, a single strong blue accent, pale blue content cards, serif headings, sans-serif body, thick left accent bars). Hex values are approximations of what appears in the slides and should be confirmed against the source file before final use.

---

## 1. Design Principles

1. **The code is the interface.** Chrome recedes; the highlighted source line is the loudest thing on screen.
2. **Risk is read before it is parsed.** Severity is legible from colour and position alone, before reading a word.
3. **Explanation sits next to evidence.** Never a reason on one screen and the line on another.
4. **Calm, not alarming.** This is a review tool, not an incident alert. No red-flashing dashboards; the developer is triaging, not firefighting.
5. **Accessible by default.** Colour is never the only carrier of meaning — always paired with an icon, a label, or a score.

---

## 2. Colour System

### 2.1 Brand / Core (from the deck)

| Token | Hex | Use |
|-------|-----|-----|
| `--brand-primary` | `#0B76B8` | Accent bars, active states, primary buttons, numbered badges |
| `--brand-primary-dark` | `#075C90` | Hover, pressed, text-on-light links |
| `--brand-surface` | `#C5E7FA` | Filled content cards (deck's "Research Gaps" / "Novelty" cards) |
| `--brand-surface-soft` | `#E8F4FC` | Subtle section backgrounds, table header rows |
| `--brand-border` | `#9CCDE8` | Card outlines, dividers on tinted surfaces |

### 2.2 Neutrals

| Token | Hex | Use |
|-------|-----|-----|
| `--bg-page` | `#FAFBFC` | Page background (deck's near-white) |
| `--bg-gradient` | `linear-gradient(160deg, #FAFBFC 0%, #EEF5FA 100%)` | Optional page wash, matching slide backgrounds |
| `--bg-elevated` | `#FFFFFF` | Cards, panels, modals |
| `--text-primary` | `#111418` | Headings (deck uses near-black serif) |
| `--text-body` | `#2E353D` | Body copy |
| `--text-muted` | `#697682` | Captions, metadata, line numbers |
| `--border-subtle` | `#DEE4EA` | Default borders, table rules |

### 2.3 Risk / Severity Scale

The one place colour carries real semantics. A **sequential** scale, not a traffic light — risk is continuous.

| Token | Hex | Score range | Label | Icon |
|-------|-----|-------------|-------|------|
| `--risk-critical` | `#A4243B` | 0.90 – 1.00 | Critical | ▲▲ |
| `--risk-high` | `#D1495B` | 0.70 – 0.89 | High | ▲ |
| `--risk-medium` | `#E08A2E` | 0.45 – 0.69 | Medium | ● |
| `--risk-low` | `#C9A227` | 0.20 – 0.44 | Low | ○ |
| `--risk-none` | `#3F8F5C` | 0.00 – 0.19 | Clean | ✓ |

Line-highlight backgrounds use the same hues at 10–14% alpha with a 3px solid left border at full opacity — echoing the deck's accent-bar treatment.

`[RECOMMENDATION]` Thresholds are placeholders until calibration in Phase 6. Bind them to a config file (`configs/ui/risk_bands.yaml`) so the UI updates when the model's calibration does. Do not hard-code.

### 2.4 Semantic (system feedback)

| Token | Hex | Use |
|-------|-----|-----|
| `--info` | `#0B76B8` | Neutral notices |
| `--success` | `#3F8F5C` | Analysis complete |
| `--warning` | `#E08A2E` | Partial parse, truncated input |
| `--error` | `#A4243B` | Parse failure, API error |

### 2.5 Graph Visualization

For rendering AST/CFG/DFG/CPG figures — categorical, colour-blind-safe (Okabe–Ito derived):

| Element | Hex |
|---------|-----|
| AST node | `#5B8FF9` |
| CFG node | `#61DDAA` |
| DFG node | `#F6BD16` |
| Call node | `#7262FD` |
| Flagged node | `#D1495B` (+ 2px outline) |
| Edge default | `#B8C2CC` |
| Evidence path edge | `#0B76B8`, 2.5px, animated dash in the UI / solid in print |

### 2.6 Dark Mode

`[RECOMMENDATION]` Ship light-only for v1; developers reviewing code will expect dark eventually. Reserve tokens now: invert neutrals to `#0E1216` / `#161C22` / `#E6EBF0`, keep brand hue, lift risk colours ~12% in lightness for contrast on dark.

---

## 3. Typography

Matching the deck: **serif display + sans body**.

| Role | Family | Fallback stack |
|------|--------|----------------|
| Display / H1–H2 | A high-contrast transitional serif (deck style) — `[RECOMMENDATION]` **Source Serif 4** or **Lora** | `Georgia, 'Times New Roman', serif` |
| Body / UI | **Inter** | `-apple-system, 'Segoe UI', Roboto, sans-serif` |
| Code | **JetBrains Mono** | `'SF Mono', Consolas, 'Liberation Mono', monospace` |

All three are open-licensed and self-hostable — no external font CDN in the Docker image.

### 3.1 Type Scale (1.25 modular, 16px base)

| Token | Size / Line height | Weight | Use |
|-------|-------------------|--------|-----|
| `display` | 39 / 46 | 700 serif | Page title |
| `h1` | 31 / 38 | 700 serif | Section heading |
| `h2` | 25 / 32 | 600 serif | Card heading |
| `h3` | 20 / 28 | 600 sans | Sub-heading |
| `body-lg` | 18 / 28 | 400 sans | Explanation text |
| `body` | 16 / 24 | 400 sans | Default |
| `body-sm` | 14 / 20 | 400 sans | Metadata, captions |
| `caption` | 12 / 16 | 500 sans | Labels, badges |
| `code` | 14 / 22 | 400 mono | Source display |
| `code-sm` | 12 / 20 | 400 mono | Line numbers, gutter |

**Rules:** max 70 characters per line for prose; never justify; never letter-space body text; code is never proportional.

---

## 4. Spacing, Radius, Elevation

- **Spacing scale (4px base):** 4, 8, 12, 16, 24, 32, 48, 64. Nothing off-scale.
- **Radius:** `sm` 4px (badges, inputs) · `md` 8px (cards) · `lg` 12px (panels) · code blocks 8px.
- **Elevation:** flat by default. `e1` = `0 1px 2px rgba(17,20,24,.06)` for cards; `e2` = `0 4px 12px rgba(17,20,24,.10)` for popovers. Nothing heavier — the deck's aesthetic is flat with borders, not drop-shadowed.
- **Borders:** 1px `--border-subtle` default; **3–4px `--brand-primary` left bar** for emphasis blocks, directly reusing the deck's motif.

---

## 5. Core Components

### 5.1 Code Viewer (the primary surface)

```
┌─────────────────────────────────────────────────────┐
│  auth.c · copy_user_input()        [Critical 0.91]  │  ← header
├──┬──────────────────────────────────────────────────┤
│41│  char buf[64];                                   │
│43│  gets(input);                            ░ 0.31  │  ← low, subtle tint
│45│▌ strcpy(buf, input);                     █ 0.94  │  ← flagged: 3px bar
│46│  return buf;                                     │
└──┴──────────────────────────────────────────────────┘
```

- Fixed-width gutter, `code-sm`, `--text-muted`.
- Flagged lines: tinted background + full-opacity left bar + right-aligned score chip.
- Hovering a flagged line dims unrelated lines to 60% and draws the evidence dataflow path in the gutter.
- Clicking opens the explanation panel anchored to that line.

### 5.2 Explanation Panel

Layout order is fixed and mirrors the deck's Expected Output slide: **label → location → reason → evidence**.

```
┌──────────────────────────────────────┐
│ ▲▲  Buffer Overflow      Line 45     │
│                                      │
│ Unbounded input is copied using      │  body-lg
│ strcpy() without size checking.      │
│                                      │
│ ── Evidence ────────────────────     │  caption, muted
│ Data flow:  41 → 43 → 45             │
│ Attention:  strcpy · input · buf     │
│ Graph:      2 CPG nodes  [view →]    │
└──────────────────────────────────────┘
```

Evidence is always visible, never behind a disclosure toggle. If the model has no evidence to show, the explanation is not shown either (FR-15).

### 5.3 Risk Badge

Pill, `caption` weight 600, background at 12% alpha, text at full colour, 1px border at 40% alpha, icon + label + numeric score. Never colour alone.

### 5.4 Summary Bar

Sticky header: file name · verdict · aggregate risk · count of flagged lines · model version. Model version is always visible — reviewers need to know which checkpoint produced a claim.

### 5.5 Empty / Error States

| State | Treatment |
|-------|-----------|
| No vulnerabilities found | `--risk-none` check, plain sentence, plus the disclaimer required by `Rules.md` §7.8 ("not proof of safety") |
| Parse failure | `--error` panel, quotes the failing construct, suggests a fix |
| Truncated input | `--warning` banner naming exactly which lines were not analyzed |

Never show an empty result that looks like a clean result.

---

## 6. Accessibility

- Body text ≥ 4.5:1 contrast; large text ≥ 3:1. `--text-muted` on `--bg-page` verified at 4.6:1.
- Every risk level pairs colour with an icon **and** a text label **and** a number.
- Full keyboard navigation: `Tab` through flagged lines, `Enter` opens explanation, `Esc` closes.
- ARIA: code viewer is a labelled region; flagged lines announce "line 45, critical risk, 0.94, buffer overflow".
- Respect `prefers-reduced-motion` — evidence-path animation becomes static.
- Never rely on hover alone for information.

---

## 7. Research Figure Style

Consistency between the paper and the product is part of the argument.

- **Palette:** §2.5 categorical for graphs; single-hue sequential `#E8F4FC → #0B76B8` for heatmaps (attention maps, confusion matrices).
- **Fonts:** Inter for labels, JetBrains Mono for code snippets in figures, serif only for figure titles in slides (not in paper figures — journals prefer sans).
- **Sizes:** minimum 8pt label text at final print size; 300 DPI raster or, preferably, vector (PDF/SVG).
- **Charts:** no chartjunk, no 3D, no gradients on bars. Error bars mandatory wherever multiple seeds exist (`Rules.md` §5.2).
- **Attention heatmaps:** always accompanied by the source lines they index; a heatmap without its code is unreadable.
- **Generation:** figures are produced by `scripts/make_figures.py` from logged run data. Hand-edited figures are not permitted (`Rules.md` §0.3).

---

## 8. Voice & Tone (output text)

The explanation module's writing is part of the design.

- **One sentence, cause first.** "Unbounded input is copied using `strcpy()` without size checking." — matches the deck's own example.
- **Name the mechanism, not the fear.** Not "This could let attackers take over your system."
- **Concrete symbols.** Reference the actual function/variable names from the code.
- **No hedging stacks.** "may possibly potentially" — pick one level of confidence and let the risk score carry uncertainty.
- **No attack instructions**, ever (`Rules.md` §7.2).
- **Second person is fine, blame is not.** "This line copies…" not "You forgot to…".

---

## 9. Deliverables Checklist

- [ ] `configs/ui/tokens.json` — single source of truth for colours, spacing, type
- [ ] CSS custom properties generated from tokens; no hex literals in components
- [ ] `configs/ui/risk_bands.yaml` — thresholds, updated after Phase 6 calibration
- [ ] Self-hosted font files in the Docker image
- [ ] Component states documented: default, hover, focus, active, disabled, loading, error
- [ ] Contrast audit recorded in `Memory.md`
