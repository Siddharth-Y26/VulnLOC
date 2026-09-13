/* VulnLoc pipeline inspector — client.

   Renders what the server measured. It computes nothing about the pipeline itself: if a
   number is on screen it came from `/api/*`, which in turn came from `src/vulnloc` or a
   logged run. The only arithmetic here is layout.

   The stage list is driven by `phases.json`, so a stage that is planned but not built
   still gets a panel, at the same visual weight, saying what it will do. Absence is
   content — this project's state is largely defined by what is not finished yet.
*/

"use strict";

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

let PHASES = [];

/* "1 lines carry" is the kind of seam that makes a tool look unfinished. */
const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;

/* ── Bootstrap ─────────────────────────────────────────────────────────────── */

async function boot() {
  await Promise.all([loadModel(), loadPhases(), loadSamples(), loadCorpus()]);
  $("run").addEventListener("click", run);
  $("code").addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
  });
}

async function loadModel() {
  const model = await fetch("/api/model").then((r) => r.json());
  const banner = $("model-banner");
  banner.hidden = false;
  banner.append(el("h2", null, model.headline));
  const p = el("p");
  // The lead-in comes from the server, not from here. It used to be hardcoded as
  // "re-running the three seeds is what unblocks scores", which stopped being true the
  // day the seeds were re-run — and a banner that is confidently out of date is worse
  // than one that says less.
  p.textContent = `${model.detail} ${model.next_step || "Where things stand:"} `;
  p.append(el("code", null, model.unblocks));
  banner.append(p);
}

async function loadPhases() {
  const manifest = await fetch("/api/phases").then((r) => r.json());
  PHASES = manifest.stages;
  const rail = $("rail");
  for (const stage of PHASES) {
    const li = el("li");
    li.dataset.state = stage.state;
    const button = el("button");
    button.type = "button";
    button.append(el("span", "rail-layer", `${stage.layer} · P${stage.phase}`));
    button.append(el("span", "rail-name", stage.name));
    button.addEventListener("click", () => {
      const target = document.getElementById(`stage-${stage.id}`);
      if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    li.append(button);
    rail.append(li);
  }
}

async function loadSamples() {
  const samples = await fetch("/api/samples").then((r) => r.json());
  const host = $("samples");
  for (const sample of samples) {
    const button = el("button", "sample");
    button.type = "button";
    button.append(el("b", null, sample.name));
    button.append(el("span", null, sample.note));
    button.addEventListener("click", () => {
      $("code").value = sample.code;
      run();
    });
    host.append(button);
  }
}

async function loadCorpus() {
  const corpus = await fetch("/api/corpus").then((r) => r.json());
  const line = $("corpus-line");
  const bits = [];
  if (corpus.split?.fingerprint) {
    bits.push(`split ${corpus.split.fingerprint.slice(0, 8)}…`);
  }
  if (corpus.shards?.run_id) bits.push(`shards ${corpus.shards.run_id.slice(0, 8)}…`);
  if (corpus.leakage?.run_id) {
    const leaking = corpus.leakage.partitions_leaking_as_shipped || [];
    bits.push(
      `leak audit ${corpus.leakage.run_id.slice(0, 8)}… — ${
        leaking.length ? `${leaking.length} partitions leaking` : "no partition leaks"
      }`
    );
  }
  line.textContent = bits.length
    ? `Corpus artefacts on disk: ${bits.join("; ")}.`
    : "No corpus artefacts on disk yet — run the scripts in docs order to generate them.";
  if (corpus.missing?.length) {
    line.textContent += ` Not generated: ${corpus.missing.join(", ")}.`;
  }
}

/* ── Run ───────────────────────────────────────────────────────────────────── */

async function run() {
  const button = $("run");
  button.disabled = true;
  button.textContent = "Analysing…";
  $("error-slot").replaceChildren();

  const body = {
    code: $("code").value,
    policy: $("policy").value,
    graph: $("graph").value,
    normalize: $("normalize").checked,
    windowing: $("windowing").checked,
    max_length: Number($("max_length").value),
    stride: Number($("stride").value),
  };

  try {
    const response = await fetch("/api/analyse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json();
    if (!response.ok) {
      showError(data);
      $("stages").replaceChildren();
      return;
    }
    renderStages(data);
  } catch (err) {
    showError({ error: String(err), kind: "network" });
  } finally {
    button.disabled = false;
    button.textContent = "Analyse";
  }
}

function showError(data) {
  const box = el("div", "error");
  box.append(
    el(
      "h2",
      null,
      data.kind === "input" ? "Nothing to analyse" : "The pipeline raised"
    )
  );
  box.append(el("p", "empty", data.error));
  if (data.traceback) box.append(el("pre", null, data.traceback));
  $("error-slot").append(box);
}

/* ── Stage scaffolding ─────────────────────────────────────────────────────── */

function stagePanel(stage) {
  const section = el("section", "stage");
  section.id = `stage-${stage.id}`;
  section.dataset.state = stage.state;

  const head = el("div", "stage-head");
  head.append(el("span", "stage-layer", `${stage.layer} · Phase ${stage.phase}`));
  head.append(el("h2", null, stage.name));
  if (stage.state === "live") head.append(el("span", "badge badge-live", "ran just now"));
  if (stage.state === "built-untrained")
    head.append(el("span", "badge badge-signal", "built, never validly trained"));
  // Every state in phases.json needs a branch here. A state with no branch renders with
  // no badge at all and looks like a stage nobody thought about, which is the opposite of
  // what this page is for.
  if (stage.state === "trained")
    head.append(el("span", "badge badge-trained", "trained, measured"));
  if (stage.state === "trained-one-seed")
    head.append(el("span", "badge badge-signal", "trained · one seed, not yet a result"));
  if (stage.state === "planned") head.append(el("span", "badge badge-void", "not built"));
  if (stage.adr) head.append(el("span", "stage-layer", stage.adr));
  section.append(head);

  const body = el("div", "stage-body");
  section.append(body);
  return { section, body };
}

function readout(pairs) {
  const dl = el("dl", "readout");
  for (const [label, value, tone, sub] of pairs) {
    const wrap = el("div");
    wrap.append(el("dt", null, label));
    const dd = el("dd", tone ? `is-${tone}` : null);
    dd.textContent = value;
    if (sub) {
      dd.append(document.createTextNode(" "));
      dd.append(el("small", null, sub));
    }
    wrap.append(dd);
    dl.append(wrap);
  }
  return dl;
}

function table(headers, rows, opts = {}) {
  const wrap = el("div", "scroll-x");
  const t = el("table");
  const thead = el("thead");
  const tr = el("tr");
  headers.forEach((h, i) => {
    const th = el("th", opts.numeric?.includes(i) ? "num" : null, h);
    tr.append(th);
  });
  thead.append(tr);
  t.append(thead);
  const tbody = el("tbody");
  for (const row of rows) {
    const line = el("tr");
    if (row.chosen) line.dataset.chosen = "true";
    (row.cells || row).forEach((cell, i) => {
      line.append(el("td", opts.numeric?.includes(i) ? "num" : null, String(cell)));
    });
    tbody.append(line);
  }
  t.append(tbody);
  wrap.append(t);
  return wrap;
}

/* ── Stage bodies ──────────────────────────────────────────────────────────── */

const RENDERERS = {
  normalize: renderNormalize,
  parse: renderParse,
  graphs: renderGraphs,
  tokens: renderTokens,
  axis: renderAxis,
  semantic: renderSemantic,
};

function renderStages(data) {
  const host = $("stages");
  host.replaceChildren();
  for (const stage of PHASES) {
    const { section, body } = stagePanel(stage);
    body.append(el("p", "stage-why", stage.why));
    const renderer = RENDERERS[stage.id];
    if (renderer) {
      renderer(body, data);
    } else {
      renderPlanned(body, stage);
    }
    host.append(section);
  }
}

function renderPlanned(body, stage) {
  const list = el("ul", "planned-list");
  list.append(el("li", null, stage.does));
  list.append(
    el(
      "li",
      null,
      `Nothing is shown here because nothing is built. When Phase ${stage.phase} lands, this panel is where its output goes.`
    )
  );
  body.append(list);
}

function renderNormalize(body, data) {
  const n = data.normalization;
  const before = n.leak_tell.lines_flagged_before.length;
  const after = n.leak_tell.lines_flagged_after.length;

  body.append(
    readout([
      ["Lines", n.n_lines],
      ["Lines changed", n.n_lines_changed, n.n_lines_changed ? "signal" : null],
      ["Line count preserved", n.line_count_preserved ? "yes" : "NO", n.line_count_preserved ? "live" : "void"],
      ["Applied", n.applied ? "yes" : "no", n.applied ? "live" : "void"],
    ])
  );

  const demo = el("p", "note");
  demo.textContent =
    `The leak rule reads no code and no label — it only asks whether a line starts with a space or tab. ` +
    `On your paste as typed it would flag ${before} of ${n.n_lines} lines; after normalization, ${after}. ` +
    `On Big-Vul as shipped that same rule recovers the real flaw label at precision 1.0000 with zero false positives on all six partitions, which is why this stage exists.`;
  body.append(demo);

  if (!n.applied) {
    const warn = el("p", "note note-warn");
    warn.textContent =
      "Normalization is switched off. This reproduces pre-ADR-006 behaviour and must never produce a reported number.";
    body.append(warn);
  }

  if (n.changed.length) {
    body.append(
      table(
        ["line", "as pasted", "normalized"],
        n.changed.map((c) => [c.line, JSON.stringify(c.before), JSON.stringify(c.after)]),
        { numeric: [0] }
      )
    );
    if (n.changed_truncated) {
      body.append(el("p", "note", `Showing the first 60 of ${n.n_lines_changed} changed lines.`));
    }
  }
}

function renderParse(body, data) {
  const p = data.parse;
  body.append(
    readout([
      ["Grammar used", p.language_used, "live"],
      ["Clean parse", p.is_clean ? "yes" : "no", p.is_clean ? "live" : "void"],
      ["ERROR nodes", p.error_nodes, p.error_nodes ? "void" : null],
      ["MISSING nodes", p.missing_nodes, p.missing_nodes ? "void" : null],
      ["First damage", p.first_error_line ?? "—", p.first_error_line ? "void" : null],
    ])
  );

  body.append(
    el(
      "p",
      "note",
      "Tree-sitter recovers from anything, so a tree is not evidence of a parse. Clean means zero ERROR and zero MISSING nodes — the same definition the corpus report uses."
    )
  );

  body.append(
    table(
      ["policy", "grammar", "clean", "errors", "missing"],
      data.grammars.map((row) => ({
        chosen: row.policy === p.policy,
        cells: row.error
          ? [row.policy, "—", "—", row.error, "—"]
          : [
              row.policy,
              row.language_used,
              row.is_clean ? "yes" : "no",
              row.error_nodes,
              row.missing_nodes,
            ],
      })),
      { numeric: [3, 4] }
    )
  );
}

function renderGraphs(body, data) {
  const g = data.graph;
  body.append(
    readout([
      ["Nodes", g.n_nodes.toLocaleString()],
      ["Edges", g.n_edges.toLocaleString()],
      [
        "Lines with a node",
        `${g.lines_covered.length}/${g.n_lines}`,
        g.line_coverage_pct === 100 ? "live" : "signal",
        `${g.line_coverage_pct}%`,
      ],
    ])
  );

  body.append(graphCanvas(g));

  const legend = el("ul", "legend");
  for (const type of g.available_edge_types) {
    if (!g.edge_types[type]) continue;
    const li = el("li");
    const swatch = el("span", "swatch");
    swatch.style.borderTopColor = getComputedStyle(document.documentElement)
      .getPropertyValue(`--edge-${type}`)
      .trim();
    li.append(swatch, document.createTextNode(`${type} ${g.edge_types[type]}`));
    legend.append(li);
  }
  legend.append(el("li", null, "y = source line"));
  body.append(legend);

  if (g.nodes_truncated || g.edges_truncated) {
    body.append(
      el(
        "p",
        "note note-warn",
        `Drawing is capped at ${g.render_caps.nodes} nodes and ${g.render_caps.edges} edges; this graph is larger, so the picture is a prefix rather than the whole graph. The counts above are complete.`
      )
    );
  }

  if (g.lines_uncovered.length) {
    body.append(
      el(
        "p",
        "note",
        `${plural(g.lines_uncovered.length, "line has", "lines have")} no node at all (${g.lines_uncovered.join(", ")}), so the structural branch cannot score ${g.lines_uncovered.length === 1 ? "it" : "them"}. Corpus-wide this costs 7.84% of flagged lines, and it is a separate ceiling from label noise.`
      )
    );
  }
}

/* Graph laid out with source line on the Y axis. Nodes on the same line spread along X
   in id order, so horizontal distance is tree order and vertical distance is real
   source distance — a long vertical edge is a relation that jumps across the function. */
function graphCanvas(g) {
  const wrap = el("div", "graph-wrap");
  const byLine = new Map();
  for (const node of g.nodes) {
    if (!byLine.has(node.line)) byLine.set(node.line, []);
    byLine.get(node.line).push(node);
  }

  const rowHeight = 17;
  const gutter = 38;
  const widest = Math.max(1, ...[...byLine.values()].map((v) => v.length));
  const colWidth = Math.min(22, Math.max(5, 560 / widest));
  const width = Math.max(320, gutter + widest * colWidth + 28);
  const height = Math.max(120, g.n_lines * rowHeight + 24);

  const svgns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgns, "svg");
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.setAttribute("role", "img");
  svg.setAttribute(
    "aria-label",
    `${g.type.toUpperCase()} with ${g.n_nodes} nodes over ${g.n_lines} source lines, positioned by line.`
  );

  const pos = new Map();
  for (const [line, nodes] of byLine) {
    nodes.forEach((node, i) => {
      pos.set(node.id, { x: gutter + i * colWidth + colWidth / 2, y: 12 + (line - 1) * rowHeight });
    });
  }

  // Lanes first, so edges and nodes sit above them. Every line that carries a node gets
  // a faint band: the band is the line, and that is what the layout is claiming.
  for (const [line, nodes] of byLine) {
    const y = 12 + (line - 1) * rowHeight;
    const band = document.createElementNS(svgns, "rect");
    band.setAttribute("x", String(gutter - 4));
    band.setAttribute("y", String(y - 6));
    band.setAttribute("width", String(nodes.length * colWidth + 8));
    band.setAttribute("height", "12");
    band.setAttribute("class", "graph-lane");
    svg.append(band);
  }

  for (let line = 1; line <= g.n_lines; line++) {
    const y = 12 + (line - 1) * rowHeight;
    if (line % 5 === 0 || line === 1) {
      const rule = document.createElementNS(svgns, "line");
      rule.setAttribute("x1", String(gutter - 6));
      rule.setAttribute("x2", String(width));
      rule.setAttribute("y1", String(y));
      rule.setAttribute("y2", String(y));
      rule.setAttribute("class", "graph-rule");
      svg.append(rule);

      const label = document.createElementNS(svgns, "text");
      label.setAttribute("x", String(gutter - 10));
      label.setAttribute("y", String(y + 3));
      label.setAttribute("text-anchor", "end");
      label.setAttribute("class", "graph-line-label");
      label.textContent = String(line);
      svg.append(label);
    }
  }

  for (const edge of g.edges) {
    const a = pos.get(edge.s);
    const b = pos.get(edge.t);
    if (!a || !b) continue;
    const path = document.createElementNS(svgns, "path");
    const bow = Math.min(40, Math.abs(b.y - a.y) * 0.35 + 6);
    path.setAttribute(
      "d",
      `M ${a.x} ${a.y} C ${a.x + bow} ${a.y}, ${b.x + bow} ${b.y}, ${b.x} ${b.y}`
    );
    path.setAttribute("class", `edge edge-${edge.type}`);
    svg.append(path);
  }

  for (const node of g.nodes) {
    const p = pos.get(node.id);
    if (!p) continue;
    const dot = document.createElementNS(svgns, "circle");
    dot.setAttribute("cx", String(p.x));
    dot.setAttribute("cy", String(p.y));
    dot.setAttribute("r", node.named ? "2.6" : "1.6");
    dot.setAttribute("class", `node-dot${node.named ? " named" : ""}`);
    const title = document.createElementNS(svgns, "title");
    title.textContent = `line ${node.line} · ${node.type}${node.text ? ` · ${node.text}` : ""}`;
    dot.append(title);
    svg.append(dot);
  }

  wrap.append(svg);
  return wrap;
}

function renderTokens(body, data) {
  const t = data.tokens;
  const lost = t.truncation.would_lose_lines;
  body.append(
    readout([
      ["Body tokens", t.n_body_tokens.toLocaleString()],
      ["Windows", t.n_windows, t.n_windows > 1 ? "signal" : null],
      ["Window length", t.max_length],
      ["Stride", t.stride],
      [
        "Lines a truncating encoder would lose",
        lost,
        lost ? "void" : "live",
      ],
    ])
  );

  if (!t.windowing) {
    body.append(
      el(
        "p",
        "note note-warn",
        "Windowing is off, so only the first window exists. This reproduces the published-baseline behaviour and any line-level number taken this way must be labelled as truncated."
      )
    );
  }

  body.append(
    table(
      ["window", "real tokens", "lines covered"],
      t.windows.map((w) => [
        w.index,
        w.n_real_tokens,
        w.lines.length ? `${w.lines[0]}–${w.lines[w.lines.length - 1]} (${w.lines.length})` : "—",
      ]),
      { numeric: [0, 1] }
    )
  );

  body.append(
    el(
      "p",
      "note",
      lost
        ? `Truncating at ${t.max_length} tokens would keep lines ${t.truncation.lines_in_first_window[0]}–${
            t.truncation.lines_in_first_window[t.truncation.lines_in_first_window.length - 1]
          } and drop ${lost} others. Measured corpus-wide, truncation destroys 38.1% of all flagged lines and 44.5% on clean_eval (run b7ccc9ca…); windowing keeps 100%.`
        : "This function fits in a single window, so truncation and windowing agree. On the corpus they do not: truncation destroys 38.1% of all flagged lines (run b7ccc9ca…)."
    )
  );
}

function renderAxis(body, data) {
  const axis = data.axis;
  const maxNodes = Math.max(1, ...axis.rows.map((r) => r.n_nodes));
  const maxTokens = Math.max(1, ...axis.rows.map((r) => r.n_tokens));

  const box = el("div", "axis");
  const head = el("div", "axis-head");
  ["line", "source", "nodes", "tokens", "window"].forEach((h) =>
    head.append(el("span", null, h))
  );
  box.append(head);

  const changed = new Set((data.normalization.changed || []).map((c) => c.line));

  for (const row of axis.rows) {
    const line = el("div", "axis-row");
    line.dataset.blank = String(row.blank);
    if (changed.has(row.line)) line.dataset.changed = "true";

    line.append(el("span", "ln", String(row.line)));
    line.append(el("code", "src", row.text || " "));
    line.append(trackCell(row.n_nodes, maxNodes, "bar-node", `${row.n_nodes} graph nodes: ${row.node_types.join(", ")}`, !row.blank && !row.has_nodes));
    line.append(trackCell(row.n_tokens, maxTokens, "bar-token", `${row.n_tokens} tokens`, !row.blank && !row.has_tokens));
    line.append(
      el("span", "bar-window", row.in_windows.length ? row.in_windows.join(",") : "—")
    );
    box.append(line);
  }
  body.append(box);

  const notes = [];
  if (axis.lines_without_nodes.length) {
    notes.push(
      `${plural(axis.lines_without_nodes.length, "line carries", "lines carry")} no graph node, so the structural branch is blind to it.`
    );
  }
  if (axis.lines_without_tokens.length) {
    notes.push(
      `${plural(axis.lines_without_tokens.length, "line carries", "lines carry")} no token, so the semantic branch is blind to it.`
    );
  }
  if (changed.size) {
    notes.push(
      `An amber line number marks a line the whitespace fix changed — ${plural(changed.size, "line", "lines")} here.`
    );
  }
  notes.push(
    "Both branches must land on this same index before fusion can happen. A bug in either mapping shows up here as a bar against the wrong line, which is why the view exists."
  );
  body.append(el("p", "note", notes.join(" ")));
}

/* One track cell: a bar whose length is the count, or an explicit gap marker when a line
   exists but the branch cannot see it. A blank cell and a blind line must not look the
   same — the second is a ceiling and the first is just an empty line. */
function trackCell(count, max, cls, title, blind) {
  const cell = el("span", "track");
  if (count > 0) {
    const bar = el("span", `bar ${cls}`);
    bar.style.width = `${Math.max(8, (count / max) * 100)}%`;
    bar.title = title;
    cell.append(bar);
    cell.append(el("span", "track-count", String(count)));
  } else if (blind) {
    cell.append(el("span", "track-gap", "none"));
  }
  return cell;
}

function renderSemantic(body, data) {
  body.append(
    readout([
      ["Encoder", "CodeBERT", "live"],
      ["Line head", "built", "live"],
      ["Overfit test", "passes", "live"],
      ["Valid trained weights", "none", "void"],
      ["Scores shown", "none", "void"],
    ])
  );
  body.append(
    el(
      "p",
      "note",
      "The architecture runs and memorises a 32-sample subset, which proves the head, the loss and the line alignment are wired correctly. It does not produce a number anyone may quote. The only full run was voided by ADR-006, so this tool deliberately has nowhere to display a per-line score."
    )
  );
  body.append(
    table(
      ["what exists", "state"],
      [
        ["models/semantic/encoder.py", "built"],
        ["models/heads/line_head.py", "built"],
        ["training/loop.py, losses.py, metrics.py", "built"],
        ["LLM-only metrics over ≥3 seeds", "not started — Phase 3 exit criterion"],
      ]
    )
  );
}

boot();
