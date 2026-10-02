/* VulnLoc — client.

   Two views over one paste. FINDINGS is for the developer: the lines to review first,
   ranked by the Phase 3 semantic model, with any known weakness pattern on each line and
   its usual fix. DETAILED REPORT shows each pipeline stage's output on the paste, and the
   structural model's own ranking beside the semantic one.

   Renders what the server measured. If a number is on screen it came from `/api/*`, which
   in turn came from `src/vulnloc` or a seed report on disk. The only arithmetic here is
   layout.

   The model and the templates are two different sources and are never blended on screen.
   The model ranks a line and gives no reason; a template matches the line's text.

   Keep it short (owner, 2026-09-15): each stage shows one sentence, its numbers, and its
   picture. Explanations belong in the docs, not on the page.
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
let RESULTS = {};

const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
const pct = (x) => `${Math.round(x * 100)}%`;

/* ── Bootstrap ─────────────────────────────────────────────────────────────── */

async function boot() {
  setupTabs();
  await Promise.all([loadResults(), loadPhases(), loadSamples()]);
  $("run").addEventListener("click", run);
  $("code").addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
  });

  // `?sample=0&tab=report` opens straight onto a sample — for sharing a view, and for
  // headless screenshots that cannot click.
  const params = new URLSearchParams(location.search);
  if (TABS.includes(params.get("tab"))) selectTab(params.get("tab"));
  const index = Number(params.get("sample"));
  if (params.has("sample") && SAMPLES[index]) {
    $("code").value = SAMPLES[index].code;
    run();
  }
}

/* ── Tabs ──────────────────────────────────────────────────────────────────── */

const TABS = ["findings", "report"];

function selectTab(name, focus = false) {
  for (const tab of TABS) {
    const button = $(`tab-${tab}`);
    const selected = tab === name;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    $(`panel-${tab}`).hidden = !selected;
    if (selected && focus) button.focus();
  }
  try {
    localStorage.setItem("vulnloc.tab", name);
  } catch {
    /* Storage can be unavailable; the tab still switches, it just is not remembered. */
  }
}

function setupTabs() {
  TABS.forEach((name, index) => {
    const button = $(`tab-${name}`);
    button.addEventListener("click", () => selectTab(name));
    button.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const step = e.key === "ArrowRight" ? 1 : -1;
      selectTab(TABS[(index + step + TABS.length) % TABS.length], true);
    });
  });
  let remembered = null;
  try {
    remembered = localStorage.getItem("vulnloc.tab");
  } catch {
    remembered = null;
  }
  selectTab(TABS.includes(remembered) ? remembered : "findings");
}

async function loadResults() {
  RESULTS = await fetch("/api/results").then((r) => r.json());
}

async function loadPhases() {
  const manifest = await fetch("/api/phases").then((r) => r.json());
  PHASES = manifest.stages;
  const rail = $("rail");
  for (const stage of builtStages()) {
    const li = el("li");
    li.dataset.state = stage.state;
    const button = el("button");
    button.type = "button";
    button.append(el("span", "rail-name", stage.name));
    button.addEventListener("click", () => {
      const target = document.getElementById(`stage-${stage.id}`);
      if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    li.append(button);
    rail.append(li);
  }
}

const builtStages = () => PHASES.filter((s) => s.state !== "planned");
const plannedStages = () => PHASES.filter((s) => s.state === "planned");

let SAMPLES = [];

async function loadSamples() {
  const samples = await fetch("/api/samples").then((r) => r.json());
  SAMPLES = samples;
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
    button.title = sample.note;
    host.append(button);
  }
}

/* ── Run ───────────────────────────────────────────────────────────────────── */

async function postJson(url, body) {
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    let data;
    try {
      data = await response.json();
    } catch {
      data = {};
    }
    if (!response.ok) {
      return { error: { ...data, status: response.status } };
    }
    return { data };
  } catch (err) {
    return { error: { error: String(err), kind: "network" } };
  }
}

async function run() {
  const button = $("run");
  button.disabled = true;
  button.textContent = "Analysing…";
  $("error-slot").replaceChildren();

  // Three independent requests: the report is fast, the models are not on their first
  // call, and none should wait for another.
  const code = $("code").value;
  const findings = postJson("/api/findings", { code });
  const structural = postJson("/api/structural", { code });
  await Promise.all([runReport(code, findings, structural), runFindings(findings)]);
  button.disabled = false;
  button.textContent = "Analyse";
}

async function runFindings(pending) {
  const host = $("findings");
  host.replaceChildren(el("p", "empty loading", "Ranking lines… the first run loads the model."));
  const { data, error } = await pending;
  host.replaceChildren();
  if (error) {
    host.append(errorBox(error));
    return;
  }
  renderFindings(host, data);
}

async function runReport(code, findings, structural) {
  const body = {
    code,
    policy: $("policy").value,
    graph: $("graph").value,
    normalize: $("normalize").checked,
    windowing: $("windowing").checked,
    max_length: Number($("max_length").value),
    stride: Number($("stride").value),
  };
  const { data, error } = await postJson("/api/analyse", body);
  if (error) {
    showError(error);
    $("stages").replaceChildren();
    return;
  }
  renderStages(data, { findings, structural });
}

function errorBox(data) {
  const box = el("div", "error");
  const title =
    data.kind === "input"
      ? "Nothing to analyse"
      : data.kind === "model"
        ? "The model raised"
        : data.kind === "network"
          ? "Could not reach the server"
          : "The pipeline raised";
  box.append(el("h2", null, title));
  // FastAPI's own errors (404, 405, 422) carry `detail`, not `error`. A box with a title
  // and no message is what a stale server looked like on 2026-09-15.
  const detail =
    data.error ||
    (typeof data.detail === "string" ? data.detail : data.detail && JSON.stringify(data.detail)) ||
    "No message.";
  box.append(el("p", "empty", data.status ? `HTTP ${data.status}: ${detail}` : detail));
  if (data.status === 404 || data.status === 405) {
    box.append(el("p", "empty", "The server may be older than this page. Restart it."));
  }
  if (data.traceback) box.append(el("pre", null, data.traceback));
  return box;
}

function showError(data) {
  $("error-slot").append(errorBox(data));
}

/* ── Findings ──────────────────────────────────────────────────────────────── */

function renderFindings(host, data) {
  const m = data.measured;
  const run = m.run || {};

  const trust = el("div", "trust");
  const trustHead = el("p", "trust-lead");
  trustHead.append(el("b", null, "Lines to review first. "));
  trustHead.append(
    document.createTextNode(
      run.top_5 != null
        ? `On held-out data the real flawed line was in its top ${data.top_k} ${pct(run.top_5)} of the time.`
        : ""
    )
  );
  trust.append(trustHead);
  const caveats = el("ul", "trust-caveats");
  caveats.append(
    el("li", null, "It assumes there is a flaw and ranks where. It cannot tell you code is safe."),
    el("li", null, "Fix advice comes from pattern templates, not from the model.")
  );
  trust.append(caveats);
  host.append(trust);

  const list = el("ol", "findings");
  for (const finding of data.top) {
    list.append(findingCard(finding));
  }
  host.append(list);

  if (data.patterns_elsewhere.length) {
    const other = el("section", "elsewhere");
    other.append(el("h3", null, "Known patterns on lower-ranked lines"));
    const rows = el("ul", "elsewhere-list");
    for (const item of data.patterns_elsewhere) {
      for (const p of item.patterns) {
        const li = el("li");
        li.append(el("span", "ln-chip", `L${item.line}`));
        li.append(el("span", "cwe", p.cwe));
        li.append(el("span", null, `${p.title} — `));
        li.append(el("code", null, item.text));
        rows.append(li);
      }
    }
    other.append(rows);
    host.append(other);
  }

  host.append(sourceMap(data));

  const foot = el("p", "note");
  foot.textContent = "Scores are for ordering only, not probabilities.";
  if (data.beyond_line_cap) {
    foot.textContent += ` ${plural(data.beyond_line_cap, "line", "lines")} past line 512 cannot be scored.`;
  }
  host.append(foot);
}

function findingCard(finding) {
  const li = el("li", "finding");
  li.id = `finding-L${finding.line}`;

  const head = el("div", "finding-head");
  head.append(el("span", "finding-rank", `#${finding.rank}`));
  head.append(el("span", "finding-line", `Line ${finding.line}`));
  const meter = el("span", "meter");
  meter.title = `score ${finding.score.toFixed(3)}`;
  const fill = el("span", "meter-fill");
  fill.style.width = `${Math.max(4, finding.score * 100)}%`;
  meter.append(fill);
  head.append(meter);
  head.append(
    el(
      "span",
      finding.patterns.length ? "finding-tag has-pattern" : "finding-tag",
      finding.patterns.length ? plural(finding.patterns.length, "known pattern", "known patterns") : "no known pattern"
    )
  );
  li.append(head);

  const snippet = el("div", "snippet");
  for (const row of finding.context) {
    const line = el("div", "snippet-row");
    if (row.line === finding.line) line.dataset.focus = "true";
    line.append(el("span", "ln", String(row.line)));
    line.append(el("code", "src", row.text || " "));
    snippet.append(line);
  }
  li.append(snippet);

  for (const p of finding.patterns) {
    const block = el("div", "pattern");
    const title = el("p", "pattern-title");
    title.append(el("span", "cwe", p.cwe), document.createTextNode(p.title));
    block.append(title);

    const check = el("p", "pattern-text");
    check.append(el("b", null, "Check  "), document.createTextNode(p.problem));
    block.append(check);
    const fix = el("p", "pattern-text");
    fix.append(el("b", null, "Fix  "), document.createTextNode(p.fix));
    block.append(fix);

    const pair = el("div", "before-after");
    const before = el("div");
    before.append(el("span", "pair-label", "typical"), el("pre", null, p.example.before));
    const after = el("div");
    after.append(el("span", "pair-label", "safer"), el("pre", null, p.example.after));
    pair.append(before, after);
    block.append(pair);
    li.append(block);
  }
  return li;
}

/* The whole function with the model's ordering alongside. Top-ranked lines carry their rank
   and link to their card; every other scored line is shaded by score, faintly. */
function sourceMap(data) {
  const wrap = el("section", "source-map");
  wrap.append(el("h3", null, "Whole function"));
  const rankByLine = new Map(data.top.map((f) => [f.line, f.rank]));
  const box = el("div", "axis");
  for (const entry of data.scores) {
    const row = el("div", "map-row");
    const rank = rankByLine.get(entry.line);
    if (rank) row.dataset.top = "true";
    row.append(el("span", "ln", String(entry.line)));
    const heat = el("span", "heat");
    if (entry.score == null) {
      heat.textContent = "—";
      heat.title = "not scored";
    } else if (rank) {
      const a = el("a", "heat-rank", `#${rank}`);
      a.href = `#finding-L${entry.line}`;
      heat.append(a);
    } else {
      heat.style.setProperty("--heat", String(entry.score));
      heat.classList.add("heat-shade");
      heat.title = `rank ${entry.rank}`;
    }
    row.append(heat);
    row.append(el("code", "src", data.lines[entry.line - 1] || " "));
    box.append(row);
  }
  wrap.append(box);
  return wrap;
}

/* ── Stage scaffolding ─────────────────────────────────────────────────────── */

const BADGES = {
  live: ["badge badge-live", "ran on your code"],
  trained: ["badge badge-trained", "trained"],
};

function stagePanel(stage) {
  const section = el("section", "stage");
  section.id = `stage-${stage.id}`;
  section.dataset.state = stage.state;

  const head = el("div", "stage-head");
  head.append(el("span", "stage-layer", `Phase ${stage.phase}`));
  head.append(el("h2", null, stage.name));
  const badge = BADGES[stage.state];
  if (badge) head.append(el("span", badge[0], badge[1]));
  section.append(head);

  const body = el("div", "stage-body");
  body.append(el("p", "stage-why", stage.does));
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
    tr.append(el("th", opts.numeric?.includes(i) ? "num" : null, h));
  });
  thead.append(tr);
  t.append(thead);
  const tbody = el("tbody");
  for (const row of rows) {
    const line = el("tr");
    if (row.chosen) line.dataset.chosen = "true";
    (row.cells || row).forEach((cell, i) => {
      const td = el("td", opts.numeric?.includes(i) ? "num" : null);
      if (cell instanceof Node) td.append(cell);
      else td.textContent = String(cell);
      line.append(td);
    });
    tbody.append(line);
  }
  t.append(tbody);
  wrap.append(t);
  return wrap;
}

/* A collapsed extra: the detail a curious reader can open, out of everyone else's way. */
function more(label, content) {
  const d = el("details", "more");
  d.append(el("summary", null, label));
  d.append(content);
  return d;
}

/* ── Stage bodies ──────────────────────────────────────────────────────────── */

const RENDERERS = {
  normalize: renderNormalize,
  parse: renderParse,
  graphs: renderGraphs,
  tokens: renderTokens,
  axis: renderAxis,
  semantic: renderSemantic,
  structural: renderStructural,
  fusion: renderFusion,
};

function renderStages(data, pending) {
  const host = $("stages");
  host.replaceChildren();
  for (const stage of builtStages()) {
    const { section, body } = stagePanel(stage);
    const renderer = RENDERERS[stage.id];
    if (renderer) renderer(body, data, pending);
    host.append(section);
  }
  host.append(renderPlanned());
}

function renderPlanned() {
  const section = el("section", "stage");
  section.dataset.state = "planned";
  const head = el("div", "stage-head");
  head.append(el("h2", null, "Not built yet"));
  section.append(head);
  const body = el("div", "stage-body");
  const list = el("ul", "planned-list");
  for (const stage of plannedStages()) {
    const li = el("li");
    li.append(el("b", null, `Phase ${stage.phase} · ${stage.name}: `));
    li.append(document.createTextNode(stage.does));
    list.append(li);
  }
  body.append(list);
  section.append(body);
  return section;
}

function renderNormalize(body, data) {
  const n = data.normalization;
  body.append(
    readout([
      ["Lines", n.n_lines],
      ["Lines changed", n.n_lines_changed],
    ])
  );
  if (!n.applied) {
    body.append(el("p", "note note-warn", "Normalization is off. The models expect it on."));
  }
  if (n.changed.length) {
    body.append(
      more(
        "Show changed lines",
        table(
          ["line", "before", "after"],
          n.changed.map((c) => [c.line, JSON.stringify(c.before), JSON.stringify(c.after)]),
          { numeric: [0] }
        )
      )
    );
  }
}

function renderParse(body, data) {
  const p = data.parse;
  const pairs = [
    ["Grammar", p.language_used === "cpp" ? "C++" : p.language_used === "c" ? "C" : p.language_used, "live"],
    ["Clean parse", p.is_clean ? "yes" : "no", p.is_clean ? "live" : "void"],
  ];
  if (!p.is_clean) {
    pairs.push(["Errors", p.error_nodes + p.missing_nodes, "void"]);
    pairs.push(["First error on line", p.first_error_line ?? "—", "void"]);
  }
  body.append(readout(pairs));
  body.append(
    more(
      "Compare grammars",
      table(
        ["policy", "grammar", "clean", "errors"],
        data.grammars.map((row) => ({
          chosen: row.policy === p.policy,
          cells: row.error
            ? [row.policy, "—", "—", row.error]
            : [row.policy, row.language_used, row.is_clean ? "yes" : "no", row.error_nodes + row.missing_nodes],
        })),
        { numeric: [3] }
      )
    )
  );
}

function renderGraphs(body, data) {
  const g = data.graph;
  const pairs = [
    ["Nodes", g.n_nodes.toLocaleString()],
    ["Edges", g.n_edges.toLocaleString()],
    ["Lines with a node", `${g.lines_covered.length}/${g.n_lines}`],
  ];
  body.append(readout(pairs));
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
  legend.append(el("li", null, "rows = source lines"));
  body.append(legend);

  if (g.nodes_truncated || g.edges_truncated) {
    body.append(el("p", "note", `Picture shows the first ${g.render_caps.nodes} nodes only.`));
  }
}

/* Graph laid out with source line on the Y axis. Nodes on the same line spread along X
   in id order, so a long vertical edge is a relation that jumps across the function. */
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
    path.setAttribute("d", `M ${a.x} ${a.y} C ${a.x + bow} ${a.y}, ${b.x + bow} ${b.y}, ${b.x} ${b.y}`);
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
  body.append(
    readout([
      ["Tokens", t.n_body_tokens.toLocaleString()],
      ["Windows", t.n_windows],
    ])
  );
  if (!t.windowing && t.truncation.would_lose_lines) {
    body.append(
      el("p", "note note-warn", `Windowing is off: ${plural(t.truncation.would_lose_lines, "line is", "lines are")} cut.`)
    );
  }
  if (t.n_windows > 1) {
    body.append(
      table(
        ["window", "tokens", "lines"],
        t.windows.map((w) => [
          w.index,
          w.n_real_tokens,
          w.lines.length ? `${w.lines[0]}–${w.lines[w.lines.length - 1]}` : "—",
        ]),
        { numeric: [0, 1] }
      )
    );
  }
}

function renderAxis(body, data) {
  const axis = data.axis;
  const maxNodes = Math.max(1, ...axis.rows.map((r) => r.n_nodes));
  const maxTokens = Math.max(1, ...axis.rows.map((r) => r.n_tokens));

  const box = el("div", "axis");
  const head = el("div", "axis-head");
  ["line", "source", "nodes", "tokens", "window"].forEach((h) => head.append(el("span", null, h)));
  box.append(head);

  for (const row of axis.rows) {
    const line = el("div", "axis-row");
    line.dataset.blank = String(row.blank);
    line.append(el("span", "ln", String(row.line)));
    line.append(el("code", "src", row.text || " "));
    line.append(trackCell(row.n_nodes, maxNodes, "bar-node", `${row.n_nodes} graph nodes: ${row.node_types.join(", ")}`, !row.blank && !row.has_nodes));
    line.append(trackCell(row.n_tokens, maxTokens, "bar-token", `${row.n_tokens} tokens`, !row.blank && !row.has_tokens));
    line.append(el("span", "bar-window", row.in_windows.length ? row.in_windows.join(",") : "—"));
    box.append(line);
  }
  body.append(box);
}

/* A bar whose length is the count, or "none" when a line has code but a branch cannot see it. */
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

/* Held-out accuracy for one model, from its seed report. */
function accuracy(name, label) {
  const r = RESULTS[name];
  if (!r || r.missing) return [[label, "no report"]];
  return [
    [`${label} top-5 hit rate`, pct(r.top_5), "live", `± ${Math.round(r.top_5_std * 100)} pts`],
    [`${label} top-1 hit rate`, pct(r.top_1)],
  ];
}

function renderSemantic(body) {
  body.append(readout(accuracy("semantic", "Held-out")));
  const go = el("button", "link-button", "See its ranking in Findings →");
  go.type = "button";
  go.addEventListener("click", () => selectTab("findings", true));
  body.append(go);
}

async function renderStructural(body, _data, pending) {
  body.append(readout(accuracy("structural", "Held-out")));

  const slot = el("div");
  slot.append(el("p", "empty loading", "Ranking lines… the first run loads the model."));
  body.append(slot);

  const [structural, findings] = await Promise.all([pending.structural, pending.findings]);
  slot.replaceChildren();
  if (structural.error) {
    slot.append(errorBox(structural.error));
    return;
  }
  const s = structural.data;
  const semanticRank = new Map((findings.data?.scores || []).map((e) => [e.line, e.rank]));

  slot.append(el("h3", "sub", "Top lines on your code"));
  slot.append(
    table(
      ["#", "line", "code", "semantic model's rank"],
      s.top.map((e) => {
        const code = el("code", null, e.text);
        const other = semanticRank.get(e.line);
        return [e.rank, e.line, code, other ?? "—"];
      }),
      { numeric: [0, 1, 3] }
    )
  );
  if (s.lines_without_node.length) {
    slot.append(
      el("p", "note", `Not scored (no graph node): ${plural(s.lines_without_node.length, "line", "lines")} — ${s.lines_without_node.join(", ")}.`)
    );
  }
}

function renderFusion(body) {
  const rows = [
    ["cross_attention", "Cross-attention"],
    ["concat", "Concatenation"],
    ["sum", "Sum"],
    ["gated", "Gated"],
  ];
  const semantic = RESULTS.semantic;
  const tableRows = rows
    .filter(([name]) => RESULTS[name] && !RESULTS[name].missing)
    .map(([name, label]) => ({
      chosen: name === "cross_attention",
      cells: [label, pct(RESULTS[name].top_5), pct(RESULTS[name].top_1)],
    }));
  if (semantic && !semantic.missing) {
    tableRows.push({ cells: ["Semantic model alone (Findings)", pct(semantic.top_5), pct(semantic.top_1)] });
  }
  body.append(table(["method", "top-5 hit rate", "top-1 hit rate"], tableRows, { numeric: [1, 2] }));
  body.append(
    el("p", "note", "The fusion runs used a frozen semantic model to save GPU time, which is why they score below it. A full run is pending.")
  );
}

boot();
