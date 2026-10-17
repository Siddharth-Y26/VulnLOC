/* VulnLoc — client.

   Two views over one paste. FINDINGS is for the developer: the lines to review first,
   ranked by the released model, with a reason and the usual fix on each line a known
   weakness pattern matches. DETAILED REPORT shows each pipeline stage's output on the
   paste, each branch's own ranking, and what was measured for the checkpoint.

   Renders what the server measured. If a number is on screen it came from `/api/*`, which
   in turn came from `src/vulnloc` or a seed report on disk. The only arithmetic here is
   layout.

   The model and the templates are two different sources and are never blended on screen.
   The model ranks a line and gives no reason; a template matches the line's text.

   Owner, 2026-10-12: no count of flagged lines as a headline, and a reason only for a
   line a pattern matches. The model's ordering stays visible in the whole-function view.

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
let refreshCurrentStage = () => {};

const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
const pct = (x) => `${Math.round(x * 100)}%`;

/* ── Bootstrap ─────────────────────────────────────────────────────────────── */

async function boot() {
  setupTabs();
  await Promise.all([loadResults(), loadPhases(), loadSamples()]);
  refreshCurrentStage = trackCurrentStage();
  $("run").addEventListener("click", () => run());
  $("code").addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
  });

  // `?sample=0&tab=report` opens straight onto a sample — for sharing a view, and for
  // headless screenshots that cannot click. The tab named in the URL is kept.
  const params = new URLSearchParams(location.search);
  const urlTab = TABS.includes(params.get("tab")) ? params.get("tab") : null;
  if (urlTab) selectTab(urlTab);
  const index = Number(params.get("sample"));
  if (params.has("sample") && SAMPLES[index]) {
    pickSample(SAMPLES[index], { keepTab: Boolean(urlTab) });
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
  if (name === "report") refreshCurrentStage();
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
    li.dataset.stage = stage.id;
    const button = el("button");
    button.type = "button";
    button.append(el("span", "rail-name", stage.name));
    button.addEventListener("click", () => {
      const target = document.getElementById(`stage-${stage.id}`);
      if (!target) return;
      const smooth = !matchMedia("(prefers-reduced-motion: reduce)").matches;
      target.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "start" });
      markCurrentStage(stage.id);
    });
    li.append(button);
    rail.append(li);
  }
}

/* The sidebar marks the stage being read: the last one whose top is within a short reach
   of the results column's top. Recomputed on scroll, which is cheap at a dozen stages. */
function markCurrentStage(id) {
  for (const li of $("rail").children) {
    const button = li.querySelector("button");
    if (li.dataset.stage === id) button.setAttribute("aria-current", "true");
    else button.removeAttribute("aria-current");
  }
}

function trackCurrentStage() {
  const column = $("results");
  let queued = false;
  const update = () => {
    queued = false;
    if ($("panel-report").hidden) return;
    const sections = [...document.querySelectorAll("#stages > .stage[id]")];
    if (!sections.length) return;
    // Wide screens scroll the column; narrow ones scroll the page.
    const scroller = column.scrollHeight > column.clientHeight ? column : null;
    const top = scroller ? scroller.getBoundingClientRect().top : 0;
    // Narrow screens: the sticky header covers the page's top edge.
    const header = scroller ? 0 : document.querySelector(".masthead").offsetHeight;
    const reach = header + 96;
    let current = sections[0];
    for (const section of sections) {
      if (section.getBoundingClientRect().top - top <= reach) current = section;
    }
    // At the very end the last stages can never reach the top; mark the last one.
    const end = scroller
      ? scroller.scrollTop + scroller.clientHeight >= scroller.scrollHeight - 2
      : window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2;
    if (end && (scroller ? scroller.scrollTop : window.scrollY) > 0) current = sections.at(-1);
    markCurrentStage(current.id.replace(/^stage-/, ""));
  };
  const queue = () => {
    if (!queued) {
      queued = true;
      requestAnimationFrame(update);
    }
  };
  column.addEventListener("scroll", queue, { passive: true });
  window.addEventListener("scroll", queue, { passive: true });
  return update;
}

const builtStages = () => PHASES.filter((s) => s.state !== "planned");
const plannedStages = () => PHASES.filter((s) => s.state === "planned");

let SAMPLES = [];

const SAMPLE_GROUPS = [
  ["weakness", "A weakness was written in"],
  ["none", "No weakness intended"],
  ["pipeline", "Pipeline samples"],
];

function pickSample(sample, opts = {}) {
  $("code").value = sample.code;
  run(opts);
}

async function loadSamples() {
  const samples = await fetch("/api/samples").then((r) => r.json());
  SAMPLES = samples;
  const host = $("samples");
  SAMPLE_GROUPS.forEach(([group, title], index) => {
    const members = samples.filter((sample) => sample.group === group);
    if (!members.length) return;
    const box = el("details", "sample-group");
    box.open = index === 0;
    box.append(el("summary", null, `${title} (${members.length})`));
    for (const sample of members) {
      const button = el("button", "sample");
      button.type = "button";
      button.append(el("b", null, sample.name));
      button.append(el("span", null, sample.note));
      button.addEventListener("click", () => pickSample(sample));
      button.title = sample.note;
      box.append(button);
    }
    host.append(box);
  });
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

/* New code always lands on Findings, at the top (owner, 2026-10-17); only a shared
   `?tab=` link keeps the tab it names. */
async function run({ keepTab = false } = {}) {
  if (!keepTab) selectTab("findings");
  $("results").scrollTop = 0;
  const button = $("run");
  button.disabled = true;
  button.textContent = "Analysing…";
  $("error-slot").replaceChildren();

  // Independent requests: the report is fast, the models are not on their first call, and
  // none should wait for another.
  const code = $("code").value;
  const pending = {
    findings: postJson("/api/findings", { code }),
    semantic: postJson("/api/semantic", { code }),
    structural: postJson("/api/structural", { code }),
  };
  try {
    await Promise.all([runReport(code, pending), runFindings(pending.findings)]);
  } catch (err) {
    showError({ error: String(err), kind: "page" });
  } finally {
    button.disabled = false;
    button.textContent = "Analyse";
  }
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
  try {
    renderFindings(host, data);
  } catch (err) {
    // An answer in a shape this script cannot draw: a server and a page of different ages.
    host.replaceChildren(errorBox({ error: String(err), kind: "page" }));
  }
}

async function runReport(code, pending) {
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
  renderStages(data, pending);
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
          : data.kind === "page"
            ? "The page could not draw the answer"
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
  if (data.kind === "page") {
    box.append(el("p", "empty", "The page and the server may be of different ages. Reload with Ctrl+F5, or restart the server."));
  }
  if (data.traceback) box.append(el("pre", null, data.traceback));
  return box;
}

function showError(data) {
  $("error-slot").append(errorBox(data));
}

/* ── Findings ──────────────────────────────────────────────────────────────── */

function renderFindings(host, data) {
  // A card only where a known weakness pattern matches a line above the model's cut-off.
  if (data.cards.length) {
    const list = el("ol", "findings");
    for (const card of data.cards) list.append(findingCard(card));
    host.append(list);
  } else {
    host.append(
      el(
        "p",
        "no-pattern no-cards",
        "No known weakness pattern matches a line the model ranked above its cut-off. That is not a statement that the function is safe."
      )
    );
  }

  host.append(topLines(data));
  host.append(sourceMap(data));

  const foot = el("p", "note");
  foot.textContent = "Scores are calibrated probabilities and are low by construction. Read the order.";
  if (data.coverage.lines_beyond_cap) {
    foot.textContent += ` ${plural(data.coverage.lines_beyond_cap, "line", "lines")} past line ${data.coverage.line_cap} cannot be scored.`;
  }
  host.append(foot);
}

function findingCard(card) {
  const li = el("li", "finding");
  li.id = `finding-L${card.line}`;

  const head = el("div", "finding-head");
  head.append(el("span", "finding-rank", `#${card.rank}`));
  head.append(el("span", "finding-line", `Line ${card.line}`));
  const meter = el("span", "meter");
  meter.title = `score ${card.score.toFixed(3)}`;
  const fill = el("span", "meter-fill");
  fill.style.width = `${Math.max(4, card.score * 100)}%`;
  meter.append(fill);
  head.append(meter);
  head.append(el("span", "finding-tag has-pattern", "known pattern"));
  li.append(head);

  const snippet = el("div", "snippet");
  for (const row of card.context) {
    const line = el("div", "snippet-row");
    if (row.line === card.line) line.dataset.focus = "true";
    line.append(el("span", "ln", String(row.line)));
    line.append(el("code", "src", row.text || " "));
    snippet.append(line);
  }
  li.append(snippet);

  const block = el("div", "pattern");
  const title = el("p", "pattern-title");
  if (card.cwe) title.append(el("span", "cwe", card.cwe));
  title.append(document.createTextNode(card.label));
  block.append(title);

  // The service's sentence, shown whole: it was verified against the prediction as written.
  const why = el("p", "pattern-text");
  why.append(el("b", null, "Why  "), document.createTextNode(card.explanation));
  block.append(why);

  if (card.advice) {
    const fix = el("p", "pattern-text");
    fix.append(el("b", null, "Fix  "), document.createTextNode(card.advice.fix));
    block.append(fix);

    const pair = el("div", "before-after");
    const before = el("div");
    before.append(el("span", "pair-label", "typical"), el("pre", null, card.advice.example.before));
    const after = el("div");
    after.append(el("span", "pair-label", "safer"), el("pre", null, card.advice.example.after));
    pair.append(before, after);
    block.append(pair);
  }

  block.append(more("What the model weighted", evidence(card)));
  li.append(block);
  return li;
}

/* The fusion has two attention maps, so a line can be weighted in each. The data-flow path
   is static analysis of the function, not something the model reported. */
function evidence(card) {
  const weighted = (direction) =>
    card.attended
      .filter((attended) => attended.direction === direction)
      .map((attended) => `line ${attended.line} (${attended.weight.toFixed(2)})`)
      .join(", ") || "none recorded";
  return readout([
    ["Text branch weighted", weighted("sem_from_str")],
    ["Graph branch weighted", weighted("str_from_sem")],
    ["Tokens", card.tokens.length ? card.tokens.join(" · ") : "none"],
    [
      "Data-flow path (static)",
      card.dataflow_path.length
        ? `lines ${card.dataflow_path.join(" → ")}${card.dataflow_symbols.length ? ` · through ${card.dataflow_symbols.join(", ")}` : ""}`
        : "none found",
    ],
  ]);
}

/* The top of the model's ordering, pattern or not. The ordering is what was measured. */
function topLines(data) {
  const wrap = el("section", "elsewhere");
  wrap.append(el("h3", null, "The model's highest-ranked lines"));
  const rows = el("ul", "elsewhere-list");
  for (const entry of data.top) {
    const li = el("li");
    li.append(el("span", "ln-chip", `#${entry.rank}`));
    li.append(el("span", "ln-chip", `L${entry.line}`));
    li.append(el("code", null, entry.code));
    if (entry.has_card) {
      const link = el("a", "heat-rank", "pattern ↑");
      link.href = `#finding-L${entry.line}`;
      li.append(link);
    }
    rows.append(li);
  }
  if (!data.top.length) rows.append(el("li", "dim", "No line could be scored."));
  wrap.append(rows);
  return wrap;
}

/* The whole function with the model's ordering alongside. The highest-ranked lines carry
   their rank; every other scored line is shaded by score, faintly. */
function sourceMap(data) {
  const wrap = el("section", "source-map");
  wrap.append(el("h3", null, "Whole function"));
  const topRank = new Map(data.top.map((entry) => [entry.line, entry.rank]));
  const cardLines = new Set(data.cards.map((card) => card.line));
  const box = el("div", "axis");
  for (const entry of data.scores) {
    const row = el("div", "map-row");
    const rank = topRank.get(entry.line);
    if (rank) row.dataset.top = "true";
    row.append(el("span", "ln", String(entry.line)));
    const heat = el("span", "heat");
    if (entry.score == null) {
      heat.textContent = "—";
      heat.title = "not scored";
    } else if (rank && cardLines.has(entry.line)) {
      const a = el("a", "heat-rank", `#${rank}`);
      a.href = `#finding-L${entry.line}`;
      heat.append(a);
    } else if (rank) {
      heat.append(el("span", "heat-rank", `#${rank}`));
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
  localize: renderLocalize,
  explain: renderExplain,
  evaluate: renderEvaluate,
  serve: renderServe,
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
  // Every stage is built now; an empty "Not built yet" box would say otherwise.
  if (plannedStages().length) host.append(renderPlanned());
  refreshCurrentStage();
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

/* A slot that fills when one model has answered. A model that failed shows its real
   error here, never an empty table. */
async function whenAnswered(body, request, draw) {
  const slot = el("div");
  slot.append(el("p", "empty loading", "Ranking lines… the first run loads the model."));
  body.append(slot);
  const { data, error } = await request;
  slot.replaceChildren();
  if (error) {
    slot.append(errorBox(error));
    return;
  }
  draw(slot, data);
}

const rankMap = (answer) => new Map((answer.data?.scores || []).map((e) => [e.line, e.rank]));
const rankCell = (ranks, line) => ranks.get(line) ?? "—";

async function renderSemantic(body, _data, pending) {
  body.append(readout(accuracy("semantic", "Held-out")));
  body.append(
    el("p", "note", "Trained before comments were stripped: that figure is inflated by a formatting artefact (found 2026-10-07).")
  );
  const slot = el("div");
  body.append(slot);
  const released = rankMap(await pending.findings);
  await whenAnswered(slot, pending.semantic, (inner, s) => {
    inner.append(el("h3", "sub", "Top lines on your code"));
    inner.append(
      table(
        ["#", "line", "code", "released model's rank"],
        s.top.map((e) => [e.rank, e.line, el("code", null, e.text), rankCell(released, e.line)]),
        { numeric: [0, 1, 3] }
      )
    );
  });
}

async function renderStructural(body, _data, pending) {
  body.append(readout(accuracy("structural", "Held-out")));
  const slot = el("div");
  body.append(slot);
  const [semantic, findings] = await Promise.all([pending.semantic, pending.findings]);
  const semanticRank = rankMap(semantic);
  const released = rankMap(findings);
  await whenAnswered(slot, pending.structural, (inner, s) => {
    inner.append(el("h3", "sub", "Top lines on your code"));
    inner.append(
      table(
        ["#", "line", "code", "semantic model's rank", "released model's rank"],
        s.top.map((e) => [e.rank, e.line, el("code", null, e.text), rankCell(semanticRank, e.line), rankCell(released, e.line)]),
        { numeric: [0, 1, 3, 4] }
      )
    );
    if (s.lines_without_node.length) {
      inner.append(
        el("p", "note", `Not scored (no graph node): ${plural(s.lines_without_node.length, "line", "lines")} — ${s.lines_without_node.join(", ")}.`)
      );
    }
  });
}

async function renderFusion(body, _data, pending) {
  const slot = el("div");
  body.append(slot);
  const rows = [
    ["cross_attention", "Cross-attention"],
    ["concat", "Concatenation"],
    ["sum", "Sum"],
    ["gated", "Gated"],
  ];
  const tableRows = rows
    .filter(([name]) => RESULTS[name] && !RESULTS[name].missing)
    .map(([name, label]) => ({
      chosen: name === "cross_attention",
      cells: [label, pct(RESULTS[name].top_5), pct(RESULTS[name].top_1)],
    }));
  const screen = el("div");
  screen.append(table(["method", "top-5 hit rate", "top-1 hit rate"], tableRows, { numeric: [1, 2] }));
  screen.append(
    el("p", "note", "A screen with the semantic model frozen, on text that still had comments. Fine-tuned and held out, the methods tie; see Evaluation.")
  );
  body.append(more("Compare fusion methods (validation screen)", screen));

  const [semantic, structural] = await Promise.all([pending.semantic, pending.structural]);
  const semanticRank = rankMap(semantic);
  const structuralRank = rankMap(structural);
  await whenAnswered(slot, pending.findings, (inner, f) => {
    inner.append(el("h3", "sub", "Top lines on your code"));
    inner.append(
      table(
        ["#", "line", "code", "semantic alone", "structural alone"],
        f.top.map((e) => [e.rank, e.line, el("code", null, e.code), rankCell(semanticRank, e.line), rankCell(structuralRank, e.line)]),
        { numeric: [0, 1, 3, 4] }
      )
    );
  });
}

function renderLocalize(body, _data, pending) {
  return whenAnswered(body, pending.findings, (slot, f) => {
    slot.append(
      readout([
        ["Cut-off", f.threshold.toFixed(3)],
        ["Calibration", f.model.calibrator],
      ])
    );
    slot.append(
      table(
        ["#", "line", "score", "over cut-off", "code"],
        f.top.map((e) => [e.rank, e.line, e.score.toFixed(3), e.flagged ? "yes" : "no", el("code", null, e.code)]),
        { numeric: [0, 1, 2] }
      )
    );
    slot.append(
      el("p", "note", "A short function almost always has a line over the cut-off, vulnerable or not. Read the order, not the flag.")
    );
  });
}

function renderExplain(body, _data, pending) {
  return whenAnswered(body, pending.findings, (slot, f) => {
    if (f.cards.length) {
      slot.append(
        table(
          ["line", "pattern", "model weighted", "data flow (static)"],
          f.cards.map((card) => [
            card.line,
            `${card.label}${card.cwe ? ` · ${card.cwe}` : ""}`,
            [...new Set(card.attended.map((a) => a.line))].join(", ") || "—",
            card.dataflow_path.length ? card.dataflow_path.join(" → ") : "—",
          ]),
          { numeric: [0] }
        )
      );
    } else {
      slot.append(el("p", "empty", "No pattern matches a line over the cut-off, so no reason is given."));
    }
    const matched = new Set(f.patterns.matched);
    const chips = el("ul", "pattern-chips");
    for (const pattern of f.patterns.checked) {
      // Several patterns share a label (three kinds of buffer overflow), so the id is shown.
      chips.append(
        el("li", matched.has(pattern.id) ? "matched" : null, `${pattern.label}: ${pattern.id}${pattern.cwe ? ` · ${pattern.cwe}` : ""}`)
      );
    }
    slot.append(more(`The ${f.patterns.checked.length} patterns tested`, chips));
    if (f.coverage.explanations_rejected) {
      slot.append(
        el("p", "note note-warn", `${plural(f.coverage.explanations_rejected, "explanation", "explanations")} failed verification against the prediction and ${f.coverage.explanations_rejected === 1 ? "is" : "are"} not shown.`)
      );
    }
  });
}

function renderEvaluate(body, _data, pending) {
  return whenAnswered(body, pending.findings, (slot, f) => {
    const m = f.measured;
    if (!m) {
      slot.append(el("p", "empty", "This bundle carries no measured record."));
      return;
    }
    slot.append(
      readout([
        ["Held-out functions", m.units.toLocaleString(), null, m.partition],
        ["Flawed line in top 5", pct(m.top_5), "live"],
        ["Flawed line first", pct(m.top_1)],
        ["Top 5, reading top-down", pct(m.source_order_top_5)],
        ["Labelled lines over the cut-off", pct(m.line_recall)],
        ["Lines over the cut-off that were labelled", pct(m.line_precision)],
      ])
    );
    slot.append(
      el("p", "note", "Every one of those functions contains a vulnerability, so none of this is a rate of detecting vulnerable functions.")
    );
    const record = f.pattern_record;
    if (record) {
      slot.append(
        el("p", "note", `On validation a pattern matched a line over the cut-off in ${pct(record.vulnerable.raised / record.vulnerable.n)} of ${record.vulnerable.n} vulnerable functions and ${pct(record.not_vulnerable.raised / record.not_vulnerable.n)} of ${record.not_vulnerable.n.toLocaleString()} harmless ones (an experiment).`)
      );
    }
  });
}

function renderServe(body, _data, pending) {
  return whenAnswered(body, pending.findings, (slot, f) => {
    const c = f.coverage;
    slot.append(
      readout([
        ["Bundle", f.model.bundle],
        ["Run", f.model.run_id.slice(0, 8)],
        ["Code", f.model.git_sha],
        ["Answered in", `${f.seconds.toFixed(2)} s`, null, f.model.device],
      ])
    );
    slot.append(
      readout([
        ["Lines", c.lines],
        ["Scored", c.lines_scored],
        ["Past the line cap", c.lines_beyond_cap, c.lines_beyond_cap ? "void" : null],
        ["Code with nothing to score", c.lines_without_features, c.lines_without_features ? "void" : null],
        ["Blank, never offered", c.blank_lines],
      ])
    );
    if (!c.parse_clean) {
      slot.append(
        el("p", "note note-warn", c.first_parse_error_line
          ? `The parser had to repair this function, first at line ${c.first_parse_error_line}; the structural branch read a damaged graph.`
          : "The parser had to repair this function; the structural branch read a damaged graph.")
      );
    }
    if (!c.dataflow_available) {
      slot.append(el("p", "note note-warn", "No data-flow graph could be built, so no reason carries a data-flow path."));
    }
  });
}

boot();
