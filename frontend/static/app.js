/* VulnLoc — client.

   Two views over one paste. FINDINGS is for the developer: the lines to review first,
   ranked by the Phase 3 semantic model, with any known weakness pattern on each line and
   its usual fix. DETAILED REPORT is the pipeline inspector: every stage that ran.

   Renders what the server measured. It computes nothing about the pipeline itself: if a
   number is on screen it came from `/api/*`, which in turn came from `src/vulnloc` or a
   logged run. The only arithmetic here is layout.

   The model and the templates are two different sources and are never blended on screen.
   The model ranks a line and gives no reason; a template matches the line's text and knows
   nothing about the model. Merging them into one "because" would invent an explanation
   Phase 7 has not built.

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
  setupTabs();
  await Promise.all([loadModel(), loadPhases(), loadSamples(), loadCorpus()]);
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

let MODEL = null;

async function loadModel() {
  const model = await fetch("/api/model").then((r) => r.json());
  MODEL = model;
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

  // The two views are independent requests: the report is fast, the model is not on its
  // first call, and neither should wait for the other.
  const code = $("code").value;
  await Promise.all([runReport(code), runFindings(code)]);
  button.disabled = false;
  button.textContent = "Analyse";
}

async function runFindings(code) {
  const host = $("findings");
  host.replaceChildren(
    el(
      "p",
      "empty loading",
      "Ranking lines… The first analysis loads the model, which can take a minute while training runs on this machine."
    )
  );
  try {
    const response = await fetch("/api/findings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    });
    const data = await response.json();
    host.replaceChildren();
    if (!response.ok) {
      host.append(errorBox(data));
      return;
    }
    renderFindings(host, data);
  } catch (err) {
    host.replaceChildren(errorBox({ error: String(err), kind: "network" }));
  }
}

async function runReport(code) {
  const body = {
    code,
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
  }
}

function errorBox(data) {
  const box = el("div", "error");
  const title =
    data.kind === "input"
      ? "Nothing to analyse"
      : data.kind === "model"
        ? "The model raised"
        : "The pipeline raised";
  box.append(el("h2", null, title));
  box.append(el("p", "empty", data.error));
  if (data.traceback) box.append(el("pre", null, data.traceback));
  return box;
}

function showError(data) {
  $("error-slot").append(errorBox(data));
}

/* ── Findings ──────────────────────────────────────────────────────────────── */

const pct = (x) => `${Math.round(x * 100)}%`;

function renderFindings(host, data) {
  const m = data.measured;
  const run = m.run || {};

  // The one paragraph a developer needs before trusting a ranking. Numbers come from the
  // seed report on disk; nothing here is typed in by hand.
  const trust = el("div", "trust");
  const trustHead = el("p", "trust-lead");
  trustHead.append(el("b", null, "Lines to review first, ranked by the model. "));
  trustHead.append(
    document.createTextNode(
      run.top_1 != null
        ? `On held-out ${m.eval_split} functions the real flawed line was its first pick ${pct(run.top_1)} of the time and in its top ${data.top_k} ${pct(run.top_5)} of the time.`
        : "Its held-out accuracy could not be read from the seed report."
    )
  );
  trust.append(trustHead);
  const caveats = el("ul", "trust-caveats");
  caveats.append(
    el("li", null, "It assumes the function has a flaw and ranks where. It cannot tell you code is safe."),
    el("li", null, "Measured on Big-Vul (mostly Chrome and Linux). Your code is outside that data."),
    el("li", null, "Fix advice comes from pattern templates, not from the model's reasoning.")
  );
  trust.append(caveats);
  host.append(trust);

  const list = el("ol", "findings");
  for (const finding of data.top) {
    list.append(findingCard(finding, data));
  }
  host.append(list);
  if (data.skipped_punctuation_lines.length) {
    host.append(
      el(
        "p",
        "note",
        `Skipped ${plural(data.skipped_punctuation_lines.length, "line", "lines")} the model ranked this high that hold only punctuation (${data.skipped_punctuation_lines.map((l) => `L${l}`).join(", ")}), which is why the ranks above can jump.`
      )
    );
  }

  if (data.patterns_elsewhere.length) {
    const other = el("section", "elsewhere");
    other.append(el("h3", null, "Patterns on lines the model ranked lower"));
    other.append(
      el(
        "p",
        "note",
        "A template matched these lines, but the model did not put them in its top five. Worth a look; not a finding."
      )
    );
    const rows = el("ul", "elsewhere-list");
    for (const item of data.patterns_elsewhere) {
      for (const p of item.patterns) {
        const li = el("li");
        li.append(el("span", "ln-chip", `L${item.line}`));
        li.append(el("span", "cwe", p.cwe));
        li.append(el("span", null, `${p.title} — `));
        li.append(el("code", null, item.text));
        li.append(el("span", "dim", item.rank ? ` · model rank ${item.rank}` : " · not scored"));
        rows.append(li);
      }
    }
    other.append(rows);
    host.append(other);
  }

  host.append(sourceMap(data));

  const foot = el("p", "note");
  foot.textContent =
    `Scores from ${data.model.name}, run ${data.model.run_id.slice(0, 8)}… (seed ${run.seed}), on ${data.model.device}. ` +
    `Scores are shown for ordering only: the model never saw a safe function, so they are not calibrated probabilities. ` +
    `About ${m.label_noise} of the line labels it learned from are wrong (Memory.md §1.6).`;
  if (data.beyond_line_cap) {
    foot.textContent += ` ${plural(data.beyond_line_cap, "line", "lines")} past line 512 cannot be scored.`;
  }
  host.append(foot);
}

function findingCard(finding, data) {
  const li = el("li", "finding");
  li.id = `finding-L${finding.line}`;

  const head = el("div", "finding-head");
  head.append(el("span", "finding-rank", `#${finding.rank}`));
  head.append(el("span", "finding-line", `Line ${finding.line}`));
  const meter = el("span", "meter");
  meter.title = `score ${finding.score.toFixed(3)} — for ordering, not a probability`;
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

  if (!finding.patterns.length) {
    li.append(
      el(
        "p",
        "no-pattern",
        "No template matches this line. The model ranks it highly but gives no reason, so read it with the lines around it: what reaches it, and what it writes."
      )
    );
    return li;
  }

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
   and link to their card; every other scored line is shaded by score, faintly, so the eye
   reads rank first and gradient second. */
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
      heat.title = "no tokens on this line, so it is not scored";
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
  if (stage.state === "built-unrun")
    head.append(el("span", "badge badge-signal", "built · not yet run, gated"));
  if (stage.state === "running")
    head.append(el("span", "badge badge-signal", "running · no result yet"));
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
  // The fallback used to say "nothing is built" for every stage without a renderer, which
  // was false for the trained structural branch. What is absent differs by state.
  const absent = {
    planned: `Nothing is shown here because nothing is built. When Phase ${stage.phase} lands, this panel is where its output goes.`,
    trained: "Built and measured, but this tool does not run it on your paste; only the semantic model scores lines in Findings.",
    "built-unrun": "Built, not yet run. No output exists to show.",
    running: "Being run now. No output exists to show until a result is logged.",
  };
  list.append(el("li", null, absent[stage.state] || "No output to show for this stage."));
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

function renderSemantic(body) {
  const m = MODEL?.measured;
  const three = m?.three_seed;
  if (!three) {
    body.append(el("p", "note note-warn", "The Phase 3 seed report could not be read, so no measured number is shown."));
    return;
  }
  body.append(
    readout([
      ["Encoder", "CodeBERT", "live"],
      ["Seeds", String(m.n_seeds), "live"],
      [`Top-5 (${m.eval_split})`, three.top_5_mean.toFixed(4), "live", `± ${three.top_5_std.toFixed(4)}`],
      [`Top-1 (${m.eval_split})`, three.top_1_mean.toFixed(4), "live"],
      ["Scores shown", "Findings tab", "signal", `seed ${m.run.seed}`],
    ])
  );
  body.append(
    el(
      "p",
      "note",
      `Findings runs this model (run ${m.run.run_id.slice(0, 8)}…, seed ${m.run.seed} — the default seed, not the best of three, so the choice is not made on validation data) on your paste. Its function-level F1 of 1.000 is degenerate — every evaluation unit is vulnerable — and is never shown. Run 1843412a… remains void (ADR-006).`
    )
  );
}

boot();
