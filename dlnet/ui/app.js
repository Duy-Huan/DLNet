/* DLNet web UI (optional; the CLI is the main tool). Plain JavaScript, no build step, no network
 * access beyond this machine. Every visible word comes from i18n/<lang>.json through t(); the two
 * language files must have the same keys (tests/test_ui.py). Data is put on the page with
 * textContent only (h()), never as HTML. */
"use strict";

const TOKEN = document.querySelector('meta[name="dlnet-token"]').content;
const VERSION = document.querySelector('meta[name="dlnet-version"]').content;
const LANGS = ["en", "vi"];
const BUDGETS = ["5m", "10m", "1h", "400ep"];
const DATA_EXT = [".npz", ".npy", ".mat", ".csv"];
const TEXT_EXT = [".md", ".py", ".yaml", ".yml", ".json", ".txt", ".tex", ".log", ".csv", ".ipynb"];
const S = { dict: {}, lang: "en", state: null, timer: null, view: null };

/* ------------------------------------------------------------------ helpers */

function store(key, value) {
  try {
    if (value === undefined) return localStorage.getItem("dlnet." + key);
    localStorage.setItem("dlnet." + key, value);
  } catch (e) { /* private mode: settings are not remembered */ }
  return null;
}

function t(key, params) {
  let s = (S.dict[S.lang] || {})[key];
  if (s === undefined) s = (S.dict.en || {})[key];
  if (s === undefined) return key;
  if (params) for (const k of Object.keys(params)) s = s.split("{" + k + "}").join(String(params[k]));
  return s;
}

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else if (k === "value") el.value = v;
    else if (k === "checked") el.checked = !!v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

async function api(method, path, body) {
  const opt = { method, headers: { "X-DLNet-Token": TOKEN } };
  if (body !== undefined) {
    opt.headers["Content-Type"] = "application/json";
    opt.body = JSON.stringify(body);
  }
  const r = await fetch(path, opt);
  let data = {};
  try { data = await r.json(); } catch (e) { data = {}; }
  if (!r.ok) throw new Error(data.error || r.status + " " + r.statusText);
  return data;
}

const tool = async (name, args) => (await api("POST", "/api/tool/" + name, args || {})).result;
const fileUrl = (rel, download) => "/files/" + rel.split("/").map(encodeURIComponent).join("/") +
  "?t=" + encodeURIComponent(TOKEN) + (download ? "&download=1" : "");

function toast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove("show"), 2600);
}

async function copy(text) {
  try { await navigator.clipboard.writeText(text); toast(t("common.copied")); }
  catch (e) { toast(t("common.copy_failed")); }
}

const fmt = (x, d = 4) => (x === null || x === undefined || Number.isNaN(x) ? "—" : Number(x).toFixed(d));
const ext = (p) => { const i = p.lastIndexOf("."); return i < 0 ? "" : p.slice(i).toLowerCase(); };
function bytes(n) {
  if (n === undefined || n === null) return "";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return (i ? n.toFixed(1) : n) + " " + u[i];
}
function duration(s) {
  if (s === null || s === undefined) return "?";
  if (s < 90) return t("unit.seconds", { n: Math.round(s) });
  if (s < 5400) return t("unit.minutes", { n: Math.round(s / 60) });
  return t("unit.hours", { n: (s / 3600).toFixed(1) });
}
const statusText = (s) => t("status." + (s || "empty"));
const statusEl = (s) => h("span", { class: "status s-" + (s || "empty") }, statusText(s));

/** argv of a job -> the command a user would type (`dlnet design ...`), one option per line. */
function commandText(argv) {
  if (!argv || !argv.length) return "";
  let i = argv.indexOf("design");
  if (i < 0) i = argv.indexOf("plan");
  const parts = ["dlnet", ...argv.slice(Math.max(i, 0))];
  const quote = (a) => (/^[\w./:=+,@%-]+$/.test(a) ? a : "'" + a.replace(/'/g, "'\\''") + "'");
  const lines = [];
  let cur = [];
  for (const p of parts) {
    if (p.startsWith("--") && cur.length > 1) { lines.push(cur.join(" ")); cur = []; }
    cur.push(quote(p));
  }
  lines.push(cur.join(" "));
  return lines.join(" \\\n  ");
}

function every(ms, fn) {
  clearInterval(S.timer);
  S.timer = setInterval(() => { if (!document.hidden) fn(); }, ms);
}

/* ------------------------------------------------------------------ shell */

function applyChrome() {
  document.documentElement.lang = S.lang;
  document.title = "DLNet " + VERSION;
  for (const el of document.querySelectorAll("[data-i18n]")) el.textContent = t(el.dataset.i18n);
  document.getElementById("lang").textContent = S.lang === "en" ? "VI" : "EN";
  document.getElementById("lang").setAttribute("aria-label", t("chrome.language"));
  const th = store("theme") || "auto";
  document.getElementById("theme").textContent = t("theme." + th);
  document.getElementById("theme").setAttribute("aria-label", t("chrome.theme"));
  const ws = S.state ? S.state.workspace : "";
  const link = document.getElementById("ws-link");
  link.textContent = ws ? ws + " ▾" : t("nav.workspace");
  link.title = t("nav.workspace");
}

function setTheme(next) {
  store("theme", next);
  if (next === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", next);
  applyChrome();
}

async function loadState() {
  S.state = await api("GET", "/api/state");
  applyChrome();
}

const ROUTES = {
  workspace: viewWorkspace, runs: viewRuns, files: viewFiles, new: viewNew, run: viewRun, ai: viewAI,
};

async function route() {
  clearInterval(S.timer);
  const raw = location.hash.replace(/^#\/?/, "") || "runs";
  const [path, query] = raw.split("?");
  const [name, ...rest] = path.split("/");
  const params = Object.fromEntries(new URLSearchParams(query || ""));
  const fn = ROUTES[name] || viewRuns;
  for (const a of document.querySelectorAll("[data-nav]")) {
    a.classList.toggle("on", a.dataset.nav === name || (name === "run" && a.dataset.nav === "runs") ||
      (name === "new" && a.dataset.nav === "runs"));
  }
  const main = document.getElementById("view");
  main.className = name === "new" ? "narrow" : "";
  S.view = name;
  try {
    await fn(main, decodeURIComponent(rest.join("/")), params);
  } catch (e) {
    main.replaceChildren(h("h1", {}, t("error.title")), h("p", { class: "bad" }, e.message));
  }
}

/* ------------------------------------------------------------------ 0. workspace */

async function viewWorkspace(main) {
  let browsePath = S.state.workspace;
  const list = h("div", { class: "list" });
  const pathInput = h("input", { type: "text", class: "mono", value: browsePath, "aria-label": t("ws.folder") });
  const here = h("p", { class: "hint" });
  const openBtn = h("button", { class: "primary", type: "button", onclick: () => openWs(browsePath, false) }, t("ws.open_this"));
  const newName = h("input", { type: "text", class: "mono", placeholder: t("ws.new_placeholder"), "aria-label": t("ws.new_name") });

  async function openWs(path, create) {
    try {
      await api("POST", "/api/workspace", { path, create });
      await loadState();
      toast(t("ws.opened", { path: S.state.workspace }));
      location.hash = "#/runs";
    } catch (e) { toast(e.message); }
  }
  async function browse(p) {
    try {
      const r = await api("GET", "/api/browse?path=" + encodeURIComponent(p || ""));
      browsePath = r.path;
      pathInput.value = r.path;
      here.textContent = r.workspace ? t("ws.is_workspace") : t("ws.not_workspace");
      list.replaceChildren(
        r.parent ? h("button", { class: "item", type: "button", onclick: () => browse(r.parent) },
          h("span", { class: "mono" }, "../"), h("span", { class: "meta" }, t("ws.up"))) : null,
        r.dirs.length ? r.dirs.map((d) => h("button", { class: "item", type: "button", onclick: () => browse(r.path + "/" + d.name) },
          h("span", { class: "mono" }, d.name + "/"), d.workspace ? h("span", { class: "meta" }, t("ws.badge")) : null))
          : h("p", { class: "hint" }, t("ws.no_subfolders")));
    } catch (e) { toast(e.message); }
  }
  const recent = S.state.recent || [];
  main.replaceChildren(
    h("h1", {}, t("ws.title")),
    h("p", { class: "lead" }, t("ws.lead")),
    h("section", {}, h("h2", {}, t("ws.recent")),
      recent.length ? h("div", { class: "list" }, recent.map((p) => h("button", { class: "item" + (p === S.state.workspace ? " on" : ""), type: "button", onclick: () => openWs(p, false) },
        h("span", { class: "mono" }, p), p === S.state.workspace ? h("span", { class: "meta" }, t("ws.current")) : null)))
        : h("p", { class: "hint" }, t("ws.no_recent"))),
    h("section", {}, h("h2", {}, t("ws.browse")),
      h("div", { class: "row" }, pathInput, h("button", { type: "button", onclick: () => browse(pathInput.value) }, t("ws.go")), openBtn),
      here, h("div", { class: "card", style: "margin-top:12px; padding:8px" }, list)),
    h("section", {}, h("h2", {}, t("ws.create")),
      h("div", { class: "row" }, newName, h("button", { type: "button", onclick: () => {
        const n = newName.value.trim();
        if (!/^[\w.-]+$/.test(n)) { toast(t("ws.bad_name")); return; }
        openWs(browsePath + "/" + n, true);
      } }, t("ws.create_btn"))),
      h("p", { class: "hint" }, t("ws.create_hint"))));
  browse(browsePath);
}

/* ------------------------------------------------------------------ 1. runs */

async function viewRuns(main) {
  const body = h("tbody");
  const lead = h("p", { class: "lead" });
  main.replaceChildren(
    h("div", { class: "row", style: "justify-content:space-between" }, h("h1", {}, t("runs.title")),
      h("a", { class: "btn primary", href: "#/new" }, t("runs.new"))),
    lead,
    h("div", { class: "scroll" }, h("table", { style: "min-width:720px" },
      h("thead", {}, h("tr", {}, h("th", {}, t("runs.col.run")), h("th", {}, t("runs.col.top1")),
        h("th", { class: "r" }, t("runs.col.val")), h("th", { class: "r" }, t("runs.col.test")), h("th", { class: "r" }, t("runs.col.status")))),
      body)),
    h("p", { class: "hint mono", style: "margin-top:32px" }, "dlnet ui --workspace " + S.state.workspace));
  async function load() {
    const { runs } = await api("GET", "/api/runs");
    lead.textContent = t("runs.lead", { n: runs.length });
    if (!runs.length) {
      body.replaceChildren(h("tr", {}, h("td", { colspan: 5 }, h("div", { class: "empty" }, t("runs.empty")))));
      return false;
    }
    body.replaceChildren(...runs.map((r) => {
      const sub = [r.task, r.budget, r.papers ? t("runs.papers", { n: r.papers }) : null].filter(Boolean).join(" · ");
      const busy = r.status === "running" || r.status === "queued";
      return h("tr", {},
        h("td", {}, h("a", { href: "#/run/" + encodeURIComponent(r.name), style: "font-weight:500; color:var(--text)" }, r.name),
          h("span", { class: "sub" }, sub || r.path)),
        h("td", {}, busy ? h("span", { class: "muted mono", style: "font-size:13px" }, (r.progress || "").slice(0, 80)) : (r.top1 || "—")),
        h("td", { class: "r num" }, fmt(r.val), r.val_std ? h("span", { class: "pm" }, " ±" + fmt(r.val_std)) : null),
        h("td", { class: "r num" }, fmt(r.test)),
        h("td", { class: "r" }, statusEl(r.status)));
    }));
    return runs.some((r) => r.status === "running" || r.status === "queued");
  }
  if (await load()) every(3000, async () => { if (S.view === "runs" && !(await load())) clearInterval(S.timer); });
}

/* ------------------------------------------------------------------ 4. run detail */

function verdictText(v) {
  const k = { better: "better", worse: "worse", "no significant difference": "same", "not tested": "untested" }[v];
  return k ? t("verdict." + k) : "—";
}

async function viewRun(main, name) {
  async function load() {
    const d = await api("GET", "/api/run/" + encodeURIComponent(name));
    const b = d.board;
    const run = d.run || {};
    const kids = [h("h1", {}, name)];
    if (b) {
      const top = b.top_k[0];
      kids.push(h("p", { class: "lead", style: "margin-bottom:8px" },
        [b.task, t("run.metric", { m: b.metric }), (b.budget || {}).budget ? t("run.budget", { b: b.budget.budget }) : null].filter(Boolean).join(" · ")));
      const v = b.verdict || {};
      kids.push(h("p", {}, v.beats_reference ? t("run.beats") : t("run.not_beats")));
      // Top-K + reference + papers in one table
      const rows = b.top_k.map((r) => h("tr", {},
        h("td", { class: "muted" }, r.rank), h("td", {}, h("span", { style: r.rank === 1 ? "font-weight:500" : "" }, r.name)),
        h("td", { class: "r num" }, (r.params || 0).toLocaleString()),
        h("td", { class: "r num" }, fmt(r.val_mean), h("span", { class: "pm" }, r.val_std ? " ±" + fmt(r.val_std) : "")),
        h("td", { class: "r num" }, fmt((r.test || {}).score)), h("td", {}, "")));
      const ref = (b.reference || []).slice().sort((a, c) => (c.val || 0) - (a.val || 0))[0];
      if (ref) rows.push(h("tr", { class: "sep" }, h("td", { class: "muted" }, "R"),
        h("td", {}, ref.name, h("span", { class: "sub" }, t("run.classical"))), h("td", { class: "r num" }, "—"),
        h("td", { class: "r num" }, fmt(ref.val)), h("td", { class: "r num" }, fmt((ref.test || {}).score)), h("td", {}, "")));
      const papers = (b.papers || []).filter((p) => p.common);
      papers.forEach((p, i) => {
        const vs = p.vs_top1 || {};
        rows.push(h("tr", { class: i === 0 ? "sep" : "" }, h("td", { class: "muted" }, "P"),
          h("td", {}, p.paper || p.name, h("span", { class: "sub" }, t("run.paper_level", { l: p.level || "?" }))),
          h("td", { class: "r num" }, (p.params || 0).toLocaleString()),
          h("td", { class: "r num" }, fmt(p.common.val_mean), h("span", { class: "pm" }, p.common.val_std ? " ±" + fmt(p.common.val_std) : "")),
          h("td", { class: "r num" }, fmt((p.common.test || {}).score)),
          h("td", { class: "num", style: "font-size:13px" }, vs.diff === null || vs.diff === undefined ? "—"
            : (vs.diff >= 0 ? "+" : "") + fmt(vs.diff) + (vs.p_holm === null || vs.p_holm === undefined ? "" : " · p " + fmt(vs.p_holm, 3)),
            h("span", { class: "sub" }, verdictText(vs.verdict)))));
      });
      kids.push(h("section", {}, h("h2", {}, t("run.results")), h("div", { class: "scroll" }, h("table", { style: "min-width:700px" },
        h("thead", {}, h("tr", {}, h("th", {}, "#"), h("th", {}, t("run.col.network")), h("th", { class: "r" }, t("run.col.params")),
          h("th", { class: "r" }, t("run.col.val", { n: top.n_runs || 1 })), h("th", { class: "r" }, t("run.col.test")), h("th", {}, t("run.col.vs")))),
        h("tbody", {}, rows))),
        papers.length ? h("p", { class: "hint" }, t("run.papers_note", { n: top.n_runs || 1 })) : null,
        (b.papers_failed || []).length ? h("p", { class: "hint warn" }, t("run.papers_failed", { n: b.papers_failed.length }) + " " +
          b.papers_failed.map((f) => f.card.split("/").pop() + ": " + f.error).join("; ")) : null));
      // diagram + files
      const files = (d.files || []).filter((f) => f.type === "file");
      const svg = files.find((f) => f.path.endsWith("/architecture.svg") && f.path.includes("/" + top.name + "/"));
      kids.push(h("section", { class: "split" },
        h("div", { class: "grow" }, h("h2", {}, top.name),
          svg ? h("div", { class: "diagram" }, h("img", { src: fileUrl(svg.path), alt: t("run.diagram_alt", { name: top.name }) }))
            : h("div", { class: "empty" }, t("run.no_diagram"))),
        h("div", { class: "side" }, h("h2", {}, t("run.files")), h("div", { class: "list" },
          files.filter((f) => /\/(leaderboard\.md|results\.tex|baselines\.ipynb)$/.test(f.path) || /\/top\d+_[^/]+\/(model\.py|report\.md|train\.py)$/.test(f.path))
            .slice(0, 14).map((f) => h("a", { class: "item", href: fileUrl(f.path), target: "_blank", rel: "noopener" },
              h("span", { class: "mono", title: f.path }, f.path.split("/").slice(2).join("/")), h("span", { class: "meta" }, bytes(f.bytes))))),
          h("a", { class: "btn", style: "margin-top:12px", href: "#", onclick: async (e) => {
            e.preventDefault();
            try { const z = await tool("dlnet_archive", { path: run.path }); location.href = fileUrl(z.path || z.zip, true); }
            catch (err) { toast(err.message); }
          } }, t("run.download_zip")))));
    } else {
      kids.push(h("p", { class: "lead" }, statusText(run.status)));
    }
    // command
    const cmd = commandText(run.command);
    kids.push(h("section", {}, h("h2", {}, t("run.command")),
      cmd ? h("div", { class: "row", style: "align-items:flex-start" }, h("pre", { class: "code", style: "flex:1 1 420px" }, cmd),
        h("button", { type: "button", onclick: () => copy(cmd.replace(/ \\\n {2}/g, " ")) }, t("common.copy")))
        : h("p", { class: "hint" }, t("run.no_command"))));
    if (d.latex) {
      kids.push(h("section", {}, h("h2", {}, t("run.latex")),
        h("div", { class: "row", style: "margin-bottom:10px" }, h("button", { type: "button", onclick: () => copy(d.latex) }, t("common.copy")),
          h("a", { class: "btn", href: fileUrl(run.path + "/results.tex", true) }, t("common.download"))),
        h("pre", { class: "code" }, d.latex)));
    }
    if (d.log && d.log.length && run.status !== "done" && run.status !== "succeeded") {
      kids.push(h("section", {}, h("h2", {}, t("run.log")), h("pre", { class: "code" }, d.log.join("\n")),
        run.status === "running" || run.status === "queued" ? h("button", { type: "button", style: "margin-top:10px", onclick: async () => {
          try { await tool("dlnet_cancel", { job_id: run.job_id }); toast(t("run.cancelled")); load(); } catch (e) { toast(e.message); }
        } }, t("run.cancel")) : null));
    }
    main.replaceChildren(h("p", {}, h("a", { href: "#/runs" }, "← " + t("nav.runs"))), ...kids);
    return run.status === "running" || run.status === "queued";
  }
  if (await load()) every(4000, async () => { if (S.view === "run" && !(await load())) clearInterval(S.timer); });
}

/* ------------------------------------------------------------------ 2. files */

async function viewFiles(main) {
  const tree = h("div", { class: "list" });
  const preview = h("section", { class: "grow card", style: "margin-top:0" });
  let selected = null;
  const drop = h("div", { class: "drop" }, t("files.drop"));
  const picker = h("input", { type: "file", multiple: true, style: "display:none", onchange: () => upload(picker.files) });
  drop.append(" ", h("button", { type: "button", class: "chip", onclick: () => picker.click() }, t("files.choose")), picker);
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); upload(e.dataTransfer.files); });

  async function upload(files) {
    for (const f of files) {
      const dest = (/\.ya?ml$/i.test(f.name) ? "papers/" : "data/") + f.name.replace(/[^\w.-]/g, "_");
      toast(t("files.uploading", { name: f.name }));
      const r = await fetch(fileUrl(dest).split("?")[0], { method: "PUT", body: f, headers: { "X-DLNet-Token": TOKEN } });
      if (!r.ok) { toast(((await r.json().catch(() => ({}))).error) || r.statusText); return; }
    }
    toast(t("files.uploaded", { n: files.length }));
    load();
  }
  async function show(entry) {
    selected = entry.path;
    for (const b of tree.querySelectorAll(".item")) b.classList.toggle("on", b.dataset.path === selected);
    const kids = [h("div", { class: "mono", style: "font-weight:500; overflow-wrap:anywhere" }, entry.path),
      h("div", { class: "hint", style: "margin:4px 0 18px" }, bytes(entry.bytes) + " · " + entry.modified)];
    const e = ext(entry.path);
    const actions = h("div", { class: "row", style: "margin-top:18px" });
    if (DATA_EXT.includes(e) && entry.path.startsWith("data/")) {
      const info = h("div", { class: "grid3" }, h("div", { class: "hint" }, t("files.detecting")));
      kids.push(info);
      actions.append(h("a", { class: "btn primary", href: "#/new?data=" + encodeURIComponent(entry.path) }, t("files.use_input")));
      tool("dlnet_profile", { data: entry.path }).then((p) => {
        if (selected !== entry.path) return;
        const pr = p.profile || {};
        info.replaceChildren(
          kv(t("files.kind"), t("kind." + (p.plugin || "unknown"))),
          kv(t("files.task"), pr.task || "—"),
          kv(t("files.shape"), pr.shape ? pr.shape.join(" × ") : (pr.n_features ? t("files.features", { n: pr.n_features }) : "—")),
          kv(t("files.samples"), pr.n_train ? t("files.train_samples", { n: pr.n_train }) : "—"),
          kv(t("files.classes"), pr.n_classes || "—"));
      }).catch((err) => { if (selected === entry.path) info.replaceChildren(h("p", { class: "hint warn" }, t("files.not_data") + " " + err.message)); });
    }
    if (TEXT_EXT.includes(e) && entry.bytes < 2_000_000) {
      const pre = h("pre", { class: "code", style: "max-height:420px; overflow:auto" }, "…");
      kids.push(h("div", { style: "margin-top:16px" }, pre));
      tool("dlnet_read_file", { path: entry.path, limit: 6000 }).then((r) => { pre.textContent = r.text + (r.more ? "\n…" : ""); })
        .catch((err) => { pre.textContent = err.message; });
    }
    actions.append(h("a", { class: "btn", href: fileUrl(entry.path), target: "_blank", rel: "noopener" }, t("files.open_raw")),
      h("a", { class: "btn", href: fileUrl(entry.path, true) }, t("common.download")));
    kids.push(actions);
    preview.replaceChildren(...kids);
  }
  async function load() {
    const r = await tool("dlnet_files", { depth: 3, limit: 600 });
    const entries = r.entries.filter((e) => !e.path.startsWith("jobs/"));
    let last = null;
    const items = [];
    for (const e of entries) {
      const folder = e.path.includes("/") ? e.path.slice(0, e.path.lastIndexOf("/") + 1) : "./";
      if (folder !== last) { items.push(h("div", { class: "mono hint", style: "padding:14px 12px 4px; overflow-wrap:anywhere" }, folder)); last = folder; }
      if (e.type === "dir") continue;
      items.push(h("button", { class: "item", type: "button", "data-path": e.path, onclick: () => show(e) },
        h("span", { class: "mono", title: e.path }, e.path.slice(folder === "./" ? 0 : folder.length)), h("span", { class: "meta" }, bytes(e.bytes))));
    }
    tree.replaceChildren(...(items.length ? items : [h("p", { class: "hint" }, t("files.empty"))]));
    if (!selected) preview.replaceChildren(h("p", { class: "hint" }, t("files.pick")));
  }
  main.replaceChildren(
    h("h1", {}, t("files.title")),
    h("p", { class: "lead" }, t("files.lead")),
    drop,
    h("div", { class: "split", style: "margin-top:24px" }, h("nav", { class: "side", "aria-label": t("files.title") }, tree), preview));
  await load();
}

function kv(k, v) { return h("div", { class: "kv" }, h("div", { class: "k" }, k), h("div", { class: "v" }, v)); }

/* ------------------------------------------------------------------ 3. new run (+ plan) */

async function viewNew(main, _rest, params) {
  const r = await tool("dlnet_files", { depth: 4, limit: 1000 });
  const files = r.entries.filter((e) => e.type === "file");
  const dataFiles = files.filter((e) => e.path.startsWith("data/") && DATA_EXT.includes(ext(e.path)));
  const cards = files.filter((e) => /\.ya?ml$/i.test(e.path) && !e.path.startsWith("runs/"));
  const st = { data: params.data || (dataFiles[0] || {}).path || "", gt: "", name: "", budget: "10m", topk: 3,
    papers: new Set(), recipe: false, plugin: null };
  if (!dataFiles.length) {
    main.replaceChildren(h("h1", {}, t("new.title")), h("div", { class: "empty" }, t("new.no_data"), " ",
      h("a", { href: "#/files" }, t("new.go_files"))));
    return;
  }
  const info = h("div", { class: "grid3 sunken" });
  const gtWrap = h("div", { class: "field" });
  const cmdPre = h("pre", { class: "code" });
  const planBox = h("div");
  const budgets = h("div", { class: "row" });
  const custom = h("input", { type: "text", class: "mono", style: "width:110px", placeholder: "2h", "aria-label": t("new.custom_budget"),
    oninput: () => { if (custom.value.trim()) { st.budget = custom.value.trim(); drawBudgets(); update(); } } });

  function args() {
    const a = { data: st.data, budget: st.budget, topk: Number(st.topk) || 3 };
    if (st.gt) a.gt = st.gt;
    if (st.name) a.name = st.name;
    if (st.papers.size) { a.papers = [...st.papers]; if (st.recipe) a.paper_recipe = true; }
    return a;
  }
  function command() {
    const a = args();
    const out = "runs/" + (a.name || "<name>");
    const argv = ["dlnet", "design", "--data", a.data];
    if (a.gt) argv.push("--gt", a.gt);
    argv.push("--budget", a.budget, "--topk", String(a.topk));
    if (a.papers) { argv.push("--papers", ...a.papers); if (a.paper_recipe) argv.push("--paper-recipe"); }
    argv.push("--out", out);
    return commandText(argv);
  }
  function update() { cmdPre.textContent = command(); }
  function drawBudgets() {
    budgets.replaceChildren(...BUDGETS.map((b) => h("button", { type: "button", class: "choice" + (b === st.budget ? " on" : ""),
      onclick: () => { st.budget = b; custom.value = ""; drawBudgets(); update(); } }, b)), custom);
  }
  async function detect() {
    info.replaceChildren(h("div", { class: "hint" }, t("files.detecting")));
    gtWrap.replaceChildren();
    try {
      const p = await tool("dlnet_profile", st.gt ? { data: st.data, gt: st.gt } : { data: st.data });
      const pr = p.profile || {};
      st.plugin = p.plugin;
      info.replaceChildren(kv(t("files.kind"), t("kind." + (p.plugin || "unknown"))),
        kv(t("files.shape"), pr.shape ? pr.shape.join(" × ") : (pr.n_features ? t("files.features", { n: pr.n_features }) : "—")),
        kv(t("files.classes"), pr.n_classes || "—"));
    } catch (e) {
      info.replaceChildren(h("p", { class: "hint warn" }, e.message));
    }
    if (ext(st.data) === ".mat" || st.plugin === "hsi") drawGt();
  }
  function drawGt() {
    const others = dataFiles.filter((f) => f.path !== st.data && DATA_EXT.includes(ext(f.path)));
    if (!st.gt) {
      const guess = others.find((f) => /gt|ground/i.test(f.path) && f.path.split("/").slice(0, -1).join("/") === st.data.split("/").slice(0, -1).join("/"));
      if (guess) { st.gt = guess.path; update(); }
    }
    gtWrap.replaceChildren(h("span", {}, t("new.gt")), h("select", { onchange: (e) => { st.gt = e.target.value; update(); } },
      h("option", { value: "" }, t("new.gt_none")), others.map((f) => h("option", { value: f.path, selected: f.path === st.gt ? "selected" : null }, f.path))),
      h("p", { class: "hint" }, t("new.gt_hint")));
  }
  async function estimate(btn) {
    btn.disabled = true;
    planBox.replaceChildren(h("p", { class: "hint" }, t("new.estimating")));
    try {
      let job = await tool("dlnet_plan", { ...args(), name: undefined });
      while (job.ok === null || job.ok === undefined) {
        if (S.view !== "new") return;
        planBox.replaceChildren(h("p", { class: "hint" }, t("new.estimating"), " ", h("span", { class: "mono" }, ((job.log_tail || []).slice(-1)[0] || "").slice(0, 90))));
        await new Promise((res) => setTimeout(res, 2000));
        job = await tool("dlnet_job", { job_id: job.job_id, log_lines: 2 });
      }
      if (!job.ok) throw new Error((job.error || "") + " " + (job.log_tail || []).slice(-3).join(" "));
      planBox.replaceChildren(renderPlan(job.plan));
    } catch (e) {
      planBox.replaceChildren(h("p", { class: "bad" }, t("new.plan_failed") + " " + e.message));
    } finally { btn.disabled = false; }
  }
  async function start(btn) {
    btn.disabled = true;
    try {
      const job = await tool("dlnet_design", args());
      toast(t("new.started"));
      const name = (job.out || "").split("/").pop();
      location.hash = name ? "#/run/" + encodeURIComponent(name) : "#/runs";
    } catch (e) { toast(e.message); btn.disabled = false; }
  }
  const estBtn = h("button", { type: "button", onclick: (e) => estimate(e.currentTarget) }, t("new.estimate"));
  const runBtn = h("button", { type: "button", class: "primary", onclick: (e) => start(e.currentTarget) }, t("new.run"));
  main.replaceChildren(
    h("h1", {}, t("new.title")),
    h("p", { class: "lead" }, t("new.lead")),
    h("section", { style: "margin-top:0" }, h("h2", {}, t("new.input")),
      h("div", { class: "field" }, h("select", { "aria-label": t("new.input"), onchange: (e) => { st.data = e.target.value; st.gt = ""; update(); detect(); } },
        dataFiles.map((f) => h("option", { value: f.path, selected: f.path === st.data ? "selected" : null }, f.path + "  (" + bytes(f.bytes) + ")")))),
      info, gtWrap),
    h("section", {}, h("h2", {}, t("new.output")),
      h("div", { class: "row" }, h("span", { class: "mono muted" }, "runs/"), h("input", { type: "text", class: "mono", placeholder: t("new.name_placeholder"), "aria-label": t("new.output"),
        oninput: (e) => { st.name = e.target.value.trim(); update(); } })),
      h("p", { class: "hint" }, t("new.output_hint"))),
    h("section", {}, h("h2", {}, t("new.budget")), budgets, h("p", { class: "hint" }, t("new.budget_hint")),
      h("label", { class: "check", style: "margin-top:8px" }, t("new.topk"), h("input", { type: "number", min: 1, max: 10, value: st.topk, style: "width:72px",
        oninput: (e) => { st.topk = e.target.value; update(); } }))),
    h("section", {}, h("h2", {}, t("new.papers")),
      cards.length ? h("div", {}, cards.map((c) => h("label", { class: "check", style: "display:flex" },
        h("input", { type: "checkbox", onchange: (e) => { if (e.target.checked) st.papers.add(c.path); else st.papers.delete(c.path); update(); } }),
        h("span", { class: "mono" }, c.path))),
        h("label", { class: "check" }, h("input", { type: "checkbox", onchange: (e) => { st.recipe = e.target.checked; update(); } }), t("new.recipe")))
        : h("p", { class: "hint" }, t("new.no_cards")),
      h("p", { class: "hint" }, t("new.papers_hint"))),
    h("section", {}, h("h2", {}, t("new.command")), cmdPre,
      h("div", { class: "row", style: "margin-top:12px" }, estBtn, runBtn,
        h("button", { type: "button", onclick: () => copy(command().replace(/ \\\n {2}/g, " ")) }, t("new.copy_command"))),
      planBox));
  drawBudgets();
  update();
  detect();
}

function renderPlan(p) {
  const rows = [];
  const tm = p.timing || {};
  rows.push(kv(t("plan.candidates"), t("plan.candidates_v", { valid: p.candidates.valid, gen: p.candidates.generated, fam: p.candidates.families.length })));
  rows.push(kv(t("plan.epoch"), tm.sec_per_epoch_median ? t("plan.epoch_v", { s: tm.sec_per_epoch_median.toFixed(2), lo: tm.sec_per_epoch_range[0].toFixed(2), hi: tm.sec_per_epoch_range[1].toFixed(2) }) : "—"));
  rows.push(kv(t("plan.design"), duration(p.seconds_design)));
  if (p.papers) rows.push(kv(t("plan.papers_time"), duration(p.seconds_papers)));
  rows.push(kv(t("plan.total"), duration(p.seconds_total)));
  const ref = (p.reference || [])[0];
  if (ref) rows.push(kv(t("plan.to_beat"), ref.name + " · val " + fmt(ref.val) + " · test " + fmt(ref.test)));
  const kids = [h("h2", { style: "margin-top:28px" }, t("plan.title")), h("div", { class: "grid3 sunken" }, rows)];
  if (p.rungs) kids.push(h("p", { class: "hint" }, p.rungs.map((r) => t("plan.rung", { r: r.rung, e: r.epochs, n: r.candidates })).join(" → ")));
  if (p.papers) {
    kids.push(h("div", { class: "scroll", style: "margin-top:14px" }, h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, t("plan.paper")),
      h("th", { class: "r" }, t("run.col.params")), h("th", { class: "r" }, t("plan.time")), h("th", {}, t("plan.reported")), h("th", {}, t("runs.col.status")))),
      h("tbody", {}, p.papers.map((x) => h("tr", {}, h("td", {}, x.paper || x.card.split("/").pop()),
        h("td", { class: "r num" }, x.params ? x.params.toLocaleString() : "—"), h("td", { class: "r num" }, x.code === "ok" ? duration(x.seconds) : "—"),
        h("td", { style: "font-size:13px" }, (x.reported || []).filter((y) => y.metric).map((y) => (y.dataset || "?") + ": " + y.metric + " " + y.value).join("; ") || "—"),
        h("td", { class: x.code === "ok" ? "good" : "warn", style: "font-size:13px" }, t("plan.status." + (x.code || "fail"), x.detail || {}))))))));
  }
  for (const w of p.warnings || []) kids.push(h("p", { class: "warn" }, t("plan.warn." + w.code, w.params || {})));
  kids.push(h("p", { class: "hint" }, t("plan.note")));
  return h("div", {}, kids);
}

/* ------------------------------------------------------------------ 5. AI / MCP */

const CLIENTS = ["claude", "codex", "antigravity"];

function clientSnippet(c, url) {
  if (c === "claude") return 'export DLNET_TOKEN=<token>\nclaude mcp add --transport http dlnet ' + url + ' \\\n  --header "Authorization: Bearer $DLNET_TOKEN"';
  if (c === "codex") return '# ~/.codex/config.toml\n[mcp_servers.dlnet]\nurl = "' + url + '"\nbearer_token_env_var = "DLNET_TOKEN"\n\n# or: codex mcp add dlnet --url ' + url + ' --bearer-token-env-var DLNET_TOKEN';
  return '// mcp_config.json (Antigravity: Manage MCP servers → View raw config)\n{\n  "mcpServers": {\n    "dlnet": {\n      "serverUrl": "' + url + '",\n      "headers": { "Authorization": "Bearer <token>" }\n    }\n  }\n}';
}

async function viewAI(main) {
  const st = { client: store("client") || "claude", showToken: false };
  const form = { lan: false, port: 8765, max_jobs: 1, max_budget: "12h", allow_fetch: false };
  const statusBox = h("div");
  const tokenBox = h("div");
  const snippet = h("pre", { class: "code" });
  const tabs = h("div", { class: "tabs", role: "tablist" });
  const logPre = h("pre", { class: "code", style: "max-height:260px; overflow:auto" });
  const testOut = h("p", { class: "hint" });
  const serveCmd = h("pre", { class: "code" });
  let status = null;

  const inputs = {
    lan: h("input", { type: "checkbox", onchange: (e) => { form.lan = e.target.checked; drawCmd(); } }),
    port: h("input", { type: "number", value: 8765, style: "width:100px", oninput: (e) => { form.port = Number(e.target.value); drawCmd(); } }),
    max_jobs: h("input", { type: "number", value: 1, min: 1, max: 8, style: "width:80px", oninput: (e) => { form.max_jobs = Number(e.target.value); drawCmd(); } }),
    max_budget: h("input", { type: "text", class: "mono", value: "12h", style: "width:100px", oninput: (e) => { form.max_budget = e.target.value; drawCmd(); } }),
    allow_fetch: h("input", { type: "checkbox", onchange: (e) => { form.allow_fetch = e.target.checked; drawCmd(); } }),
  };
  function url() { return status && status.running ? status.url : "http://" + (form.lan ? "<this-computer-ip>" : "127.0.0.1") + ":" + form.port + "/mcp"; }
  function drawCmd() {
    const a = ["dlnet serve --workspace " + S.state.workspace, "--port " + form.port];
    if (form.lan) a.push("--host 0.0.0.0");
    if (form.max_jobs !== 1) a.push("--max-jobs " + form.max_jobs);
    if (form.max_budget !== "12h") a.push("--max-budget " + form.max_budget);
    if (form.allow_fetch) a.push("--allow-fetch");
    serveCmd.textContent = "export DLNET_TOKEN=<token>\n" + a.join(" \\\n  ");
    snippet.textContent = clientSnippet(st.client, url());
  }
  function drawTabs() {
    tabs.replaceChildren(...CLIENTS.map((c) => h("button", { type: "button", role: "tab", class: c === st.client ? "on" : "", "aria-selected": c === st.client ? "true" : "false",
      onclick: () => { st.client = c; store("client", c); drawTabs(); drawCmd(); } }, t("ai.client." + c))));
  }
  function drawStatus() {
    const on = status && status.running;
    statusBox.replaceChildren(h("div", { class: "row", style: "justify-content:space-between" },
      h("div", {}, h("span", { class: "status " + (on ? "s-running" : "s-cancelled") }, on ? t("ai.running") : t("ai.stopped")),
        on ? h("span", { class: "mono muted", style: "margin-left:12px; font-size:13px" }, status.url) : null),
      h("div", { class: "row" },
        on ? h("button", { type: "button", onclick: stop }, t("ai.stop")) : h("button", { type: "button", class: "primary", onclick: start }, t("ai.start")),
        h("button", { type: "button", onclick: test, disabled: !on }, t("ai.test")))));
    tokenBox.replaceChildren();
    if (on && status.token) {
      tokenBox.append(h("div", { class: "row", style: "margin-top:14px" }, h("span", { class: "muted" }, t("ai.token")),
        h("span", { class: "mono" }, st.showToken ? status.token : "•".repeat(16)),
        h("button", { type: "button", class: "chip", onclick: () => { st.showToken = !st.showToken; drawStatus(); } }, st.showToken ? t("ai.hide") : t("ai.show")),
        h("button", { type: "button", class: "chip", onclick: () => copy(status.token) }, t("common.copy"))));
    }
    if (status && !on && status.exit_code !== null && status.exit_code !== undefined) tokenBox.append(h("p", { class: "bad" }, t("ai.exited", { code: status.exit_code })));
    drawCmd();
  }
  async function refresh() {
    status = await api("GET", "/api/mcp");
    logPre.textContent = (status.log || []).slice(-40).join("\n") || t("ai.no_activity");
    drawStatus();
  }
  async function start() {
    try { status = await api("POST", "/api/mcp/start", form); toast(t("ai.started")); await refresh(); }
    catch (e) { toast(e.message); }
  }
  async function stop() { status = await api("POST", "/api/mcp/stop", {}); await refresh(); }
  async function test() {
    testOut.textContent = t("ai.testing");
    const r = await api("POST", "/api/mcp/test", {});
    testOut.className = r.ok ? "good" : "bad";
    testOut.textContent = r.ok ? t("ai.test_ok", { tools: r.tools, ms: r.ms, proto: r.protocol }) : t("ai.test_fail", { err: r.error });
  }
  const { tools } = await api("GET", "/api/tools");
  main.replaceChildren(
    h("h1", {}, t("ai.title")),
    h("p", { class: "lead" }, t("ai.lead")),
    h("section", { style: "margin-top:0" }, h("h2", {}, t("ai.server")), h("div", { class: "card" }, statusBox, tokenBox, testOut,
      h("div", { class: "row", style: "margin-top:18px; gap:20px; align-items:flex-end" },
        h("label", { class: "field", style: "margin:0" }, h("span", {}, t("ai.port")), inputs.port),
        h("label", { class: "field", style: "margin:0" }, h("span", {}, t("ai.max_jobs")), inputs.max_jobs),
        h("label", { class: "field", style: "margin:0" }, h("span", {}, t("ai.max_budget")), inputs.max_budget)),
      h("div", { style: "margin-top:12px" },
        h("label", { class: "check", style: "display:flex" }, inputs.lan, t("ai.lan")),
        h("label", { class: "check", style: "display:flex" }, inputs.allow_fetch, t("ai.fetch"))),
      h("p", { class: "hint" }, t("ai.lan_hint")))),
    h("section", {}, h("h2", {}, t("ai.same_in_terminal")), serveCmd),
    h("section", {}, h("h2", {}, t("ai.connect")), tabs, snippet,
      h("div", { class: "row", style: "margin-top:10px" }, h("button", { type: "button", onclick: () => copy(snippet.textContent) }, t("common.copy"))),
      h("p", { class: "hint" }, t("ai.connect_hint"))),
    h("section", {}, h("h2", {}, t("ai.abilities", { n: tools.length })),
      h("ul", { class: "plain" }, tools.map((x) => h("li", {}, h("span", { class: "mono", style: "font-size:13px" }, x.name), " — ", t("tool." + x.name))))),
    h("section", {}, h("h2", {}, t("ai.activity")), logPre));
  drawTabs();
  await refresh();
  every(3000, () => { if (S.view === "ai") refresh(); });
}

/* ------------------------------------------------------------------ start */

async function boot() {
  const saved = store("lang");
  S.lang = LANGS.includes(saved) ? saved : ((navigator.language || "en").toLowerCase().startsWith("vi") ? "vi" : "en");
  const [en, vi] = await Promise.all(LANGS.map((l) => fetch("/static/i18n/" + l + ".json").then((r) => r.json())));
  S.dict = { en, vi };
  document.getElementById("lang").addEventListener("click", () => {
    S.lang = S.lang === "en" ? "vi" : "en";
    store("lang", S.lang);
    applyChrome();
    route();
  });
  document.getElementById("theme").addEventListener("click", () => {
    const order = ["auto", "light", "dark"];
    setTheme(order[(order.indexOf(store("theme") || "auto") + 1) % order.length]);
  });
  await loadState();
  window.addEventListener("hashchange", route);
  route();
}

boot();
