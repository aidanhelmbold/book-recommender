/* Canvas renderer for the book map.
 *
 * Positions arrive already laid out by the server, in a fixed 1000x700 space —
 * the client only pans, zooms and draws. That split is deliberate: ForceAtlas2
 * on the induced subgraph belongs where the graph already lives, and it keeps a
 * given query drawing the same picture every time instead of re-simulating.
 */

const LAYOUT = { width: 1000, height: 700 };
const NODE_MIN = 4.5;
const NODE_MAX = 13;

const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const tip = document.getElementById("tip");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const emptyEl = document.getElementById("empty");
const legendEl = document.getElementById("legend");
const clustersEl = document.getElementById("clusters");
const bridgesEl = document.getElementById("bridges");
const bridgeListEl = document.getElementById("bridge-list");

let graph = { nodes: [], edges: [], community_labels: {} };
let view = { scale: 1, x: 0, y: 0 };
let hovered = null;
let selected = null;
let bridgeIds = new Set();

const css = (name) =>
  getComputedStyle(document.querySelector(".viz-root")).getPropertyValue(name).trim();

const roleColour = (role) =>
  role === "seed" ? css("--series-seed") : role === "recommendation" ? css("--series-rec") : css("--series-ctx");

/* --- geometry ----------------------------------------------------------- */

function resize() {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  fitToView();
}

function fitToView() {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width) return;
  // Uniform scale: letting x and y scale independently would stretch the graph
  // and destroy the relative distances that are the entire payload of a map.
  const scale = Math.min(rect.width / LAYOUT.width, rect.height / LAYOUT.height);
  view = {
    scale,
    x: (rect.width - LAYOUT.width * scale) / 2,
    y: (rect.height - LAYOUT.height * scale) / 2,
  };
  draw();
}

const toScreen = (node) => ({
  x: node.x * view.scale + view.x,
  y: node.y * view.scale + view.y,
});

function radiusOf(node) {
  if (node.role === "seed") return NODE_MAX;
  if (node.score == null) return NODE_MIN;
  const best = Math.max(...graph.nodes.map((n) => n.score ?? 0), 1e-9);
  // sqrt so the eye compares *area*, which is how a circle reads.
  return NODE_MIN + (NODE_MAX - NODE_MIN - 2) * Math.sqrt(node.score / best);
}

function nodeAt(px, py) {
  let found = null;
  // Reverse order so the node drawn last (on top) is the one you hit.
  for (let i = graph.nodes.length - 1; i >= 0; i -= 1) {
    const node = graph.nodes[i];
    const { x, y } = toScreen(node);
    // Hit target is deliberately larger than the mark.
    const reach = radiusOf(node) + 6;
    if ((px - x) ** 2 + (py - y) ** 2 <= reach ** 2) {
      found = node;
      break;
    }
  }
  return found;
}

/* --- drawing ------------------------------------------------------------ */

function draw() {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!graph.nodes.length) return;

  const byId = new Map(graph.nodes.map((n) => [n.work_id, n]));
  const focus = hovered ?? selected;
  const adjacent = new Set();
  if (focus) {
    for (const edge of graph.edges) {
      if (edge.src === focus.work_id) adjacent.add(edge.dst);
      if (edge.dst === focus.work_id) adjacent.add(edge.src);
    }
  }

  // Edges first and recessive: they are the grid of this chart, not its subject.
  ctx.strokeStyle = css("--edge");
  for (const edge of graph.edges) {
    const a = byId.get(edge.src);
    const b = byId.get(edge.dst);
    if (!a || !b) continue;
    const lit = focus && (edge.src === focus.work_id || edge.dst === focus.work_id);
    ctx.globalAlpha = lit ? 0.5 : focus ? 0.05 : 0.14;
    ctx.lineWidth = lit ? 1.8 : 1;
    const p = toScreen(a);
    const q = toScreen(b);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.lineTo(q.x, q.y);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  for (const node of graph.nodes) {
    const { x, y } = toScreen(node);
    const r = radiusOf(node);
    const dim = focus && node !== focus && !adjacent.has(node.work_id);

    ctx.globalAlpha = dim ? 0.28 : 1;
    // 2px surface ring, so two touching nodes still read as two nodes.
    ctx.beginPath();
    ctx.arc(x, y, r + 2, 0, Math.PI * 2);
    ctx.fillStyle = css("--surface-1");
    ctx.fill();

    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = roleColour(node.role);
    ctx.fill();

    // Bridges get a double ring rather than a colour of their own: the role
    // palette is capped at three validated hues, so a fourth would be
    // indistinguishable from blue for a protanopic reader. Geometry is not.
    if (bridgeIds.has(node.work_id)) {
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = css("--text-secondary");
      for (const gap of [3, 6]) {
        ctx.beginPath();
        ctx.arc(x, y, r + gap, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    // Seeds also carry a ring: role must not rest on hue alone.
    if (node.role === "seed") {
      ctx.lineWidth = 2;
      ctx.strokeStyle = css("--text-primary");
      ctx.beginPath();
      ctx.arc(x, y, r + 4, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;

  drawDirectLabels(focus, adjacent);
}

function drawDirectLabels(focus, adjacent) {
  // Selective direct labels: seeds always, plus whatever is in focus. Labelling
  // every node turns the map into a wall of text.
  const wanted = graph.nodes.filter(
    (n) => n.role === "seed" || n === focus || (focus && adjacent.has(n.work_id)),
  );
  ctx.font = "500 12px ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  for (const node of wanted) {
    const { x, y } = toScreen(node);
    const text = node.title.length > 34 ? `${node.title.slice(0, 33)}…` : node.title;
    const width = ctx.measureText(text).width;
    const top = y - radiusOf(node) - 8;
    ctx.fillStyle = css("--surface-1");
    ctx.globalAlpha = 0.82;
    ctx.fillRect(x - width / 2 - 3, top - 13, width + 6, 16);
    ctx.globalAlpha = 1;
    // Labels wear text tokens, never the series colour.
    ctx.fillStyle = css("--text-primary");
    ctx.fillText(text, x, top);
  }
}

/* --- interaction -------------------------------------------------------- */

function showTip(node, px, py) {
  const community = node.community == null ? null : graph.community_labels[String(node.community)];
  const bits = [
    `<strong>${escapeHtml(node.title)}</strong>`,
    node.authors.length ? `<span class="meta">${escapeHtml(node.authors.join(", "))}</span>` : "",
    community ? `<span class="meta">cluster: ${escapeHtml(community)}</span>` : "",
    node.score != null ? `<span class="meta">score ${node.score.toFixed(4)}</span>` : "",
    `<span class="meta">${node.degree} connections</span>`,
  ].filter(Boolean);
  tip.innerHTML = bits.join("<br>");
  tip.hidden = false;
  const rect = canvas.getBoundingClientRect();
  const box = tip.getBoundingClientRect();
  // Flip before overflowing rather than clipping at the edge.
  const left = px + 14 + box.width > rect.width ? px - box.width - 14 : px + 14;
  const top = py + 14 + box.height > rect.height ? py - box.height - 14 : py + 14;
  tip.style.left = `${Math.max(4, left)}px`;
  tip.style.top = `${Math.max(4, top)}px`;
}

const escapeHtml = (text) =>
  text.replace(/[&<>"']/g, (ch) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch],
  );

let dragging = null;

canvas.addEventListener("mousemove", (event) => {
  const rect = canvas.getBoundingClientRect();
  const px = event.clientX - rect.left;
  const py = event.clientY - rect.top;

  if (dragging) {
    view.x += px - dragging.x;
    view.y += py - dragging.y;
    dragging = { x: px, y: py };
    draw();
    return;
  }

  const found = nodeAt(px, py);
  if (found !== hovered) {
    hovered = found;
    draw();
  }
  if (found) showTip(found, px, py);
  else tip.hidden = true;
});

canvas.addEventListener("mouseleave", () => {
  hovered = null;
  dragging = null;
  tip.hidden = true;
  canvas.classList.remove("dragging");
  draw();
});

canvas.addEventListener("mousedown", (event) => {
  const rect = canvas.getBoundingClientRect();
  dragging = { x: event.clientX - rect.left, y: event.clientY - rect.top };
  canvas.classList.add("dragging");
});

window.addEventListener("mouseup", () => {
  dragging = null;
  canvas.classList.remove("dragging");
});

canvas.addEventListener("click", (event) => {
  const rect = canvas.getBoundingClientRect();
  const found = nodeAt(event.clientX - rect.left, event.clientY - rect.top);
  if (!found) return;
  selected = selected === found ? null : found;
  markActive(selected?.work_id);
  draw();
  if (selected) expand(selected.work_id);
});

canvas.addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    const factor = Math.exp(-event.deltaY * 0.0016);
    const next = Math.min(8, Math.max(0.25, view.scale * factor));
    // Zoom about the cursor, so the thing under the pointer stays put.
    view.x = px - ((px - view.x) * next) / view.scale;
    view.y = py - ((py - view.y) * next) / view.scale;
    view.scale = next;
    draw();
  },
  { passive: false },
);

/* --- data --------------------------------------------------------------- */

async function post(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json();
}

function seedList() {
  return document
    .getElementById("seeds")
    .value.split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

async function run() {
  const seeds = seedList();
  if (!seeds.length) {
    setStatus("Name at least one book.", "error");
    return;
  }
  setStatus("Building map…");
  const options = {
    seeds,
    beta: Number(document.getElementById("beta").value),
    mmr_lambda: Number(document.getElementById("mmr").value),
    exclude_same_author: document.getElementById("same-author").checked,
  };

  try {
    const [recs, sub] = await Promise.all([
      post("/api/recommend", { ...options, n: 20 }),
      post("/api/subgraph", { seeds, n: 30, include_neighbors: true }),
    ]);
    graph = sub;
    selected = null;
    hovered = null;
    renderResults(recs);
    renderLegend();
    renderClusters();
    loadBridges(seeds);
    fitToView();

    const parts = [`${sub.nodes.length} books, ${sub.edges.length} connections`];
    if (recs.unresolved.length) parts.push(`could not find: ${recs.unresolved.join(", ")}`);
    setStatus(parts.join(" · "), recs.unresolved.length ? "warn" : "");
  } catch (error) {
    setStatus(`Failed: ${error.message}`, "error");
  }
}

async function loadBridges(seeds) {
  if (!bridgesEl || !bridgeListEl) return;
  try {
    const params = new URLSearchParams({ seeds: seeds.join(","), top_n: "8" });
    const response = await fetch(`/api/bridges?${params}`);
    if (!response.ok) {
      // 409 means the graph has no communities yet; nothing to bridge between.
      bridgesEl.hidden = true;
      return;
    }
    const { bridges } = await response.json();
    bridgeIds = new Set(bridges.map((b) => b.work_id));
    bridgeListEl.innerHTML = "";
    for (const bridge of bridges) {
      const names = bridge.community_labels
        .map((label, i) => label || `cluster ${bridge.communities[i]}`)
        .slice(0, 2);
      const extra = bridge.communities.length - names.length;
      const li = document.createElement("li");
      const title = document.createElement("span");
      title.className = "b-title";
      title.textContent = bridge.title;
      const connects = document.createElement("span");
      connects.className = "b-connects";
      connects.textContent =
        names.join(" ↔ ") + (extra > 0 ? `  +${extra} more` : "");
      li.append(title, connects);
      li.addEventListener("click", () => {
        const node = graph.nodes.find((n) => n.work_id === bridge.work_id);
        if (node) {
          selected = node;
          draw();
        } else {
          setStatus(`${bridge.title} is not on this map — it bridges clusters further out`);
        }
      });
      bridgeListEl.append(li);
    }
    bridgesEl.hidden = bridges.length === 0;
    draw();
  } catch {
    bridgesEl.hidden = true;
  }
}

function renderLegend() {
  // Only roles actually on screen get a legend row. Advertising "nearby context"
  // when the layout contains none invites the reader to hunt for a colour that
  // is not there, and makes the legend a claim about the palette rather than
  // about this map.
  if (!legendEl) return;
  const present = new Set(graph.nodes.map((n) => n.role));
  for (const li of legendEl.children) {
    li.hidden = !present.has(li.dataset.role);
  }
  legendEl.hidden = present.size === 0;
}

function renderClusters() {
  // Clusters are named in the DOM rather than drawn at their centroid. In a dense
  // subgraph the centroid of a community's visible members frequently lands
  // inside another community's territory, so an on-canvas label asserted
  // something false -- "space / far / future" printed beside a Regency romance.
  // Text here is also selectable and reachable by a screen reader, and it is what
  // keeps cluster identity off colour, which nine communities cannot carry.
  if (!clustersEl) return;
  const counts = new Map();
  for (const node of graph.nodes) {
    if (node.community == null) continue;
    counts.set(node.community, (counts.get(node.community) ?? 0) + 1);
  }
  const rows = [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([community, count]) => {
      const label = graph.community_labels[String(community)] ?? `cluster ${community}`;
      return `<li><span class="c-name">${escapeHtml(label)}</span><span class="c-count">${count}</span></li>`;
    });
  clustersEl.innerHTML = rows.length
    ? `<h3>Clusters on this map</h3><ul>${rows.join("")}</ul>`
    : "";
}

function renderResults(payload) {
  resultsEl.innerHTML = "";
  emptyEl.hidden = payload.recommendations.length > 0;

  for (const rec of payload.recommendations) {
    const li = document.createElement("li");
    li.dataset.workId = rec.work_id;

    const title = document.createElement("div");
    title.className = "r-title";
    title.textContent = rec.title;

    const meta = document.createElement("div");
    meta.className = "r-meta";
    meta.textContent = `${rec.authors.join(", ") || "unknown"} · ${rec.score.toFixed(4)}`;

    li.append(title, meta);

    for (const exp of rec.explanations.slice(0, 2)) {
      const why = document.createElement("div");
      why.className = "r-why";
      why.textContent =
        exp.hops === 1
          ? `directly linked to ${exp.seed_title}`
          : `${exp.hops} hops from ${exp.seed_title}`;
      li.append(why);
    }

    li.addEventListener("click", () => {
      const node = graph.nodes.find((n) => n.work_id === rec.work_id);
      selected = node ?? null;
      markActive(rec.work_id);
      draw();
    });

    resultsEl.append(li);
  }
}

function markActive(workId) {
  for (const li of resultsEl.children) {
    li.classList.toggle("active", li.dataset.workId === workId);
  }
}

async function expand(workId) {
  try {
    const response = await fetch(`/api/book/${encodeURIComponent(workId)}/neighbors?limit=8`);
    if (!response.ok) return;
    const { neighbors } = await response.json();
    const known = new Set(graph.nodes.map((n) => n.work_id));
    const anchor = graph.nodes.find((n) => n.work_id === workId);
    if (!anchor) return;

    let added = 0;
    neighbors.forEach((neighbor, index) => {
      if (known.has(neighbor.work_id)) {
        graph.edges.push({ src: workId, dst: neighbor.work_id, weight: neighbor.weight });
        return;
      }
      // Placed on a ring around the anchor: a newly fetched node has no layout
      // position of its own, and re-running the server layout would move every
      // existing node, which reads as the map jumping.
      const angle = (index / Math.max(neighbors.length, 1)) * Math.PI * 2;
      graph.nodes.push({
        work_id: neighbor.work_id,
        title: neighbor.title,
        authors: neighbor.authors,
        x: anchor.x + Math.cos(angle) * 70,
        y: anchor.y + Math.sin(angle) * 70,
        community: null,
        is_seed: false,
        score: null,
        degree: 1,
        role: "context",
      });
      graph.edges.push({ src: workId, dst: neighbor.work_id, weight: neighbor.weight });
      added += 1;
    });
    if (added) setStatus(`Expanded ${anchor.title}: ${added} more books`);
    draw();
  } catch {
    /* expansion is a convenience; a failure should not disturb the map */
  }
}

function setStatus(text, kind = "") {
  statusEl.textContent = text;
  statusEl.dataset.kind = kind;
}

/* --- wiring ------------------------------------------------------------- */

document.getElementById("seed-form").addEventListener("submit", (event) => {
  event.preventDefault();
  run();
});

for (const [id, outId, digits] of [["beta", "beta-out", 2], ["mmr", "lambda-out", 2]]) {
  const input = document.getElementById(id);
  const out = document.getElementById(outId);
  input.addEventListener("input", () => {
    out.textContent = Number(input.value).toFixed(digits);
  });
}

document.getElementById("theme").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const current = root.dataset.theme || (dark ? "dark" : "light");
  root.dataset.theme = current === "dark" ? "light" : "dark";
  draw();
});

// The canvas colours are read from CSS, so an OS theme change has to trigger a
// repaint or the graph keeps the previous mode's palette.
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
window.addEventListener("resize", resize);

resize();
setStatus("");
if (legendEl) legendEl.hidden = true;
