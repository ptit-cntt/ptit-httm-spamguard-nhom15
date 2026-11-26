// SpamGuard Web Admin - vanilla JS, gọi trực tiếp API cùng origin.
// Nội dung tin nhắn là dữ liệu không tin cậy -> luôn render bằng textContent (chống XSS).

const TOKEN_KEY = "spamguard_admin_token";
const LABEL_COLORS = { ham: "#16a34a", spam: "#f59e0b", toxic: "#dc2626" };
const LABELS = ["ham", "spam", "toxic"];
const charts = {};

function getToken() {
  try { return sessionStorage.getItem(TOKEN_KEY) || ""; } catch { return ""; }
}

function setToken(token) {
  try { sessionStorage.setItem(TOKEN_KEY, token); } catch { /* private mode: bỏ qua */ }
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", "X-Admin-Token": getToken(), ...(options.headers || {}) },
  });
  if (response.status === 401) {
    showLogin("Token không hợp lệ hoặc đã hết hạn");
    throw new Error("unauthorized");
  }
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.json();
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([key, value]) => {
    if (key === "text") node.textContent = value;
    else if (key === "onclick") node.addEventListener("click", value);
    else node.setAttribute(key, value);
  });
  children.forEach((child) => node.append(child));
  return node;
}

const pill = (value, extra = "") => el("span", { class: `pill ${value} ${extra}`, text: value });
const fmtPct = (value) => `${(value * 100).toFixed(1)}%`;
const fmtTime = (iso) => new Date(iso).toLocaleString("vi-VN");

function renderTable(tableId, headers, rows) {
  const table = document.getElementById(tableId);
  table.replaceChildren(
    el("thead", {}, [el("tr", {}, headers.map((h) => el("th", { text: h })))]),
    el("tbody", {}, rows.map((cells) => el("tr", {}, cells.map((cell) => {
      if (cell instanceof Node) return el("td", {}, [cell]);
      if (cell && cell.node) return el("td", { class: cell.cls || "" }, [cell.node]);
      if (cell && typeof cell === "object") return el("td", { class: cell.cls || "", text: cell.text ?? "" });
      return el("td", { text: cell ?? "" });
    })))),
  );
  if (!rows.length) table.append(el("tbody", {}, [el("tr", {}, [el("td", { text: "Không có dữ liệu" })])]));
}

function drawChart(id, config) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), config);
}

function updateModelBadge(model) {
  if (model) document.getElementById("modelBadge").textContent = `production: v${model.version} · ${model.model_type}`;
}

async function refreshModelBadge() {
  const response = await fetch("/ready");
  if (response.ok) {
    const ready = await response.json();
    updateModelBadge({ version: ready.model_version, model_type: ready.model_type });
  }
}

// ------------------------------ Dashboard ----------------------------------
async function loadDashboard() {
  const days = document.getElementById("daysSelect").value;
  const stats = await api(`/v1/admin/stats?days=${days}`);
  document.getElementById("kpiTotal").textContent = stats.total.toLocaleString("vi-VN");
  document.getElementById("kpiSpam").textContent = fmtPct(stats.spam_rate);
  document.getElementById("kpiToxic").textContent = fmtPct(stats.toxic_rate);
  document.getElementById("kpiBlocked").textContent = (stats.by_action.block || 0).toLocaleString("vi-VN");
  document.getElementById("kpiReview").textContent = stats.pending_review;
  document.getElementById("kpiFeedback").textContent = stats.pending_feedback;
  document.getElementById("kpiLatency").textContent = `${stats.avg_latency_ms.toFixed(1)} ms`;
  updateModelBadge(stats.model);

  const dates = [...new Set(stats.timeline.map((r) => r.date))];
  drawChart("timelineChart", {
    type: "bar",
    data: {
      labels: dates,
      datasets: LABELS.map((label) => ({
        label, backgroundColor: LABEL_COLORS[label],
        data: dates.map((d) => (stats.timeline.find((r) => r.date === d && r.label === label) || {}).count || 0),
      })),
    },
    options: { responsive: true, scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } } },
  });
  drawChart("labelChart", {
    type: "doughnut",
    data: {
      labels: LABELS,
      datasets: [{ data: LABELS.map((l) => stats.by_label[l] || 0), backgroundColor: LABELS.map((l) => LABEL_COLORS[l]) }],
    },
    options: { aspectRatio: 1.6 },
  });
  const channels = [...new Set(stats.by_channel.map((r) => r.channel))];
  drawChart("channelChart", {
    type: "bar",
    data: {
      labels: channels,
      datasets: LABELS.map((label) => ({
        label, backgroundColor: LABEL_COLORS[label],
        data: channels.map((c) => (stats.by_channel.find((r) => r.channel === c && r.label === label) || {}).count || 0),
      })),
    },
    options: { indexAxis: "y", scales: { x: { stacked: true }, y: { stacked: true } } },
  });
  renderTable("sendersTable", ["Sender", "Số tin bị gắn cờ"],
    stats.top_senders.map((s) => [s.sender_id, s.flagged]));

  const ready = stats.pending_feedback >= stats.retrain_min_feedback;
  document.getElementById("loopStatus").replaceChildren(
    el("p", { text: `Feedback đã duyệt chờ retrain: ${stats.pending_feedback} / ngưỡng ${stats.retrain_min_feedback}. `
      + `Báo cáo nhãn sai từ người dùng: ${stats.user_reports} (cần admin xác nhận trước khi dùng để huấn luyện).` }),
    el("p", { text: ready
      ? "Đủ điều kiện: job retrain định kỳ sẽ huấn luyện model ứng viên, chạy quality gate và tự promote nếu đạt."
      : "Chưa đủ feedback để kích hoạt retrain." }),
  );
}

// ------------------------------ Messages -----------------------------------
async function loadMessages() {
  const params = new URLSearchParams({ limit: "100" });
  const label = document.getElementById("filterLabel").value;
  const action = document.getElementById("filterAction").value;
  const query = document.getElementById("filterQuery").value.trim();
  if (label) params.set("label", label);
  if (action) params.set("action", action);
  if (query) params.set("q", query);
  const rows = await api(`/v1/admin/messages?${params}`);
  renderTable("messagesTable", ["ID", "Thời gian", "Kênh", "Nội dung", "Nhãn", "Conf.", "Hành động", "Model", "Review"],
    rows.map((m) => [m.id, fmtTime(m.created_at), m.channel, { cls: "text", text: m.text },
      pill(m.predicted_label), m.confidence.toFixed(2), pill(m.action), `v${m.model_version}`,
      m.reviewed_label ? pill(m.reviewed_label) : "–"]));
}

// ------------------------------ Review queue -------------------------------
function reviewButtons(messageId, onDone) {
  const wrap = el("span");
  LABELS.forEach((label) => wrap.append(el("button", {
    class: "btn-sm", text: label,
    onclick: async () => { await api(`/v1/admin/messages/${messageId}/review`, { method: "POST", body: JSON.stringify({ label }) }); onDone(); },
  })));
  return wrap;
}

async function loadReview() {
  const rows = await api("/v1/admin/messages?action=review&only_unreviewed=true&limit=100");
  renderTable("reviewTable", ["ID", "Nội dung", "Dự đoán", "Conf.", "Gán nhãn đúng"],
    rows.map((m) => [m.id, { cls: "text", text: m.text }, pill(m.predicted_label), m.confidence.toFixed(2),
      reviewButtons(m.id, loadReview)]));
}

// ------------------------------ Model registry -----------------------------
async function loadModels() {
  const index = await api("/v1/admin/models");
  const production = index.aliases.production;
  renderTable("modelsTable", ["Version", "Loại", "Nguồn", "Macro-F1", "Accuracy", "Ham chặn nhầm", "p95 (ms)", "Size (MB)", "Data", "Tạo lúc", ""],
    [...index.versions].reverse().map((v) => [
      v.version === production ? pill(`v${v.version} prod`, "prod") : `v${v.version}`,
      v.model_type, v.source, v.metrics.macro_f1?.toFixed(4), v.metrics.accuracy?.toFixed(4),
      v.metrics.ham_false_block_rate != null ? fmtPct(v.metrics.ham_false_block_rate) : "–",
      v.metrics.p95_latency_ms?.toFixed(2), v.metrics.model_size_mb?.toFixed(1), v.data_version, fmtTime(v.created_at),
      v.version === production ? "" : el("button", {
        class: "btn-sm", text: "Promote",
        onclick: async () => {
          if (!confirm(`Promote v${v.version} lên production?`)) return;
          await api(`/v1/admin/models/${v.version}/promote`, { method: "POST", body: JSON.stringify({ reason: "manual promote từ Web Admin" }) });
          loadModels();
        },
      }),
    ]));
  renderTable("historyTable", ["Thời điểm", "Từ", "Sang", "Lý do"],
    [...index.history].reverse().map((h) => [fmtTime(h.at), h.from ? `v${h.from}` : "–", `v${h.to}`, h.reason]));
}

// ------------------------------ Tester -------------------------------------
async function runTest() {
  const text = document.getElementById("testText").value.trim();
  if (!text) return;
  const result = await api("/v1/messages/classify", {
    method: "POST", body: JSON.stringify({ text, channel: document.getElementById("testChannel").value }),
  });
  document.getElementById("testResult").textContent = JSON.stringify(result, null, 2);
}

// ------------------------------ Navigation ---------------------------------
const loaders = { dashboard: loadDashboard, messages: loadMessages, review: loadReview, models: loadModels, tester: async () => {} };

function showTab(name) {
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((s) => s.classList.toggle("hidden", s.id !== name));
  loaders[name]().catch((error) => console.error(error));
  refreshModelBadge().catch((error) => console.error(error));
}

function showLogin(message = "") {
  document.getElementById("login").classList.remove("hidden");
  document.querySelectorAll(".tab").forEach((s) => s.classList.add("hidden"));
  document.getElementById("loginError").textContent = message;
}

document.getElementById("loginBtn").addEventListener("click", () => {
  setToken(document.getElementById("tokenInput").value.trim());
  document.getElementById("login").classList.add("hidden");
  showTab("dashboard");
});
document.querySelectorAll("nav button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
document.getElementById("refreshBtn").addEventListener("click", loadDashboard);
document.getElementById("daysSelect").addEventListener("change", loadDashboard);
document.getElementById("searchBtn").addEventListener("click", loadMessages);
document.getElementById("testBtn").addEventListener("click", () => runTest().catch((e) => {
  document.getElementById("testResult").textContent = e.message;
}));

// Cho phép truyền token qua fragment (#token=...) khi mở từ công cụ nội bộ; fragment không gửi lên server
const hashToken = new URLSearchParams(location.hash.slice(1)).get("token");
if (hashToken) {
  setToken(hashToken);
  history.replaceState(null, "", location.pathname + location.search);
}
const initialTab = new URLSearchParams(location.search).get("tab") || "dashboard";
if (getToken()) showTab(initialTab); else showLogin();
