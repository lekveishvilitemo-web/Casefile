const state = { view: "investigate", offset: 0, limit: 8, query: "" };

const $ = (id) => document.getElementById(id);

document.querySelectorAll(".nav-btn").forEach((button) => {
  button.addEventListener("click", () => show(button.dataset.view));
});

function show(view) {
  state.view = view;
  document.querySelectorAll(".nav-btn").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  for (const name of ["investigate", "documents", "evaluation", "method"]) {
    $(`view-${name}`).hidden = name !== view;
  }
  if (view === "documents") loadDocuments();
  if (view === "evaluation") loadEvaluation();
}

async function refreshStatus() {
  try {
    const stats = await api("/api/stats");
    const provider = stats.llm_provider === "none" ? "no model loaded" : `${stats.llm_provider}`;
    $("corpus-status").textContent =
      `${stats.documents.toLocaleString()} documents · ${stats.chunks.toLocaleString()} chunks · ${provider}`;
  } catch (error) {
    $("corpus-status").textContent = "The API is not responding.";
  }
}

async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || "Request failed.");
  }
  return data;
}

$("ask-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  $("ask-error").hidden = true;
  $("answer").hidden = true;
  $("ask-loading").hidden = false;
  $("ask-button").disabled = true;
  try {
    const result = await api("/api/investigate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        chunk_strategy: $("strategy").value,
        retrieval: $("retrieval").value,
      }),
    });
    renderAnswer(result);
  } catch (error) {
    $("ask-error").hidden = false;
    $("ask-error").textContent = error.message;
  } finally {
    $("ask-loading").hidden = true;
    $("ask-button").disabled = false;
  }
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    $("question").value = chip.dataset.question;
  });
});

function renderAnswer(result) {
  const article = $("answer");
  article.hidden = false;
  article.className = "answer panel" + (result.abstained ? " abstain" : result.conflict ? " conflict" : "");
  const kicker = result.abstained ? "Not enough evidence" : result.conflict ? "Sources disagree" : "Answer";
  const pills = [
    `<span class="pill">${escapeHtml(result.chunk_strategy || "")}</span>`,
    `<span class="pill">${escapeHtml(result.retrieval || "")}</span>`,
    `<span class="pill">${result.latency_ms || 0} ms</span>`,
  ];
  if (result.cached) pills.push(`<span class="pill">cached</span>`);
  if (result.llm?.degraded) pills.push(`<span class="pill bad">model fallback</span>`);
  else if (result.llm?.provider) pills.push(`<span class="pill good">${escapeHtml(result.llm.provider)}</span>`);
  const body = escapeHtml(result.answer || "");
  const citations = (result.citations || [])
    .map((hit) => citeCard(hit))
    .join("");
  const conflicts = (result.contradictions || [])
    .map(
      (item) =>
        `<div class="cite"><strong>${escapeHtml(item.reason)}</strong><p>${escapeHtml(item.left_title)}: ${escapeHtml(item.left_excerpt)}</p><p>${escapeHtml(item.right_title)}: ${escapeHtml(item.right_excerpt)}</p></div>`
    )
    .join("");
  const steps = (result.steps || [])
    .map((step) => {
      const args = step.arguments ? escapeHtml(JSON.stringify(step.arguments)) : "";
      const extra = step.error ? ` · ${escapeHtml(step.error)}` : "";
      return `<div class="step"><strong>${escapeHtml(step.tool)} · ${escapeHtml(step.origin || "")}${extra}</strong><code>${args}</code></div>`;
    })
    .join("");
  article.innerHTML = `
    <p class="answer-kicker">${kicker}</p>
    <div class="prose">${body}</div>
    <div class="meta-row">${pills.join("")}</div>
    ${conflicts ? `<h3>Contradictions</h3><div class="cite-list">${conflicts}</div>` : ""}
    <h3>Citations</h3>
    <div class="cite-list">${citations || "<p class='muted'>No citation was attached.</p>"}</div>
    <h3>Tool trace</h3>
    <div class="trace">${steps || "<p class='muted'>No tools ran.</p>"}</div>
  `;
}

function citeCard(hit) {
  return `<div class="cite"><strong>${escapeHtml(hit.title || "Untitled")}</strong><code>${escapeHtml(hit.id)}</code><p>${escapeHtml(hit.text || "")}</p><p class="muted">${escapeHtml(hit.source || "")}</p></div>`;
}

$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData();
  const file = $("file").files[0];
  if (!file) return;
  data.append("file", file);
  data.append("force", $("force").checked ? "true" : "false");
  $("upload-message").textContent = "Indexing…";
  try {
    const result = await api("/api/documents", { method: "POST", body: data });
    $("upload-message").textContent = summarizeOutcomes(result.results);
    $("upload-form").reset();
    state.offset = 0;
    await loadDocuments();
    await refreshStatus();
  } catch (error) {
    $("upload-message").textContent = error.message;
  }
});

$("paste-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("paste-message").textContent = "Indexing…";
  try {
    const result = await api("/api/documents/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: $("paste-title").value,
        text: $("paste-text").value,
        source: "paste",
      }),
    });
    $("paste-message").textContent = summarizeOutcomes([result]);
    $("paste-form").reset();
    state.offset = 0;
    await loadDocuments();
    await refreshStatus();
  } catch (error) {
    $("paste-message").textContent = error.message;
  }
});

function summarizeOutcomes(results) {
  return (results || [])
    .map((item) => {
      if (item.duplicate) return `Duplicate of ${item.duplicate_of} (${item.reason}).`;
      if (item.created) return `Added ${item.document.title}.`;
      return `Updated ${item.document?.title || "document"}.`;
    })
    .join(" ");
}

$("doc-query").addEventListener("input", () => {
  state.query = $("doc-query").value.trim();
  state.offset = 0;
  loadDocuments();
});

$("prev-page").addEventListener("click", () => {
  state.offset = Math.max(0, state.offset - state.limit);
  loadDocuments();
});

$("next-page").addEventListener("click", () => {
  state.offset += state.limit;
  loadDocuments();
});

async function loadDocuments() {
  $("doc-error").hidden = true;
  try {
    const params = new URLSearchParams({
      limit: String(state.limit),
      offset: String(state.offset),
    });
    if (state.query) params.set("q", state.query);
    const page = await api(`/api/documents?${params}`);
    const list = $("doc-list");
    if (!page.documents.length) {
      list.innerHTML = `<div class="panel"><p>No documents on this page. Upload a file or build the HotpotQA corpus.</p></div>`;
    } else {
      list.innerHTML = page.documents
        .map(
          (doc) => `
          <div class="doc-row">
            <div>
              <strong>${escapeHtml(doc.title)}</strong>
              <div class="muted">${escapeHtml(doc.format)} · ${doc.chunk_count} chunks · ${escapeHtml(doc.source)}</div>
            </div>
            <button type="button" class="ghost" data-open="${doc.id}">Read</button>
            <button type="button" class="danger" data-delete="${doc.id}">Delete</button>
          </div>`
        )
        .join("");
    }
    const shown = Math.min(page.total, state.offset + page.documents.length);
    $("page-label").textContent = page.total ? `${state.offset + 1}–${shown} of ${page.total}` : "0 documents";
    $("prev-page").disabled = state.offset === 0;
    $("next-page").disabled = state.offset + state.limit >= page.total;
    list.querySelectorAll("[data-open]").forEach((button) => {
      button.addEventListener("click", () => openDocument(button.dataset.open));
    });
    list.querySelectorAll("[data-delete]").forEach((button) => {
      button.addEventListener("click", () => removeDocument(button.dataset.delete));
    });
  } catch (error) {
    $("doc-error").hidden = false;
    $("doc-error").textContent = error.message;
  }
}

async function openDocument(id) {
  const doc = await api(`/api/documents/${id}`);
  const detail = $("doc-detail");
  detail.hidden = false;
  detail.innerHTML = `<h3>${escapeHtml(doc.title)}</h3><p class="muted">${escapeHtml(doc.id)} · ${escapeHtml(doc.format)}</p><div>${escapeHtml(doc.text || "")}</div>`;
  detail.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function removeDocument(id) {
  if (!confirm("Delete this document and its chunks?")) return;
  try {
    await api(`/api/documents/${id}`, { method: "DELETE" });
    $("doc-detail").hidden = true;
    await loadDocuments();
    await refreshStatus();
  } catch (error) {
    $("doc-error").hidden = false;
    $("doc-error").textContent = error.message;
  }
}

async function loadEvaluation() {
  const body = $("eval-body");
  try {
    const report = await api("/api/evaluation");
    if (!report.available) {
      body.innerHTML = `<p>${escapeHtml(report.detail)}</p>`;
      return;
    }
    const rows = (report.configurations || [])
      .map((config) => {
        const m = config.metrics || {};
        return `<tr>
          <td>${escapeHtml(config.name)}</td>
          <td>${fmt(m.recall_at_5)}</td>
          <td>${fmt(m.mrr)}</td>
          <td>${fmt(m.answer_correctness)}</td>
          <td>${fmt(m.citation_accuracy)}</td>
        </tr>`;
      })
      .join("");
    body.innerHTML = `
      <p>${escapeHtml(report.summary || "")}</p>
      <table>
        <thead><tr><th>Configuration</th><th>Recall@5</th><th>MRR</th><th>Answer</th><th>Citations</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <p class="muted">Full write-up: EVALUATION_REPORT.md</p>
    `;
  } catch (error) {
    body.textContent = error.message;
  }
}

function fmt(value) {
  return typeof value === "number" ? value.toFixed(3) : "—";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

refreshStatus();
