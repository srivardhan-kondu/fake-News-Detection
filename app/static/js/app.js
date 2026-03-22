/* ==========================================================================
   Hybrid Fake News Detector — Client-Side Application
   All dashboard data is fetched from JSON API and rendered dynamically.
   ========================================================================== */

(() => {
"use strict";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
};

async function api(url, opts = {}) {
    const res = await fetch(url, {
        headers: { "Content-Type": "application/json", ...opts.headers },
        ...opts,
    });
    if (res.status === 401 || res.status === 302) {
        window.location.href = "/login";
        return null;
    }
    const json = await res.json();
    if (!res.ok) throw new Error(json.error || "Server error");
    return json;
}

function showOverlay(msg) {
    const ov = $("#loading-overlay");
    if (ov) {
        const p = $("p", ov);
        if (p) p.textContent = msg || "Loading\u2026";
        ov.classList.add("visible");
    }
}
function hideOverlay() {
    const ov = $("#loading-overlay");
    if (ov) ov.classList.remove("visible");
}

function showToast(message, type = "success") {
    const stack = $(".flash-stack") || (() => {
        const s = document.createElement("section");
        s.className = "flash-stack";
        $(".page-content").prepend(s);
        return s;
    })();
    const el = document.createElement("div");
    el.className = `flash flash-${type}`;
    el.textContent = message;
    stack.prepend(el);
    setTimeout(() => el.remove(), 5000);
}

// ---------------------------------------------------------------------------
// Chart factory (reuse / destroy previous chart on same canvas)
// ---------------------------------------------------------------------------
const chartInstances = new Map();

function renderChart(canvas, type, title, labels, data) {
    if (!canvas || !labels.length) return;
    if (chartInstances.has(canvas)) chartInstances.get(canvas).destroy();
    const isDoughnut = type === "doughnut";
    const bgColors = isDoughnut
        ? ["#c44949", "#1f8a70", "#c36c3f"]
        : ["#c36c3f", "#df9a61", "#1f8a70", "#799f8c", "#ebb989", "#b2684b",
           "#c36c3f", "#df9a61", "#1f8a70", "#799f8c"];
    const inst = new Chart(canvas, {
        type,
        data: {
            labels,
            datasets: [{ label: title, data, borderRadius: 12, backgroundColor: bgColors }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 600, easing: "easeOutQuart" },
            plugins: { legend: { display: isDoughnut } },
            scales: isDoughnut ? {} : {
                y: { beginAtZero: true, grid: { color: "rgba(23,32,42,0.08)" } },
                x: { grid: { display: false } },
            },
        },
    });
    chartInstances.set(canvas, inst);
}

// ---------------------------------------------------------------------------
// Dashboard page logic
// ---------------------------------------------------------------------------
function initDashboard() {
    const form = $("#analyze-form");
    if (!form) return;               // not on dashboard page

    let currentHistory = [];

    // --- Load initial data in parallel ---
    Promise.all([
        api("/api/metrics").catch(() => null),
        api("/api/history").catch(() => []),
    ]).then(([metrics, history]) => {
        if (metrics) renderMetrics(metrics);
        currentHistory = history || [];
        renderHistoryList(currentHistory);
        renderDistribution(currentHistory);
        // show latest result if available
        if (currentHistory.length) renderResult(currentHistory[0]);
    }).finally(hideOverlay);

    // --- Form submission via API ---
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        clearErrors();

        const title = form.title.value.trim();
        const article_text = form.article_text.value.trim();
        const article_url = form.article_url.value.trim();

        if (!article_text && !article_url) {
            setError("article_text", "Provide article text or a valid URL.");
            setError("article_url", "Provide article text or a valid URL.");
            return;
        }

        const btn = $("button[type=submit]", form);
        btn.disabled = true;
        btn.textContent = "Analyzing\u2026";
        showOverlay("Running hybrid ML + DL analysis\u2026");

        try {
            const result = await api("/api/analyze", {
                method: "POST",
                body: JSON.stringify({ title, article_text, article_url }),
            });
            // prepend to history
            currentHistory.unshift(result);
            renderHistoryList(currentHistory);
            renderDistribution(currentHistory);
            renderResult(result);
            showToast("Analysis complete \u2014 scroll down for results.");
            form.reset();
        } catch (err) {
            showToast(err.message, "danger");
        } finally {
            btn.disabled = false;
            btn.textContent = "Analyze article";
            hideOverlay();
        }
    });

    // --- Click handler for history items (event delegation) ---
    $("#history-list").addEventListener("click", async (e) => {
        const item = e.target.closest("[data-id]");
        if (!item) return;
        e.preventDefault();
        const id = item.dataset.id;
        showOverlay("Loading report\u2026");
        try {
            const result = await api(`/api/submission/${id}`);
            renderResult(result);
        } catch (err) {
            showToast(err.message, "danger");
        } finally {
            hideOverlay();
        }
    });

    // -- helpers --
    function clearErrors() {
        $$("[data-error]", form).forEach(el => el.textContent = "");
    }
    function setError(name, msg) {
        const el = $(`[data-error="${name}"]`, form);
        if (el) el.textContent = msg;
    }

    // --- Render functions ---
    function renderMetrics(m) {
        const sec = $("#metrics-section");
        if (!sec) return;

        const details = m.model_details || [];
        const preproc = m.preprocessing_config || {};
        const df = m.decision_factors || {};

        sec.innerHTML = `
            <!-- Quick stats -->
            <div class="metrics-quick-grid fade-in visible">
                <div class="metric-card accent-card">
                    <span>Best ML model</span>
                    <strong>${esc((m.best_ml_model || "").replace(/_/g, " "))}</strong>
                </div>
                <div class="metric-card">
                    <span>Best ML F1</span>
                    <strong>${(m.performance.ml.f1_score * 100).toFixed(2)}%</strong>
                </div>
                <div class="metric-card">
                    <span>DL F1</span>
                    <strong>${(m.performance.dl.f1_score * 100).toFixed(2)}%</strong>
                </div>
                <div class="metric-card">
                    <span>Training set size</span>
                    <strong>${m.dataset_size.toLocaleString()}</strong>
                </div>
            </div>

            <!-- Models & architecture panel -->
            <section class="card models-panel fade-in visible">
                <div class="section-heading"><div>
                    <p class="eyebrow">Pipeline architecture</p>
                    <h2>Models, Parameters & Decision Factors</h2>
                </div></div>

                <div class="model-cards-grid">
                    ${details.map(d => {
                        const isBest = d.is_best_ml;
                        const met = d.metrics || {};
                        const params = d.hyperparameters || {};
                        const paramEntries = Object.entries(params);
                        return `
                        <article class="model-info-card ${isBest ? 'model-info-best' : ''}">
                            <div class="mic-head">
                                <div>
                                    <h3>${esc(d.display_name)}</h3>
                                    <span class="mic-type">${esc(d.model_type)}</span>
                                </div>
                                ${isBest ? '<span class="badge badge-best">Best ML</span>' : ''}
                            </div>
                            <p class="mic-desc">${esc(d.description)}</p>

                            <div class="mic-section">
                                <h4>Hyperparameters</h4>
                                <table class="param-table">
                                    <tbody>
                                        ${paramEntries.map(([k, v]) =>
                                            `<tr><td class="param-key">${esc(k)}</td><td class="param-val">${esc(String(v))}</td></tr>`
                                        ).join("")}
                                    </tbody>
                                </table>
                            </div>

                            <div class="mic-section">
                                <h4>Performance (held-out test set)</h4>
                                <div class="mic-metrics-row">
                                    ${met.accuracy != null ? `<div class="mic-metric"><span>Accuracy</span><strong>${(met.accuracy * 100).toFixed(2)}%</strong></div>` : ''}
                                    ${met.precision != null ? `<div class="mic-metric"><span>Precision</span><strong>${(met.precision * 100).toFixed(2)}%</strong></div>` : ''}
                                    ${met.recall != null ? `<div class="mic-metric"><span>Recall</span><strong>${(met.recall * 100).toFixed(2)}%</strong></div>` : ''}
                                    ${met.f1_score != null ? `<div class="mic-metric"><span>F1 Score</span><strong>${(met.f1_score * 100).toFixed(2)}%</strong></div>` : ''}
                                </div>
                            </div>
                        </article>`;
                    }).join("")}
                </div>

                <!-- Preprocessing pipeline -->
                <div class="info-panels-row">
                    <article class="sub-card info-sub-card">
                        <h3>Text Preprocessing Pipeline</h3>
                        <div class="preproc-meta">
                            <span class="chip">${esc(preproc.vectorizer || 'TfidfVectorizer')}</span>
                            <span class="chip">max_features = ${preproc.max_features || 4000}</span>
                            <span class="chip">${esc(preproc.ngram_range || '(1,2)')}</span>
                        </div>
                        <ol class="preproc-steps">
                            ${(preproc.steps || []).map(s => `<li>${esc(s)}</li>`).join("")}
                        </ol>
                    </article>

                    <!-- Ensemble decision factors -->
                    <article class="sub-card info-sub-card">
                        <h3>Ensemble Decision Factors</h3>
                        <table class="param-table">
                            <tbody>
                                <tr><td class="param-key">Method</td><td class="param-val">${esc(df.method || '')}</td></tr>
                                ${Object.entries(df.per_model_weights || {}).map(([name, w]) =>
                                    `<tr><td class="param-key">${esc(name.replace(/_/g, ' '))} weight</td><td class="param-val">${(w * 100).toFixed(2)}%</td></tr>`
                                ).join('')}
                                <tr><td class="param-key">Fallback rule</td><td class="param-val">${esc(df.fallback_rule || '')}</td></tr>
                                <tr><td class="param-key">Best model selection</td><td class="param-val">${esc(df.best_model_selection || '')}</td></tr>
                            </tbody>
                        </table>
                        <div class="weight-bar-visual">
                            <div class="wb-track">
                                ${Object.entries(df.per_model_weights || {}).map(([name, w], i) => {
                                    const colors = ['wb-ml', 'wb-dl', 'wb-nb', 'wb-bilstm'];
                                    return `<div class="wb-fill ${colors[i] || 'wb-ml'}" style="width:${(w * 100)}%">${esc(name.replace(/_/g, ' ').split(' ').map(x => x[0].toUpperCase()).join(''))} ${(w * 100).toFixed(1)}%</div>`;
                                }).join('')}
                            </div>
                        </div>
                    </article>
                </div>
            </section>`;
    }

    function renderResult(r) {
        const sec = $("#result-section");
        if (!sec || !r) return;
        const isFake = r.predicted_label === "Fake News";
        const pillClass = isFake ? "status-fake" : "status-real";
        const fakeTerms = (r.explanation.fake_supporting_terms || []);
        const realTerms = (r.explanation.real_supporting_terms || []);
        const insights = (r.explanation.insights || []);
        const wfKeys = Object.keys(r.charts.word_frequency || {});
        const wfVals = Object.values(r.charts.word_frequency || {});
        const mb = r.model_breakdown;

        sec.innerHTML = `
        <section class="card result-card fade-in visible">
            <div class="section-heading">
                <div>
                    <p class="eyebrow">Current result</p>
                    <h2>${esc(r.title)}</h2>
                </div>
                <div class="status-pill ${pillClass}">${esc(r.predicted_label)}</div>
            </div>

            <div class="result-grid">
                <div><p class="result-label">Confidence score</p><h3>${r.confidence_score.toFixed(2)}%</h3></div>
                <div><p class="result-label">Credibility score</p><h3>${r.credibility_score.toFixed(2)}%</h3></div>
                <div><p class="result-label">Selection strategy</p><h3>${esc(mb.selected_strategy)}</h3></div>
            </div>

            <div class="insight-grid">
                <article class="sub-card">
                    <h3>Words supporting <span class="status-fake-text">Fake</span> classification</h3>
                    <div class="chip-row">
                        ${fakeTerms.length
                            ? fakeTerms.map(t => `<span class="chip chip-fake">${esc(t)}</span>`).join("")
                            : '<span class="chip muted-chip">No fake-supporting terms identified</span>'}
                    </div>
                </article>
                <article class="sub-card">
                    <h3>Words supporting <span class="status-real-text">Real</span> classification</h3>
                    <div class="chip-row">
                        ${realTerms.length
                            ? realTerms.map(t => `<span class="chip chip-real">${esc(t)}</span>`).join("")
                            : '<span class="chip muted-chip">No real-supporting terms identified</span>'}
                    </div>
                </article>
            </div>
            <div class="insight-grid">
                <article class="sub-card full-width-card">
                    <h3>Explainable insights</h3>
                    <ul class="clean-list">
                        ${insights.map(i => `<li>${esc(i)}</li>`).join("")}
                    </ul>
                </article>
            </div>

            <article class="sub-card transparency-panel">
                <h3>Model Pipeline Transparency</h3>
                <p class="transparency-reason">${esc(mb.decision_reason || '')}</p>
                <div class="transparency-grid">
                    ${(mb.individual_predictions || []).map(m => {
                        const isFake = m.prediction === "Fake News";
                        const barColor = isFake ? "var(--danger)" : "var(--success)";
                        const badge = m.is_best_ml ? ' <span class="badge badge-best">Best ML</span>' : '';
                        const typeBadge = m.model_type === 'DL'
                            ? '<span class="badge badge-dl">Deep Learning</span>'
                            : '<span class="badge badge-ml">Machine Learning</span>';
                        return `
                        <div class="transparency-card ${isFake ? 'border-fake' : 'border-real'}">
                            <div class="tc-header">
                                <strong>${esc(m.model_name)}</strong>
                                <div class="tc-badges">${typeBadge}${badge}</div>
                            </div>
                            <div class="tc-bar-row">
                                <div class="tc-bar-track">
                                    <div class="tc-bar-fill" style="width:${m.probability_fake}%;background:${barColor}"></div>
                                </div>
                                <span class="tc-bar-label">${m.probability_fake.toFixed(1)}%</span>
                            </div>
                            <div class="tc-meta">
                                <span class="mini-pill ${isFake ? 'status-fake' : 'status-real'}">${esc(m.prediction)}</span>
                                <span class="tc-metric">F1 ${(m.f1_score * 100).toFixed(1)}%</span>
                                <span class="tc-metric">Acc ${(m.accuracy * 100).toFixed(1)}%</span>
                                <span class="tc-metric">Weight ${(m.ensemble_weight != null ? (m.ensemble_weight * 100).toFixed(1) + '%' : '—')}</span>
                            </div>
                        </div>`;
                    }).join("")}
                </div>
            </article>

            <div class="chart-grid">
                <article class="sub-card">
                    <div class="chart-header"><h3>Word frequency</h3><span class="chart-caption">Top processed terms</span></div>
                    <div class="chart-wrap">
                    <canvas id="chart-wordfreq" class="chart-canvas"></canvas>
                    </div>
                </article>
                <article class="sub-card">
                    <div class="chart-header"><h3>Model comparison</h3><span class="chart-caption">Fake news probability by model</span></div>
                    <div class="chart-wrap">
                    <canvas id="chart-models" class="chart-canvas"></canvas>
                    </div>
                </article>
            </div>

            <div class="report-actions">
                <a class="button button-secondary" href="/report/${r.id}/pdf">Export PDF</a>
                <a class="button button-secondary" href="/report/${r.id}/csv">Export CSV</a>
            </div>
        </section>`;

        // draw charts after DOM is in place
        requestAnimationFrame(() => {
            renderChart($("#chart-wordfreq"), "bar", "Word Frequency", wfKeys, wfVals);
            const modelLabels = (mb.individual_predictions || []).map(m => m.model_name).concat(["Hybrid"]);
            const modelVals = (mb.individual_predictions || []).map(m => m.probability_fake).concat([mb.hybrid_probability_fake]);
            renderChart($("#chart-models"), "bar", "Probability Comparison", modelLabels, modelVals);
        });

        // scroll smoothly to results
        sec.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function renderHistoryList(list) {
        const el = $("#history-list");
        if (!el) return;
        if (!list.length) {
            el.innerHTML = '<p class="empty-state">No analyses stored yet.</p>';
            return;
        }
        el.innerHTML = list.slice(0, 15).map(s => `
            <a class="history-item" href="#" data-id="${s.id}">
                <div>
                    <strong>${esc(s.title)}</strong>
                    <small>${esc(s.created_at)}</small>
                </div>
                <span class="mini-pill ${s.predicted_label === "Fake News" ? "status-fake" : "status-real"}">${esc(s.predicted_label)}</span>
            </a>`).join("");
    }

    function renderDistribution(list) {
        const canvas = $("#distribution-chart");
        if (!canvas) return;
        const fake = list.filter(s => s.predicted_label === "Fake News").length;
        const real = list.filter(s => s.predicted_label === "Real News").length;
        renderChart(canvas, "doughnut", "Prediction Distribution",
            ["Fake News", "Real News"], [fake, real]);
    }
}

// ---------------------------------------------------------------------------
// History page logic
// ---------------------------------------------------------------------------
function initHistoryPage() {
    const tbody = $("#history-table-body");
    if (!tbody) return;

    api("/api/history").then(list => {
        if (!list || !list.length) {
            tbody.innerHTML = '<tr><td colspan="6" class="empty-table">No reports available yet.</td></tr>';
            return;
        }
        tbody.innerHTML = list.map(s => `
            <tr>
                <td><a href="/dashboard?view=${s.id}">${esc(s.title)}</a></td>
                <td>${esc(s.source_type)}</td>
                <td>${esc(s.predicted_label)}</td>
                <td>${s.confidence_score.toFixed(2)}%</td>
                <td>${esc(s.created_at)}</td>
                <td class="actions-cell">
                    <a href="/report/${s.id}/pdf">PDF</a>
                    <a href="/report/${s.id}/csv">CSV</a>
                </td>
            </tr>`).join("");
    }).catch(() => {
        tbody.innerHTML = '<tr><td colspan="6" class="empty-table">Failed to load reports.</td></tr>';
    });
}

// ---------------------------------------------------------------------------
// Scroll-based fade-in for static cards (home / auth pages)
// ---------------------------------------------------------------------------
function initFadeIn() {
    const cards = $$(".card, .metric-card, .sub-card, .feature-card");
    if (!("IntersectionObserver" in window) || !cards.length) return;
    const obs = new IntersectionObserver((entries) => {
        entries.forEach(e => {
            if (e.isIntersecting) { e.target.classList.add("visible"); obs.unobserve(e.target); }
        });
    }, { threshold: 0.08 });
    cards.forEach(c => { c.classList.add("fade-in"); obs.observe(c); });
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
    initDashboard();
    initHistoryPage();
    initFadeIn();

    // Render any server-rendered Chart.js canvases (home page)
    $$(".chart-canvas[data-chart-labels]").forEach(canvas => {
        const labels = JSON.parse(canvas.dataset.chartLabels || "[]");
        const values = JSON.parse(canvas.dataset.chartValues || "[]");
        if (labels.length && values.length) {
            renderChart(canvas, canvas.dataset.chartType || "bar",
                canvas.dataset.chartTitle || "Analytics", labels, values);
        }
    });
});

})();