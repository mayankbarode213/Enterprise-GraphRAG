// ============================================================
// Markdown Rendering Helper
// ============================================================
function formatMessageContent(content) {
    if (!content) return "";

    try {
        marked.setOptions({ breaks: true, gfm: true });
        let formatted = marked.parse(content);
        // Highlight graph path arrows
        formatted = formatted.replace(/(──.*?──▶)/g, '<span class="arrow-rel">$1</span>');
        return formatted;
    } catch (e) {
        console.error("Marked parsing error:", e);
        // Fallback: basic bold + arrows
        let formatted = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formatted = formatted.replace(/(──.*?──▶)/g, '<span class="arrow-rel">$1</span>');
        return formatted;
    }
}

// ============================================================
// Suggest Query Helper
// ============================================================
function suggestQuery(text) {
    document.getElementById("query-input").value = text;
    document.getElementById("query-input").focus();
}

// ============================================================
// Copy Cypher Code
// ============================================================
function copyCode() {
    const codeText = document.getElementById("cypher-block").innerText;
    navigator.clipboard.writeText(codeText).catch(() => {});
    const copyBtn = document.querySelector(".copy-btn");
    copyBtn.innerText = "Copied!";
    setTimeout(() => { copyBtn.innerText = "Copy"; }, 2000);
}

// ============================================================
// Toggle Telemetry Details Panel
// ============================================================
function toggleTelemetryDetails() {
    const panel = document.getElementById("panel-telemetry");
    panel.classList.toggle("details-open");
}

// Auto-open details when there is data to show
function openTelemetryDetails() {
    document.getElementById("panel-telemetry").classList.add("details-open");
}

// ============================================================
// Submit Query Handler
// ============================================================
async function submitQuery(event) {
    event.preventDefault();

    const queryInput = document.getElementById("query-input");
    const query = queryInput.value.trim();
    if (!query) return;

    queryInput.value = "";

    // Add user bubble
    addMessageBubble(query, "user");

    const mode = document.querySelector('input[name="search-mode"]:checked').value;

    // Toggle telemetry panel visibility based on mode
    const telemetryPanel = document.getElementById("panel-telemetry");
    if (mode === "compare") {
        telemetryPanel.style.display = "none";
    } else {
        telemetryPanel.style.display = "";
    }

    // Update badge
    const telemetryBadge = document.getElementById("telemetry-badge");
    telemetryBadge.innerText = mode === "compare" ? "Comparing Pipelines…" : "Executing Query…";
    telemetryBadge.classList.add("active");

    const submitBtn = document.getElementById("submit-btn");
    submitBtn.disabled = true;
    submitBtn.querySelector("span").innerText = "Thinking…";

    resetVisuals();

    showProcessingSpinner(mode);

    const useT2C = document.getElementById("t2c-checkbox")?.checked || false;
    console.log("[submitQuery] payload:", { query, mode, use_text2cypher: useT2C });

    try {
        const response = await fetch("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, mode, use_text2cypher: useT2C })
        });

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errText}`);
        }

        const data = await response.json();

        removeProcessingSpinner();

        if (data.compare) {
            const isGraphReject = data.graph?.tool_used === "reject";
            const isVectorReject = data.vector?.tool_used === "reject";

            if (isGraphReject || isVectorReject) {
                addMessageBubble(data.graph?.answer || data.vector?.answer, "assistant", "reject");
                renderTelemetry({
                    tool_used: "reject",
                    confidence: data.graph?.confidence ?? 1.0,
                    latency_ms: 0.0,
                    tokens_used: 0,
                    prompt_tokens: 0,
                    completion_tokens: 0
                });
                telemetryBadge.innerText = "Winner: N/A";
                telemetryBadge.classList.remove("active");
            } else {
                addComparisonBubble(data);
                telemetryBadge.innerText = `Winner: ${data.evaluation?.winner ?? "N/A"}`;
                telemetryBadge.classList.remove("active");
            }
        } else {
            addMessageBubble(data.answer, "assistant", data.tool_used);
            renderTelemetry(data);
        }

    } catch (err) {
        removeProcessingSpinner();
        console.error("Query failed:", err);
        addMessageBubble(`**Error:** ${err.message}`, "assistant error");
        telemetryBadge.innerText = "Error";
        telemetryBadge.classList.remove("active");
    } finally {
        submitBtn.disabled = false;
        submitBtn.querySelector("span").innerText = "Send";
    }
}

// ============================================================
// Comparison Bubble (side-by-side)
// ============================================================
function addComparisonBubble(data) {
    const chatConv = document.getElementById("chat-conversation");

    const msgDiv = document.createElement("div");
    msgDiv.className = "chat-message compare-mode";

    const gridDiv = document.createElement("div");
    gridDiv.className = "comparison-grid";

    // GraphRAG column
    const graphCol = document.createElement("div");
    graphCol.className = "comparison-col graph-col";
    graphCol.innerHTML = `
        <div class="col-header">
            <span class="col-title-graph">GraphRAG</span>
            <span class="col-meta-tag">Structured Traversal</span>
        </div>
        <div class="col-body">${formatMessageContent(data.graph?.answer ?? "No response.")}</div>
        <div class="col-stats">
            <div class="stat-item">
                <span class="stat-item-label">Latency</span>
                <span class="stat-item-value">${((data.graph?.latency_ms ?? 0) / 1000).toFixed(2)}s</span>
            </div>
            <div class="stat-item">
                <span class="stat-item-label">Tokens</span>
                <span class="stat-item-value">${data.graph?.tokens_used ?? "—"} <span style="font-size:0.75em;opacity:0.65;">(in:${data.graph?.prompt_tokens ?? 0}, out:${data.graph?.completion_tokens ?? 0})</span></span>
            </div>
            <div class="stat-item">
                <span class="stat-item-label">Confidence</span>
                <span class="stat-item-value">${((data.graph?.confidence ?? 0) * 100).toFixed(0)}%</span>
            </div>
        </div>
    `;

    // VectorRAG column
    const vectorCol = document.createElement("div");
    vectorCol.className = "comparison-col vector-col";
    vectorCol.innerHTML = `
        <div class="col-header">
            <span class="col-title-vector">VectorRAG</span>
            <span class="col-meta-tag">Semantic Search</span>
        </div>
        <div class="col-body">${formatMessageContent(data.vector?.answer ?? "No response.")}</div>
        <div class="col-stats">
            <div class="stat-item">
                <span class="stat-item-label">Latency</span>
                <span class="stat-item-value">${((data.vector?.latency_ms ?? 0) / 1000).toFixed(2)}s</span>
            </div>
            <div class="stat-item">
                <span class="stat-item-label">Tokens</span>
                <span class="stat-item-value">${data.vector?.tokens_used ?? "—"} <span style="font-size:0.75em;opacity:0.65;">(in:${data.vector?.prompt_tokens ?? 0}, out:${data.vector?.completion_tokens ?? 0})</span></span>
            </div>
            <div class="stat-item">
                <span class="stat-item-label">Confidence</span>
                <span class="stat-item-value">${((data.vector?.confidence ?? 0) * 100).toFixed(0)}%</span>
            </div>
        </div>
    `;

    gridDiv.appendChild(graphCol);
    gridDiv.appendChild(vectorCol);

    // Evaluation summary
    const summaryCard = document.createElement("div");
    summaryCard.className = "comparison-summary-card";

    const evaluation = data.evaluation ?? {};
    const gEval = evaluation.graph_eval ?? {};
    const vEval = evaluation.vector_eval ?? {};
    const graphReasonsHtml = (evaluation.graph_reasons ?? []).map(r => `<li>${r}</li>`).join("");
    const vectorReasonsHtml = (evaluation.vector_reasons ?? []).map(r => `<li>${r}</li>`).join("");

    summaryCard.innerHTML = `
        <div class="summary-winner-header">
            Winner: <span class="winner-badge">🏆 ${evaluation.winner ?? "N/A"}</span>
        </div>
        <div class="comparison-reasons-grid">
            <div class="reasons-box graph-reasons">
                <h4>GraphRAG Analysis</h4>
                <ul class="reasons-list">${graphReasonsHtml}</ul>
            </div>
            <div class="reasons-box vector-reasons">
                <h4>VectorRAG Analysis</h4>
                <ul class="reasons-list">${vectorReasonsHtml}</ul>
            </div>
        </div>
        <div>
            <h4 style="margin: 0.8rem 0 0.5rem; font-family: var(--font-title); font-size: 0.9rem;">Evaluation Metrics</h4>
            <table class="comparison-metrics-table">
                <thead>
                    <tr><th>Metric</th><th>GraphRAG</th><th>VectorRAG</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Correct Tool Choice</td>
                        <td>${gEval.correct_tool ? "✅ Yes" : "❌ No"}</td>
                        <td>${vEval.correct_tool ? "✅ Yes" : "❌ No"}</td>
                    </tr>
                    <tr>
                        <td>Answer Completeness</td>
                        <td>${gEval.completeness_score ?? "—"}/10</td>
                        <td>${vEval.completeness_score ?? "—"}/10</td>
                    </tr>
                    <tr>
                        <td>Multi-hop Support</td>
                        <td>${gEval.multihop_support ? "✅ Yes" : "❌ No"}</td>
                        <td>${vEval.multihop_support ? "✅ Yes" : "❌ No"}</td>
                    </tr>
                    <tr>
                        <td>Explainability</td>
                        <td>${gEval.explainability_score ?? "—"}/10</td>
                        <td>${vEval.explainability_score ?? "—"}/10</td>
                    </tr>
                    <tr>
                        <td>Confidence Score</td>
                        <td>${((data.graph?.confidence ?? 0) * 100).toFixed(0)}%</td>
                        <td>${((data.vector?.confidence ?? 0) * 100).toFixed(0)}%</td>
                    </tr>
                    <tr>
                        <td>Measured Latency</td>
                        <td>${((data.graph?.latency_ms ?? 0) / 1000).toFixed(2)} s</td>
                        <td>${((data.vector?.latency_ms ?? 0) / 1000).toFixed(2)} s</td>
                    </tr>
                    <tr>
                        <td>Tokens Used</td>
                        <td>${data.graph?.tokens_used ?? "—"} <span style="font-size:0.85em;opacity:0.65;">(in:${data.graph?.prompt_tokens ?? 0}, out:${data.graph?.completion_tokens ?? 0})</span></td>
                        <td>${data.vector?.tokens_used ?? "—"} <span style="font-size:0.85em;opacity:0.65;">(in:${data.vector?.prompt_tokens ?? 0}, out:${data.vector?.completion_tokens ?? 0})</span></td>
                    </tr>
                </tbody>
            </table>
        </div>
    `;

    msgDiv.appendChild(gridDiv);
    msgDiv.appendChild(summaryCard);
    chatConv.appendChild(msgDiv);
    msgDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ============================================================
// Append Message Bubble
// ============================================================
function addMessageBubble(text, sender, toolUsed = null) {
    const chatConv = document.getElementById("chat-conversation");

    const msgDiv = document.createElement("div");
    msgDiv.className = `chat-message ${sender}`;

    const bubbleDiv = document.createElement("div");
    bubbleDiv.className = "message-bubble";

    let headerHtml = "";
    if (sender === "assistant" && toolUsed) {
        const isGraph = toolUsed === "graph";
        const isReject = toolUsed === "reject";
        
        let titleClass = "col-title-vector";
        let titleText = "VectorRAG";
        let tagText = "Semantic Search";
        
        if (isGraph) {
            titleClass = "col-title-graph";
            titleText = "GraphRAG";
            tagText = "Structured Traversal";
        } else if (isReject) {
            titleClass = "col-title-reject";
            titleText = "System Guardrail";
            tagText = "Out of Domain";
        }

        headerHtml = `
            <div class="col-header message-header-${toolUsed}" style="margin-bottom: 0.75rem; border-bottom: 1px solid var(--border-color); padding-bottom: 0.4rem;">
                <span class="${titleClass}" style="font-family: var(--font-title); font-size: 1rem; font-weight: 700;">${titleText}</span>
                <span class="col-meta-tag">${tagText}</span>
            </div>
        `;
    }

    bubbleDiv.innerHTML = headerHtml + formatMessageContent(text);

    msgDiv.appendChild(bubbleDiv);
    chatConv.appendChild(msgDiv);
    
    if (sender === "user") {
        chatConv.scrollTop = chatConv.scrollHeight;
    } else {
        msgDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// ============================================================
// Reset Visual Routing Indicators
// ============================================================
function resetVisuals() {
    document.getElementById("route-graph").className = "route-chip graph";
    document.getElementById("route-vector").className = "route-chip vector";
    document.getElementById("pulse-ball").className = "pulse-ball";
    document.getElementById("code-container").style.display = "none";
}

// ============================================================
// Render Telemetry Strip + Details
// ============================================================
function renderTelemetry(data) {
    const telemetryBadge = document.getElementById("telemetry-badge");
    telemetryBadge.classList.remove("active");

    const isGraph = data.tool_used === "graph";
    const isReject = data.tool_used === "reject";

    if (isReject) {
        telemetryBadge.innerText = "Query Rejected";
    } else {
        telemetryBadge.innerText = isGraph ? "Graph Routed" : "Vector Routed";
    }

    // 1. Router chips
    const routeGraph  = document.getElementById("route-graph");
    const routeVector = document.getElementById("route-vector");
    const pulseBall   = document.getElementById("pulse-ball");

    if (isReject) {
        document.getElementById("routing-details").innerText =
            `Security Guardrail: Query rejected as out-of-domain (${(data.confidence * 100).toFixed(0)}% confidence). Bypassed vector and graph retrieval to save tokens and prevent hallucination.`;
    } else if (isGraph) {
        routeGraph.classList.add("active-graph");
        pulseBall.classList.add("active-to-graph");
        document.getElementById("routing-details").innerText =
            `Router selected GraphRAG with ${(data.confidence * 100).toFixed(0)}% confidence. Context retrieved via structured multi-hop graph traversal.`;
    } else {
        routeVector.classList.add("active-vector");
        pulseBall.classList.add("active-to-vector");
        document.getElementById("routing-details").innerText =
            `Router selected VectorRAG with ${(data.confidence * 100).toFixed(0)}% confidence. Context retrieved via semantic similarity chunk search.`;
    }

    // 2. Strip stats
    document.getElementById("stat-confidence").innerText = `${(data.confidence * 100).toFixed(0)}%`;
    document.getElementById("stat-latency").innerText    = `${(data.latency_ms / 1000).toFixed(2)}s`;
    document.getElementById("stat-tokens").innerHTML     = data.tokens_used ? `${data.tokens_used} <span style="font-size:0.75em;opacity:0.65;font-weight:normal;margin-left:2px;">(in:${data.prompt_tokens ?? 0}, out:${data.completion_tokens ?? 0})</span>` : "—";

    let depthVal  = "—";
    let intentVal = "—";
    let entityVal = "—";
    let cypherVal = "";

    if (isGraph && data.graph_result) {
        depthVal  = data.graph_result.depth_hops !== null ? `${data.graph_result.depth_hops} hops` : "—";
        intentVal = data.graph_result.intent      || "—";
        entityVal = data.graph_result.root_entity || "—";
        cypherVal = data.graph_result.cypher_used || "";
    } else if (!isGraph) {
        intentVal = "semantic_text_search";
        entityVal = "N/A (Document Match)";
    }

    document.getElementById("stat-depth").innerText    = depthVal;
    document.getElementById("detail-intent").innerText = intentVal;
    document.getElementById("detail-entity").innerText = entityVal;

    // 3. Cypher block
    const codeContainer = document.getElementById("code-container");
    if (cypherVal && cypherVal !== "N/A (Ambiguous Entity)") {
        document.getElementById("cypher-block").innerText = cypherVal;
        codeContainer.style.display = "block";
    } else {
        codeContainer.style.display = "none";
    }

    // 4. ReAct stepper
    const stepperSteps = document.getElementById("stepper-steps");
    stepperSteps.innerHTML = "";

    if (data.reasoning && data.reasoning.length > 0) {
        data.reasoning.forEach((step, idx) => {
            const stepDiv = document.createElement("div");
            stepDiv.className = "react-step completed";
            stepDiv.innerHTML = `
                <div class="react-step-dot"></div>
                <div class="react-step-title">
                    <span>Step ${idx + 1}</span>
                    <span class="react-step-action">${step.action}</span>
                </div>
                <div class="react-step-content">
                    <strong>Thought:</strong> ${step.thought}<br><br>
                    <strong>Observation:</strong> ${step.observation}
                </div>
            `;
            stepperSteps.appendChild(stepDiv);
        });
    } else {
        stepperSteps.innerHTML = '<div class="stepper-placeholder">No reasoning steps recorded for this query.</div>';
    }

    // Auto-open details when there's meaningful data
    if (isGraph && (cypherVal || (data.reasoning && data.reasoning.length > 0))) {
        openTelemetryDetails();
    }
}

// ============================================================
// Processing Spinner Helpers
// ============================================================
function showProcessingSpinner(mode) {
    const chatConv = document.getElementById("chat-conversation");

    const msgDiv = document.createElement("div");
    msgDiv.className = "chat-message assistant processing-spinner-message";
    msgDiv.id = "processing-spinner";

    const bubbleDiv = document.createElement("div");
    bubbleDiv.className = "message-bubble processing-bubble";

    let spinnerText = mode === "compare" ? "Comparing GraphRAG & VectorRAG..." : "Retrieving information...";

    bubbleDiv.innerHTML = `
        <div class="spinner-container">
            <div class="spinner-dot-pulse">
                <span class="pulse-dot"></span>
                <span class="pulse-dot"></span>
                <span class="pulse-dot"></span>
            </div>
            <span class="spinner-text">${spinnerText}</span>
        </div>
    `;

    msgDiv.appendChild(bubbleDiv);
    chatConv.appendChild(msgDiv);
    chatConv.scrollTop = chatConv.scrollHeight;
}

function removeProcessingSpinner() {
    const spinner = document.getElementById("processing-spinner");
    if (spinner) {
        spinner.remove();
    }
}

// ============================================================
// Render welcome message with markdown on page load
// ============================================================
function updateTelemetryPanelVisibility() {
    const checkedRadio = document.querySelector('input[name="search-mode"]:checked');
    const mode = checkedRadio ? checkedRadio.value : "auto";
    const telemetryPanel = document.getElementById("panel-telemetry");
    if (telemetryPanel) {
        if (mode === "compare") {
            telemetryPanel.style.display = "none";
        } else {
            telemetryPanel.style.display = "";
        }
    }
}

document.addEventListener("DOMContentLoaded", () => {
    // Re-render static welcome bubble with marked.js
    const welcomeBubble = document.querySelector(".chat-message.assistant .message-bubble");
    if (welcomeBubble) {
        welcomeBubble.innerHTML = formatMessageContent(welcomeBubble.textContent.trim());
    }

    // Run initial visibility check (handles page refresh caching the selected option)
    updateTelemetryPanelVisibility();

    // Set up radio button change listeners to dynamically toggle telemetry panel visibility
    document.querySelectorAll('input[name="search-mode"]').forEach(radio => {
        radio.addEventListener("change", () => {
            updateTelemetryPanelVisibility();
        });
    });
});
