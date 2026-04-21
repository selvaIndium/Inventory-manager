const PptxGenJS = require("pptxgenjs");

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Inventory Agent";
pptx.company = "Inventory Manager";
pptx.subject = "Inventory Optimization AI Agent Architecture";
pptx.title = "Inventory Optimization AI Agent - Module Architecture";

const COLOR = {
  navy: "1E2761",
  ice: "CADCFC",
  white: "FFFFFF",
  slate: "2A3359",
  charcoal: "1C1F2A",
  light: "F4F7FC",
  accent: "00A6A6",
  warning: "F9B872",
  success: "57CC99",
  muted: "6E7BA6",
  mutedDark: "4F5F8A",
  textDark: "1B1F2B",
};

function addTitle(slide, title, subtitle, dark = false) {
  if (dark) {
    slide.background = { color: COLOR.navy };
  } else {
    slide.background = { color: COLOR.light };
  }
  slide.addText(title, {
    x: 0.6,
    y: 0.4,
    w: 12.2,
    h: 0.7,
    fontFace: "Cambria",
    bold: true,
    fontSize: 34,
    color: dark ? COLOR.white : COLOR.navy,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.6,
      y: 1.1,
      w: 12.2,
      h: 0.62,
      fontFace: "Calibri",
      fontSize: 16,
      italic: true,
      color: dark ? COLOR.ice : COLOR.muted,
    });
  }
}

function addFooter(slide, text, dark = false) {
  slide.addText(text, {
    x: 0.6,
    y: 6.85,
    w: 12.2,
    h: 0.25,
    fontFace: "Calibri",
    fontSize: 10,
    color: dark ? COLOR.ice : COLOR.mutedDark,
    align: "right",
  });
}

function addPill(slide, x, y, w, label, color) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h: 0.34,
    rectRadius: 0.08,
    fill: { color },
    line: { color, pt: 0 },
  });
  slide.addText(label, {
    x: x + 0.08,
    y: y + 0.06,
    w: w - 0.16,
    h: 0.22,
    fontFace: "Calibri",
    bold: true,
    fontSize: 11,
    color: COLOR.textDark,
    align: "center",
    valign: "mid",
  });
}

function addCard(slide, cfg) {
  const { x, y, w, h, title, body, fill = COLOR.white, titleColor = COLOR.navy } = cfg;
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.08,
    fill: { color: fill },
    line: { color: "D8DEED", pt: 1 },
    shadow: { type: "outer", color: "BFC9E6", blur: 2, angle: 45, distance: 2, opacity: 0.15 },
  });
  slide.addText(title, {
    x: x + 0.16,
    y: y + 0.12,
    w: w - 0.3,
    h: 0.3,
    fontFace: "Cambria",
    bold: true,
    fontSize: 16,
    color: titleColor,
    margin: 0,
  });
  slide.addText(body, {
    x: x + 0.16,
    y: y + 0.48,
    w: w - 0.3,
    h: h - 0.58,
    fontFace: "Calibri",
    fontSize: 12,
    color: COLOR.textDark,
    breakLine: true,
    margin: 0,
    valign: "top",
  });
}

// Slide 1
{
  const s = pptx.addSlide();
  addTitle(s, "Inventory Optimization AI Agent", "Module-wise architecture, flow design, tool orchestration, and reliability model", true);
  addPill(s, 0.6, 1.8, 2.0, "Local-first", COLOR.accent);
  addPill(s, 2.8, 1.8, 2.5, "Advisory Workflow", COLOR.success);
  addPill(s, 5.5, 1.8, 2.9, "Deterministic + Agentic", COLOR.warning);

  addCard(s, {
    x: 0.6,
    y: 2.4,
    w: 4.1,
    h: 3.9,
    title: "What the Project Does",
    body: "Analyzes SKU inventory health, computes reorder signals, enriches context via runtime NetworkX graph, applies rule logic, and generates explainable recommendations in Streamlit.\n\nAdvisory-only: no direct order placement.",
    fill: "F2F6FF",
    titleColor: COLOR.navy,
  });

  addCard(s, {
    x: 4.95,
    y: 2.4,
    w: 3.8,
    h: 1.8,
    title: "Primary Outcomes",
    body: "- Priority SKU queue\n- Plain-language actions\n- Traceable tool/flow logs\n- Fast + thinking execution modes",
    fill: "EDF7F7",
    titleColor: "0D5A5A",
  });

  addCard(s, {
    x: 8.95,
    y: 2.4,
    w: 3.8,
    h: 1.8,
    title: "Core Tech Stack",
    body: "Streamlit UI\nLangGraph state machine\nFastMCP tool interface\nNetworkX runtime graph\nOllama-hosted local LLM",
    fill: "FFF8EE",
    titleColor: "8A5B15",
  });

  addCard(s, {
    x: 4.95,
    y: 4.45,
    w: 7.8,
    h: 1.85,
    title: "Balanced Operating Model",
    body: "Fast mode prioritizes speed and deterministic reliability. Thinking mode adds graph enrichment and optional planner-driven agentic loops. The architecture keeps explainability and observability in every path.",
    fill: COLOR.white,
    titleColor: COLOR.slate,
  });

  addFooter(s, "Inventory Manager | Architecture Module Deck | 1/10", true);
}

// Slide 2
{
  const s = pptx.addSlide();
  addTitle(s, "Why This Architecture", "Design constraints that drove module boundaries and execution strategy", false);

  addCard(s, {
    x: 0.6,
    y: 1.75,
    w: 6.0,
    h: 4.9,
    title: "Operational Constraints",
    body: "1) Inventory decisions must remain reviewable and auditable.\n2) Input quality varies; pipeline must tolerate partial data.\n3) The system should run locally with minimal cloud dependence.\n4) Different users need either speed-first or depth-first analysis.\n5) Agent behavior must be observable, not opaque.",
  });

  addCard(s, {
    x: 6.9,
    y: 1.75,
    w: 5.9,
    h: 2.35,
    title: "Architectural Responses",
    body: "- LangGraph for explicit state transitions\n- MCP tools for typed, reusable operations\n- Runtime NetworkX graph for contextual enrichment\n- Deterministic + planner-based agentic routes\n- Structured diagnostics and trace exports",
    fill: "F2F6FF",
  });

  addCard(s, {
    x: 6.9,
    y: 4.35,
    w: 5.9,
    h: 2.3,
    title: "File References",
    body: "- `main.py`\n- `agent/graph.py`\n- `agent/state.py`\n- `ui/runner.py`\n- `ui/tabs.py`",
    fill: "EDF7F7",
  });

  addFooter(s, "Inventory Manager | Architecture Module Deck | 2/10", false);
}

// Slide 3
{
  const s = pptx.addSlide();
  addTitle(s, "End-to-End System Architecture", "How data, tools, and model outputs move through the platform", false);

  const blocks = [
    { x: 0.8, y: 2.0, w: 2.15, h: 1.0, t: "Streamlit UI\n`ui/tabs.py`", c: "EDF7F7" },
    { x: 3.15, y: 2.0, w: 2.2, h: 1.0, t: "Runner\n`ui/runner.py`", c: "F2F6FF" },
    { x: 5.55, y: 2.0, w: 2.6, h: 1.0, t: "LangGraph App\n`agent/graph.py`", c: "FFF8EE" },
    { x: 8.4, y: 2.0, w: 2.05, h: 1.0, t: "Nodes\n`agent/nodes/*`", c: "F7F2FF" },
    { x: 10.7, y: 2.0, w: 1.85, h: 1.0, t: "Output\nJSON + UI", c: "EDF7F7" },
  ];

  blocks.forEach((b) => {
    addCard(s, { x: b.x, y: b.y, w: b.w, h: b.h, title: "", body: b.t, fill: b.c, titleColor: COLOR.navy });
  });

  for (let i = 0; i < blocks.length - 1; i++) {
    s.addShape(pptx.ShapeType.chevron, {
      x: blocks[i].x + blocks[i].w + 0.04,
      y: 2.31,
      w: 0.22,
      h: 0.35,
      fill: { color: COLOR.muted },
      line: { color: COLOR.muted, pt: 0 },
    });
  }

  addCard(s, {
    x: 0.8,
    y: 3.35,
    w: 3.55,
    h: 2.85,
    title: "Tool Layer",
    body: "MCP tool server centralizes operational primitives:\n\n- `tools/server.py`\n- `tools/load_data.py`\n- `tools/calc_metrics.py`\n- `tools/query_graph.py`\n- `tools/fetch_rules.py`",
    fill: "F2F6FF",
  });

  addCard(s, {
    x: 4.6,
    y: 3.35,
    w: 2.45,
    h: 2.85,
    title: "Knowledge",
    body: "Runtime context modules:\n\n- `knowledge/networkx_graph.py`\n- `knowledge/cache_layer.py`",
    fill: "F7FAFF",
  });

  addCard(s, {
    x: 7.2,
    y: 3.35,
    w: 5.35,
    h: 2.85,
    title: "Observability Layer",
    body: "Every run emits structured execution telemetry:\n\n- flow timeline (`flow_events`)\n- tool invocation logs (`tool_call_logs`)\n- LLM batch logs (`llm_batch_events`)\n- graph runtime stats and warnings\n\nPrimary surfaces:\n- `ui/tabs.py` (Live Trace + Diagnostics)\n- `agent/logging_utils.py`",
    fill: "EDF7F7",
  });

  addFooter(s, "Inventory Manager | Architecture Module Deck | 3/10", false);
}

// Slide 4
{
  const s = pptx.addSlide();
  addTitle(s, "Module-Wise Project Split", "Ownership boundaries and why each layer exists", false);

  const modules = [
    ["UI Layer", "`ui/tabs.py`\n`ui/runner.py`\n`ui/preflight.py`", "Handles user configuration, run gating, trace display, and exports."],
    ["Agent Core", "`agent/graph.py`\n`agent/state.py`\n`agent/nodes/*`", "Owns execution graph, routing, state mutation, and output assembly."],
    ["Tooling Layer", "`tools/server.py`\n`tools/load_data.py`\n`tools/calc_metrics.py`", "Exposes typed operational functions via MCP contracts."],
    ["Knowledge Layer", "`knowledge/networkx_graph.py`\n`knowledge/cache_layer.py`", "Builds/query runtime graph context and stores in-memory cache artifacts."],
    ["Config + Runtime", "`config/thresholds.yaml`\n`main.py`", "Defines thresholds, modes, and top-level run initialization."],
    ["Quality Layer", "`tests/unit/*`\n`tests/integration/*`\n`tests/fallback/*`", "Prevents regressions across deterministic, agentic, and fallback paths."],
  ];

  let idx = 0;
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 3; c++) {
      const m = modules[idx++];
      const x = 0.7 + c * 4.1;
      const y = 1.75 + r * 2.55;
      addCard(s, {
        x,
        y,
        w: 3.85,
        h: 2.25,
        title: m[0],
        body: `${m[1]}\n\n${m[2]}`,
        fill: r === 0 ? "F7FAFF" : "F4F8FF",
      });
    }
  }

  addFooter(s, "Inventory Manager | Architecture Module Deck | 4/10", false);
}

// Slide 5
{
  const s = pptx.addSlide();
  addTitle(s, "Deterministic Flow", "Predictable execution backbone for stable recommendations", false);

  const steps = ["Load Data", "Compute Metrics", "Enrich Context*", "Apply Rules", "Generate Prompts", "Explain + Format"];
  steps.forEach((label, i) => {
    const x = 0.8 + i * 2.0;
    addPill(s, x, 2.0, 1.8, `${i + 1}. ${label}`, i % 2 === 0 ? COLOR.slate : COLOR.accent);
    if (i < steps.length - 1) {
      s.addShape(pptx.ShapeType.chevron, {
        x: x + 1.82,
        y: 2.08,
        w: 0.16,
        h: 0.18,
        fill: { color: COLOR.muted },
        line: { color: COLOR.muted, pt: 0 },
      });
    }
  });

  addCard(s, {
    x: 0.8,
    y: 2.55,
    w: 5.95,
    h: 2.8,
    title: "Core Deterministic Node Chain",
    body: "- `agent/nodes/load_data.py`\n- `agent/nodes/calculate_metrics.py`\n- `agent/nodes/enrich_context.py` (*thinking only)\n- `agent/nodes/apply_rules.py`\n- `agent/nodes/generate_recs.py`",
    fill: "F2F6FF",
  });

  addCard(s, {
    x: 6.95,
    y: 2.55,
    w: 5.85,
    h: 2.8,
    title: "Explanation + Output Stage",
    body: "- `agent/nodes/explain_llm.py`\n- `agent/nodes/template_explanation.py`\n- `agent/nodes/format_output.py`\n\nOutput includes summary, recommendations, diagnostics metadata, and trace-friendly event logs.",
    fill: "EDF7F7",
  });

  addCard(s, {
    x: 0.8,
    y: 5.55,
    w: 12.0,
    h: 1.25,
    title: "Why This Matters",
    body: "Deterministic mode guarantees operational consistency, explicit traceability, and robust fallback behavior when model output quality is inconsistent.",
    fill: COLOR.white,
  });

  addFooter(s, "Inventory Manager | Architecture Module Deck | 5/10", false);
}

// Slide 6
{
  const s = pptx.addSlide();
  addTitle(s, "Agentic Flow: Hybrid vs Full", "Planner loop orchestration and control boundaries", false);

  addCard(s, {
    x: 0.7,
    y: 1.75,
    w: 6.0,
    h: 2.35,
    title: "Hybrid Mode (thinking)",
    body: "Deterministic backbone runs first, then planner/executor loop can orchestrate additional tool actions before explanation.\n\nFiles: `agent/graph.py`, `agent/nodes/planner_action.py`, `agent/nodes/execute_action.py`",
    fill: "EDF7F7",
  });

  addCard(s, {
    x: 6.95,
    y: 1.75,
    w: 5.85,
    h: 2.35,
    title: "Full Mode (thinking)",
    body: "Planner-first route: agent selects tool sequence and exits when done. Contract checks ensure deterministic system does not silently take over full mode.",
    fill: "F2F6FF",
  });

  addCard(s, {
    x: 0.7,
    y: 4.35,
    w: 12.1,
    h: 2.35,
    title: "Planner Loop Mechanics",
    body: "Loop: `planner_action` -> `execute_action` -> repeat until done or step cap.\n\nGuardrails:\n- Action schema validation\n- Duplicate action suppression\n- Stage-aware tool validity checks\n- Explicit stop reason (`agent_fallback_reason`)",
    fill: COLOR.white,
  });

  addFooter(s, "Inventory Manager | Architecture Module Deck | 6/10", false);
}

// Slide 7
{
  const s = pptx.addSlide();
  addTitle(s, "LangGraph Agent Module", "State contract, routing functions, and node graph composition", false);

  addCard(s, {
    x: 0.7,
    y: 1.75,
    w: 4.0,
    h: 4.9,
    title: "State Contract",
    body: "`agent/state.py` defines typed runtime fields for:\n- records, metrics, contexts\n- tool logs and flow events\n- planner loop state\n- LLM responses and reasoning\n- runtime graph diagnostics",
    fill: "F2F6FF",
  });

  addCard(s, {
    x: 4.95,
    y: 1.75,
    w: 3.9,
    h: 4.9,
    title: "Routing Layer",
    body: "`agent/graph.py` routes by mode and agent_mode:\n\n- fast -> deterministic path\n- thinking+deterministic -> enriched deterministic path\n- thinking+hybrid/full -> planner loop branches\n\nRouters: `_route_from_mode`, `_route_after_metrics`, `_route_after_planner`.",
    fill: "EDF7F7",
  });

  addCard(s, {
    x: 9.1,
    y: 1.75,
    w: 3.7,
    h: 4.9,
    title: "Why LangGraph",
    body: "- Explicit control flow\n- Deterministic state transitions\n- Easier debugging than implicit agent loops\n- Native fit for mixed deterministic + agentic orchestration\n- Streamed progress events for UI trace panels",
    fill: "FFF8EE",
  });

  addFooter(s, "Inventory Manager | Architecture Module Deck | 7/10", false);
}

// Slide 8
{
  const s = pptx.addSlide();
  addTitle(s, "MCP Tools + NetworkX Runtime Graph", "How tool contracts and graph context connect in thinking mode", false);

  addCard(s, {
    x: 0.7,
    y: 1.75,
    w: 5.95,
    h: 2.55,
    title: "MCP Tool Contracts",
    body: "Registered in `tools/server.py`:\n- `load_inventory`\n- `calc_metrics_batch`\n- `query_graph_batch`\n- `apply_rules_batch`\n\nPlanner and deterministic nodes both consume the same tool primitives.",
    fill: "EDF7F7",
  });

  addCard(s, {
    x: 6.9,
    y: 1.75,
    w: 5.9,
    h: 2.55,
    title: "NetworkX Graph Design",
    body: "`knowledge/networkx_graph.py` builds runtime graph from uploaded rows:\n- SKU nodes\n- category nodes\n- optional supplier nodes\n\nEdges encode belongs-to and supply relationships; queries return enrichment context and risk tags.",
    fill: "F2F6FF",
  });

  addCard(s, {
    x: 0.7,
    y: 4.55,
    w: 12.1,
    h: 2.15,
    title: "Connection Model + Justification",
    body: "`tools/query_graph.py` enforces runtime graph usage (thinking mode) with in-memory cache from `knowledge/cache_layer.py`.\n\nWhy this choice now:\n- local-first operation\n- low infra overhead\n- transparent debug path\n- strong fit for per-run graph construction from uploaded datasets",
    fill: COLOR.white,
  });

  addFooter(s, "Inventory Manager | Architecture Module Deck | 8/10", false);
}

// Slide 9
{
  const s = pptx.addSlide();
  addTitle(s, "Fallbacks, Guardrails, and Observability", "Reliability controls for imperfect model behavior and real-world data quality", false);

  addCard(s, {
    x: 0.7,
    y: 1.75,
    w: 4.0,
    h: 4.95,
    title: "Action Guardrails",
    body: "`agent/nodes/explain_llm.py` enforces policy corrections:\n\n- healthy + zero reorder: no reorder action\n- critical/watch: reorder-oriented guidance\n- overstock: reduce replenishment guidance\n\nCorrections are explicitly logged.",
    fill: "FFF8EE",
  });

  addCard(s, {
    x: 4.95,
    y: 1.75,
    w: 4.0,
    h: 4.95,
    title: "Fallback Strategy",
    body: "- Planner unavailable -> safe stop reason\n- Missing/partial LLM outputs -> template completion\n- Deterministic backbone remains available\n\nCore files:\n`agent/nodes/planner_action.py`\n`agent/nodes/template_explanation.py`\n`agent/nodes/format_output.py`",
    fill: "F2F6FF",
  });

  addCard(s, {
    x: 9.2,
    y: 1.75,
    w: 3.6,
    h: 4.95,
    title: "Observability",
    body: "UI diagnostics (`ui/tabs.py`) expose:\n\n- flow timeline\n- tool caller/status logs\n- LLM batch events\n- runtime graph stats\n- raw CoT (experimental)\n\nTrace exports support post-run audits.",
    fill: "EDF7F7",
  });

  addFooter(s, "Inventory Manager | Architecture Module Deck | 9/10", false);
}

// Slide 10
{
  const s = pptx.addSlide();
  addTitle(s, "Conclusion and Forward Plan", "Architecture rationale, tradeoffs, and execution direction", true);

  s.addText(
    "This architecture is intentionally biased toward operational trust over opaque autonomy. LangGraph provides explicit, inspectable control flow; MCP tools keep business operations modular and testable; and the runtime NetworkX graph grounds thinking-mode recommendations in relationships inferred directly from uploaded inventory data. Together, these choices create a system that can be explained to planners and audited by engineering teams without relying on hidden orchestration behavior.",
    {
      x: 0.9,
      y: 1.85,
      w: 11.9,
      h: 1.8,
      color: COLOR.white,
      fontFace: "Calibri",
      fontSize: 16,
      margin: 0,
      valign: "top",
    }
  );

  s.addText(
    "The implemented mode model now separates speed and depth cleanly: fast mode runs deterministic analysis for responsiveness, while thinking mode adds graph enrichment and optional planner loops for deeper context. Reliability guardrails are no longer optional extras; they are part of the core path, including planner schema validation, duplicate-action suppression, and LLM action correction that prevents contradictory guidance such as reorder recommendations for healthy, zero-reorder SKUs.",
    {
      x: 0.9,
      y: 3.95,
      w: 11.9,
      h: 1.8,
      color: COLOR.ice,
      fontFace: "Calibri",
      fontSize: 16,
      margin: 0,
      valign: "top",
    }
  );

  s.addText(
    "The next phase should focus on stronger graph-derived risk features, tighter model profiling for planner quality, and deeper test coverage of guardrail and contract behaviors. Optional raw CoT exposure should remain explicitly marked as experimental and separated from final action logic, preserving explainability without allowing unvalidated reasoning text to drive operational recommendations. This keeps the system practical for real planning teams while remaining extensible for future intelligence upgrades.",
    {
      x: 0.9,
      y: 6.0,
      w: 11.9,
      h: 0.8,
      color: COLOR.white,
      fontFace: "Calibri",
      fontSize: 15,
      margin: 0,
      valign: "top",
    }
  );

  addFooter(s, "Inventory Manager | Architecture Module Deck | 10/10", true);
}

pptx.writeFile({ fileName: "results/Inventory_Agent_Module_Architecture_v2.pptx" });
