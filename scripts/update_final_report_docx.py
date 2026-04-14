"""Rewrite Final Report.docx content for Inventory Optimization AI Agent project."""

from __future__ import annotations

from pathlib import Path

from docx import Document


PROJECT_TITLE = "Inventory Optimization AI Agent"


EXACT_REPLACEMENTS = {
    "AI-Powered Data Visualization Chatbot": PROJECT_TITLE,
    "Certified that this project report “AI-Powered Data Visualization Chatbot” is the bonafide record of work done by “Harsun Pranav - E0122043” who carried out the internship work under my supervision.": (
        "Certified that this project report \"Inventory Optimization AI Agent\" is the bonafide record "
        "of work done by \"Harsun Pranav - E0122043\" who carried out the internship work under my supervision."
    ),
    "LITERATURE REVIEW": "LITERATURE REVIEW",
    "PROPOSED METHODOLOGY": "PROPOSED METHODOLOGY",
    "SYSTEM IMPLEMENTATION": "SYSTEM IMPLEMENTATION",
    "RESULTS AND DISCUSSION": "RESULTS AND DISCUSSION",
    "CONCLUSION AND FUTURE WORK": "CONCLUSION AND FUTURE WORK",
}


LINE_UPDATES = {
    "The problem addressed in this project is the difficulty faced by users in generating data visualizations without programming knowledge. Manual chart creation is time-consuming and requires familiarity with tools such as Python or R, limiting accessibility.": (
        "The problem addressed in this project is the difficulty operations teams face in making reliable inventory decisions under demand variability, supplier lead-time shifts, and incomplete context. Manual spreadsheet workflows are slow and often inconsistent across planners."
    ),
    "The main aim of this project is to develop an AI-powered chatbot that can generate data visualizations from natural language queries. The system allows users to upload datasets and interact conversationally to produce charts automatically.": (
        "The main aim of this project is to develop an Inventory Optimization AI Agent that analyzes SKU-level stock data and produces advisory recommendations for replenishment and risk mitigation."
    ),
    "The project involves dataset input, natural language processing using a language model, generation of visualization configurations, and rendering interactive charts using visualization libraries. The system leverages modern AI models to interpret user queries and generate Plotly-based outputs.": (
        "The project integrates data loading, metric computation, rule evaluation, hybrid knowledge graph enrichment, and batched LLM explanation generation. LangGraph orchestrates the workflow and FastMCP exposes independently testable tools."
    ),
    "The dataset used consists of user-uploaded structured data such as CSV or Excel files.": (
        "The dataset used in implementation consists of structured inventory records with 25 synthetic SKUs across multiple categories and demand profiles."
    ),
    "The system is evaluated based on accuracy of generated visualizations, response time, and usability.": (
        "The system is evaluated on decision quality signals, output completeness, fallback reliability, and end-to-end latency under CPU-only constraints."
    ),
    "The main objective of this project is to design and implement an AI-powered chatbot for data visualization.": (
        "The main objective of this project is to design and implement an AI-powered decision-support agent for inventory optimization."
    ),
    "Develop a system that converts natural language queries into visualizations": (
        "Develop a system that converts inventory data into actionable SKU-level recommendations"
    ),
    "Integrate a Large Language Model for query understanding": (
        "Integrate a local LLM for plain-English explanations with strict fallback behavior"
    ),
    "Generate Plotly-based charts automatically": (
        "Generate rule-grounded reorder advisories with schema-validated JSON output"
    ),
    "Enable dataset upload and processing": "Enable local CSV/JSON ingestion and robust validation",
    "Evaluate system performance using usability and accuracy metrics": (
        "Evaluate system performance using latency, reliability, and explanation coverage metrics"
    ),
    "Supports structured datasets (CSV, Excel)": "Supports structured inventory datasets (CSV, JSON)",
    "Generates common visualizations (bar, line, pie charts)": "Calculates inventory metrics and risk status for each SKU",
    "Provides interactive chart outputs": "Produces structured advisory output and human-readable summary",
    "Web-based chatbot interface": "CLI-first workflow with optional Streamlit interface",
    "Limited to structured datasets": "Limited to structured inventory data and configured fields",
    "Accuracy depends on LLM interpretation": "Explanation quality depends on LLM availability and response quality",
    "May not handle highly complex queries": "Does not execute autonomous ordering; recommendations remain advisory",
    "Requires internet or local model setup": "Requires local runtime dependencies (Ollama optional, Docker optional)",
    "This project provides a user-friendly approach to data visualization, making analytics accessible to non-technical users.": (
        "This project provides a practical, transparent inventory decision-support workflow for planners and analysts operating with limited compute resources."
    ),
    "Students and researchers": "Operations planners and supply chain analysts",
    "Business analysts": "Procurement and inventory managers",
    "Organizations with non-technical stakeholders": "Organizations needing explainable local AI workflows",
    "The novelty lies in combining conversational AI with real-time visualization generation.": (
        "The novelty lies in combining LangGraph orchestration, FastMCP tools, hybrid graph context, and batched local LLM explanations under strict latency constraints."
    ),
    "Chapter 4: System Implementation": "Chapter 4: System Implementation",
    "Chapter 5: Results and Discussion": "Chapter 5: Results and Discussion",
    "Chapter 6: Conclusion and Future Work": "Chapter 6: Conclusion and Future Work",
    "The proposed system significantly reduces the time required to generate visualizations": (
        "The proposed system significantly reduces time required to produce consistent inventory recommendations"
    ),
    "It eliminates the need for coding, making it accessible to non-technical users": (
        "It standardizes metric computation and explanation generation for planning teams"
    ),
    "Interactive capabilities enhance user experience": (
        "Fallback logic and schema validation improve operational reliability"
    ),
    "Line Charts": "Critical SKU advisories",
    "Used for trend analysis": "Used to prioritize immediate replenishment review",
    "Example: Sales over time": "Example: SKU-003 and SKU-015 critical coverage",
    "Bar Charts": "Watch-list recommendations",
    "Used for comparisons": "Used for near-term planning actions",
    "Example: Revenue by region": "Example: SKUs with low urgency margins",
    "Pie Charts": "Overstock and healthy distribution",
    "Used for distribution": "Used to rebalance stocking strategies",
    "Example: Product category share": "Example: category-level status mix",
    "Figure 5.1: Line chart (Sales trend)": "Figure 5.1: Status distribution across SKUs",
    "Figure 5.2: Bar chart (Revenue comparison)": "Figure 5.2: Reorder urgency ranking",
    "Figure 5.3: Pie chart (Category distribution)": "Figure 5.3: Context source and fallback behavior",
}


def generic_transform(text: str) -> str:
    """Apply broad domain transformations for remaining lines."""
    out = text
    replacements = [
        ("AI-powered chatbot", "Inventory Optimization AI Agent"),
        ("data visualization chatbot", "inventory optimization agent"),
        ("data visualization", "inventory optimization"),
        ("visualizations", "inventory recommendations"),
        ("visualization", "inventory recommendation"),
        ("chart", "recommendation"),
        ("charts", "recommendations"),
        ("Plotly.js", "LangGraph output layer"),
        ("Plotly", "JSON schema output"),
        ("Next.js", "Python runtime"),
        ("React.js", "LangGraph orchestration"),
        ("Tailwind CSS", "CLI and optional Streamlit UI"),
        ("TypeScript", "Python"),
        ("Qwen 2.5 Coder", "llama3.2:1b"),
        ("query", "inventory request"),
        ("queries", "inventory requests"),
        ("dataset", "inventory dataset"),
        ("datasets", "inventory datasets"),
    ]
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def rewrite_docx(path: Path) -> None:
    """Rewrite report content in place while preserving structure and styles."""
    doc = Document(str(path))

    for paragraph in doc.paragraphs:
        original = paragraph.text.strip()
        if not original:
            continue

        if original in EXACT_REPLACEMENTS:
            paragraph.text = EXACT_REPLACEMENTS[original]
            continue

        if original in LINE_UPDATES:
            paragraph.text = LINE_UPDATES[original]
            continue

        paragraph.text = generic_transform(paragraph.text)

    doc.save(str(path))


if __name__ == "__main__":
    rewrite_docx(Path("Final Report.docx"))
