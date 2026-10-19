"""
AI-COS — Main Streamlit Dashboard (app.py)
Phase 1 functional dashboard with 7 pages:
  1. Overview
  2. Document Library
  3. Document Processing
  4. Semantic Search
  5. Knowledge Graph Explorer
  6. Settings
  7. (stub) SIPA / ASCD — Phase 2 placeholders
"""

import logging
import os
import sys
from pathlib import Path

# ── Bootstrap: ensure project root is on the Python path ──────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Initialise DB ──────────────────────────────────────────────────────────────
from database.db import init_db
init_db()

# ── Imports ───────────────────────────────────────────────────────────────────
import streamlit as st
import json

from services.document_service import (
    list_all_documents,
    get_document_detail,
    get_version_detail,
    get_extracted_text,
    get_chunks_for_version,
    get_policy_statements_for_version,
    update_policy_statement,
    update_document_metadata,
    get_system_stats,
    run_semantic_search,
    delete_document_cascade,
)
from services.ingestion_service import ingest_document
from graph.graph_builder import (
    get_graph, get_graph_summary, get_neighbors, load_graph,
    add_dataset_relationships, save_graph,
)
from graph.graph_visualizer import build_plotly_figure
from config.settings import (
    CHUNK_SIZE, CHUNK_OVERLAP, SEMANTIC_TOP_K,
    SEMANTIC_SIMILARITY_THRESHOLD, GRAPH_SEMANTIC_EDGE_THRESHOLD,
    DATASET_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-COS | Cognitive Organizational System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stMarkdown h1 {
    font-size: 1.1rem; color: #58a6ff;
}

/* Main content */
.main .block-container { padding-top: 1rem; padding-bottom: 3rem; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e2130 0%, #252d3d 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #58a6ff; }
.metric-card .number {
    font-size: 2.2rem; font-weight: 700; color: #58a6ff; line-height: 1.1;
}
.metric-card .label {
    font-size: 0.78rem; color: #8b949e; text-transform: uppercase;
    letter-spacing: 0.05em; margin-top: 0.3rem;
}

/* Status badges */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em;
}
.badge-done    { background: #0f4c2c; color: #3fb950; border: 1px solid #238636; }
.badge-pending { background: #341a00; color: #e3b341; border: 1px solid #9e6a03; }
.badge-failed  { background: #3d0a0a; color: #f85149; border: 1px solid #da3633; }
.badge-processing { background: #0b2d44; color: #58a6ff; border: 1px solid #1f6feb; }

/* Section headers */
.section-header {
    font-size: 1.3rem; font-weight: 600; color: #e6edf3;
    border-left: 3px solid #58a6ff;
    padding-left: 0.75rem; margin-bottom: 1rem;
}

/* Modality chips */
.mod-permission   { color: #3fb950; font-weight: 600; }
.mod-obligation   { color: #f0883e; font-weight: 600; }
.mod-prohibition  { color: #f85149; font-weight: 600; }
.mod-recommendation { color: #a5d6ff; font-weight: 600; }

/* Stage pipeline */
.stage-ok     { color: #3fb950; }
.stage-failed { color: #f85149; }
.stage-partial { color: #e3b341; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Navigation
# ─────────────────────────────────────────────────────────────────────────────

PAGES = {
    "🏠 Overview": "overview",
    "📚 Document Library": "library",
    "⚙️ Document Processing": "processing",
    "🔍 Semantic Search": "search",
    "🕸️ Knowledge Graph": "graph",
    "📊 Evaluation": "evaluation",
    "⚙️ Settings": "settings",
}

with st.sidebar:
    st.markdown("## 🧠 AI-COS")
    st.markdown("*Cognitive Organizational System*")
    st.markdown("---")
    page_key = st.radio(
        "Navigation",
        list(PAGES.keys()),
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Phase 1** · Document Intelligence")
    st.caption("SIPA & ASCD available in Phase 2")

page = PAGES[page_key]


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _status_badge(status: str) -> str:
    cls = {
        "done": "badge-done",
        "pending": "badge-pending",
        "failed": "badge-failed",
        "processing": "badge-processing",
    }.get(status, "badge-pending")
    return f'<span class="badge {cls}">{status.upper()}</span>'


def _modality_chip(modality: str) -> str:
    cls = {
        "permission": "mod-permission",
        "obligation": "mod-obligation",
        "prohibition": "mod-prohibition",
        "recommendation": "mod-recommendation",
    }.get(modality or "", "")
    icon = {
        "permission": "✅",
        "obligation": "⚠️",
        "prohibition": "🚫",
        "recommendation": "💡",
    }.get(modality or "", "ℹ️")
    return f'<span class="{cls}">{icon} {(modality or "unknown").title()}</span>'


def _metric_card(number, label):
    return f"""
    <div class="metric-card">
        <div class="number">{number}</div>
        <div class="label">{label}</div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Overview
# ─────────────────────────────────────────────────────────────────────────────

if page == "overview":
    st.markdown('<h1 style="color:#58a6ff;margin-bottom:0.2rem">🧠 AI-COS Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e;font-size:0.9rem">Artificial Intelligence Cognitive Organizational System — Phase 1</p>', unsafe_allow_html=True)
    st.markdown("---")

    stats = get_system_stats()

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    cards = [
        (stats.get("total_documents", 0), "Documents"),
        (stats.get("total_versions", 0), "Versions"),
        (stats.get("total_policy_statements", 0), "Policy Statements"),
        (stats.get("total_embeddings", 0), "Embeddings"),
        (stats.get("graph_nodes", 0), "Graph Nodes"),
        (stats.get("graph_edges", 0), "Graph Edges"),
    ]
    for col, (num, label) in zip([c1, c2, c3, c4, c5, c6], cards):
        col.markdown(_metric_card(num, label), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Status & Graph summary ────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="section-header">Processing Status</div>', unsafe_allow_html=True)
        s_done = stats.get("done_versions", 0)
        s_pending = stats.get("pending_versions", 0)
        s_failed = stats.get("failed_versions", 0)
        st.markdown(f"""
        | Status | Count |
        |--------|-------|
        | ✅ Done | {s_done} |
        | ⏳ Pending | {s_pending} |
        | ❌ Failed | {s_failed} |
        """)

    with col_right:
        st.markdown('<div class="section-header">Graph Node Types</div>', unsafe_allow_html=True)
        node_types = stats.get("graph_node_types", {})
        if node_types:
            for nt, cnt in sorted(node_types.items(), key=lambda x: -x[1]):
                st.markdown(f"**{nt.replace('_',' ').title()}**: {cnt}")
        else:
            st.info("No graph data yet. Upload documents to build the knowledge graph.")

    st.markdown("---")
    st.markdown('<div class="section-header">Quick Actions</div>', unsafe_allow_html=True)
    qa1, qa2, qa3 = st.columns(3)
    with qa1:
        st.info("📤 **Upload Documents**\n\nGo to Document Library → Upload to add new policy documents.")
    with qa2:
        st.info("🔍 **Semantic Search**\n\nUse the Semantic Search page to find related passages across all documents.")
    with qa3:
        st.info("🕸️ **Explore Graph**\n\nVisit the Knowledge Graph page to explore document relationships.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Document Library
# ─────────────────────────────────────────────────────────────────────────────

elif page == "library":
    st.markdown('<div class="section-header">📚 Document Library</div>', unsafe_allow_html=True)

    tab_list, tab_upload = st.tabs(["📋 All Documents", "📤 Upload New Document"])

    # ── Document list ─────────────────────────────────────────────────────────
    with tab_list:
        docs = list_all_documents()
        if not docs:
            st.info("No documents yet. Use the Upload tab to add your first document.")
        else:
            # Filters
            fc1, fc2, fc3 = st.columns(3)
            depts = sorted(set(d.get("department") or "Unknown" for d in docs))
            dtypes = sorted(set(d.get("document_type") or "Unknown" for d in docs))
            with fc1:
                dept_filter = st.selectbox("Department", ["All"] + depts, key="lib_dept")
            with fc2:
                dtype_filter = st.selectbox("Document Type", ["All"] + dtypes, key="lib_dtype")
            with fc3:
                search_q = st.text_input("🔎 Search title / series code", key="lib_search")

            filtered = docs
            if dept_filter != "All":
                filtered = [d for d in filtered if d.get("department") == dept_filter]
            if dtype_filter != "All":
                filtered = [d for d in filtered if d.get("document_type") == dtype_filter]
            if search_q:
                sq = search_q.lower()
                filtered = [
                    d for d in filtered
                    if sq in (d.get("title") or "").lower()
                    or sq in (d.get("doc_series_code") or "").lower()
                ]

            st.caption(f"Showing {len(filtered)} of {len(docs)} documents")

            for doc in filtered:
                with st.expander(
                    f"{'[' + doc['doc_series_code'] + '] ' if doc.get('doc_series_code') else ''}"
                    f"{doc['title']}  —  {doc.get('department', '—')}  "
                    f"({doc['version_count']} version{'s' if doc['version_count'] != 1 else ''})",
                    expanded=False,
                ):
                    col_meta, col_actions = st.columns([3, 1])
                    with col_meta:
                        st.markdown(f"**Document ID:** `{doc['document_id']}`")
                        st.markdown(f"**Type:** {doc.get('document_type', '—')} | **Status:** {doc.get('latest_status', '—')}")
                        st.markdown(f"**Series:** {doc.get('doc_series_code', '—')} | **Created:** {doc.get('created_at', '—')}")

                    with col_actions:
                        if st.button("🔍 View Details", key=f"view_{doc['document_id']}"):
                            st.session_state["selected_doc_id"] = doc["document_id"]
                            st.rerun()

                    # If selected, show version list
                    if st.session_state.get("selected_doc_id") == doc["document_id"]:
                        detail = get_document_detail(doc["document_id"])
                        if detail:
                            st.markdown("**Versions:**")
                            for ver in detail["versions"]:
                                vc1, vc2, vc3 = st.columns([3, 2, 1])
                                with vc1:
                                    st.markdown(
                                        f"v{ver['version_label']} — `{ver['original_filename']}`  "
                                        f"({(ver.get('file_size_bytes') or 0) // 1024} KB)"
                                    )
                                with vc2:
                                    st.markdown(
                                        f"Chunks: **{ver.get('chunk_count') or 0}** | "
                                        f"Statements: **{ver.get('policy_statement_count') or 0}**"
                                    )
                                with vc3:
                                    st.markdown(
                                        _status_badge(ver.get("processing_status", "pending")),
                                        unsafe_allow_html=True,
                                    )
                                if st.button("📄 Inspect Version", key=f"inspect_{ver['version_id']}"):
                                    st.session_state["inspect_version_id"] = ver["version_id"]
                                    st.rerun()

                        # Metadata edit
                        with st.expander("✏️ Edit Metadata", expanded=False):
                            new_title = st.text_input("Title", value=doc["title"],
                                                       key=f"edit_title_{doc['document_id']}")
                            new_dept = st.text_input("Department",
                                                      value=doc.get("department") or "",
                                                      key=f"edit_dept_{doc['document_id']}")
                            new_auth = st.text_input("Issuing Authority",
                                                      value=doc.get("issuing_authority") or "",
                                                      key=f"edit_auth_{doc['document_id']}")
                            if st.button("💾 Save Metadata", key=f"save_meta_{doc['document_id']}"):
                                success = update_document_metadata(
                                    doc["document_id"],
                                    {"title": new_title, "department": new_dept,
                                     "issuing_authority": new_auth}
                                )
                                if success:
                                    st.success("Metadata updated.")
                                    st.rerun()
                                else:
                                    st.error("Could not update metadata.")

                        # Delete
                        with st.expander("🗑️ Delete Document", expanded=False):
                            st.warning(
                                "**Warning:** Deleting this document will permanently remove "
                                "all versions, extracted text, embeddings, and graph nodes. "
                                "This action cannot be undone."
                            )
                            confirm = st.checkbox("I understand and confirm deletion",
                                                   key=f"confirm_del_{doc['document_id']}")
                            if confirm and st.button("🗑️ Delete", key=f"del_{doc['document_id']}"):
                                result = delete_document_cascade(doc["document_id"])
                                if "error" in result:
                                    st.error(result["error"])
                                else:
                                    st.success(f"Deleted: {', '.join(result['deleted'])}")
                                    st.session_state.pop("selected_doc_id", None)
                                    st.rerun()

    # ── Upload tab ─────────────────────────────────────────────────────────────
    with tab_upload:
        st.markdown("Upload a PDF, DOCX, or TXT policy document.")

        uploaded = st.file_uploader(
            "Choose a file", type=["pdf", "docx", "txt"],
            key="uploader",
        )

        if uploaded:
            st.markdown("---")
            st.markdown("**📋 Document Metadata**")
            st.caption("Fill in what you know. Leave unknown fields blank.")

            mc1, mc2 = st.columns(2)
            with mc1:
                title_in = st.text_input("Document Title *",
                                          value=Path(uploaded.name).stem.replace("_", " "),
                                          key="up_title")
                dept_in = st.text_input("Department", key="up_dept")
                dtype_in = st.selectbox(
                    "Document Type",
                    ["Policy", "Regulation", "Manual", "Guideline", "Handbook", "Other"],
                    key="up_dtype",
                )
                auth_in = st.text_input("Issuing Authority", key="up_auth")
            with mc2:
                code_in = st.text_input("Series Code (e.g. HR-001)", key="up_code")
                ver_in = st.text_input("Version Label", value="1.0", key="up_ver")
                eff_in = st.text_input("Effective Date (YYYY-MM-DD)", key="up_eff")
                pub_in = st.text_input("Publication Date (YYYY-MM-DD)", key="up_pub")

            attach_to = st.selectbox(
                "Attach to existing document series?",
                ["— Create new document —"] + [
                    f"{d['doc_series_code'] or d['title']} (id={d['document_id'][:8]})"
                    for d in list_all_documents()
                ],
                key="up_attach",
            )
            existing_doc_id = None
            if attach_to != "— Create new document —":
                # Extract document_id from the label
                existing_doc_id = attach_to.split("id=")[-1].rstrip(")")
                # Find full id
                for d in list_all_documents():
                    if d["document_id"].startswith(existing_doc_id):
                        existing_doc_id = d["document_id"]
                        break

            if st.button("🚀 Upload & Process", type="primary", key="btn_upload"):
                if not title_in.strip():
                    st.error("Document title is required.")
                else:
                    metadata = {
                        "title": title_in.strip(),
                        "department": dept_in.strip() or None,
                        "document_type": dtype_in,
                        "issuing_authority": auth_in.strip() or None,
                        "doc_series_code": code_in.strip() or None,
                        "version_label": ver_in.strip() or "1.0",
                        "effective_date": eff_in.strip() or None,
                        "publication_date": pub_in.strip() or None,
                    }

                    with st.spinner("Processing document — this may take a minute…"):
                        report = ingest_document(
                            file_bytes=uploaded.read(),
                            original_filename=uploaded.name,
                            metadata=metadata,
                            document_id=existing_doc_id,
                        )

                    # ── Processing report ──────────────────────────────────────
                    st.markdown("---")
                    st.markdown("### 📋 Processing Report")

                    if report["success"]:
                        st.success("✅ Document ingested successfully!")
                    else:
                        st.error("❌ Ingestion completed with errors.")

                    st.markdown(f"**Document ID:** `{report.get('document_id')}`")
                    st.markdown(f"**Version ID:** `{report.get('version_id')}`")

                    # Stages
                    st.markdown("**Pipeline Stages:**")
                    for stage, status in report.get("stages", {}).items():
                        icon = "✅" if str(status) == "ok" else ("⚠️" if "partial" in str(status) else "❌")
                        st.markdown(f"- {icon} **{stage.replace('_', ' ').title()}**: {status}")

                    # Stats
                    stats_r = report.get("stats", {})
                    if stats_r:
                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("Chars Extracted", stats_r.get("extracted_chars", 0))
                        s2.metric("Chunks", stats_r.get("chunk_count", 0))
                        s3.metric("Policy Statements", stats_r.get("policy_statement_count", 0))
                        s4.metric("Embeddings", stats_r.get("embedding_count", 0))

                    if report["warnings"]:
                        with st.expander(f"⚠️ {len(report['warnings'])} Warning(s)"):
                            for w in report["warnings"]:
                                st.warning(w)

                    if report["errors"]:
                        with st.expander(f"❌ {len(report['errors'])} Error(s)"):
                            for e in report["errors"]:
                                st.error(e)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Document Processing (Inspection)
# ─────────────────────────────────────────────────────────────────────────────

elif page == "processing":
    st.markdown('<div class="section-header">⚙️ Document Processing</div>', unsafe_allow_html=True)

    docs = list_all_documents()
    if not docs:
        st.info("No documents yet. Upload documents from the Document Library.")
        st.stop()

    doc_options = {
        f"{d.get('doc_series_code') or d['title']} — {d['title']}": d["document_id"]
        for d in docs
    }
    selected_doc_label = st.selectbox("Select Document", list(doc_options.keys()), key="proc_doc")
    selected_doc_id = doc_options[selected_doc_label]

    detail = get_document_detail(selected_doc_id)
    if not detail or not detail["versions"]:
        st.warning("No versions found for this document.")
        st.stop()

    ver_options = {
        f"v{v['version_label']} — {v['original_filename']} [{v['processing_status']}]": v["version_id"]
        for v in detail["versions"]
    }
    selected_ver_label = st.selectbox("Select Version", list(ver_options.keys()), key="proc_ver")
    selected_ver_id = ver_options[selected_ver_label]

    ver_detail = get_version_detail(selected_ver_id)
    if not ver_detail:
        st.error("Version details not found.")
        st.stop()

    st.markdown("---")

    # ── Stats ─────────────────────────────────────────────────────────────────
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Chars", ver_detail.get("extracted_text_length") or 0)
    p2.metric("Chunks", ver_detail.get("chunk_count") or 0)
    p3.metric("Statements", ver_detail.get("policy_statement_count") or 0)
    p4.metric("Embeddings", ver_detail.get("embedding_count") or 0)
    p5.metric("Graph Nodes+", ver_detail.get("graph_nodes_added") or 0)

    st.markdown(
        f"**Status:** {_status_badge(ver_detail.get('processing_status', 'pending'))}",
        unsafe_allow_html=True,
    )

    # ── Processing log ────────────────────────────────────────────────────────
    proc_log = ver_detail.get("processing_log", [])
    if proc_log:
        with st.expander("📋 Processing Log"):
            for entry in proc_log:
                ts = entry.get("ts", "")
                msg = entry.get("msg", "")
                st.caption(f"`{ts}` {msg}")

    tab_text, tab_chunks, tab_statements = st.tabs(
        ["📄 Extracted Text", "🧩 Chunks", "📜 Policy Statements"]
    )

    with tab_text:
        text = get_extracted_text(selected_ver_id)
        if text:
            st.text_area("Cleaned extracted text", value=text, height=400, disabled=True,
                          key="proc_text_display")
            st.caption(f"{len(text):,} characters")
        else:
            st.info("No extracted text available.")

    with tab_chunks:
        chunks = get_chunks_for_version(selected_ver_id)
        if not chunks:
            st.info("No chunks found.")
        else:
            st.caption(f"{len(chunks)} chunks")
            for ch in chunks[:50]:  # show first 50
                with st.expander(
                    f"Chunk {ch['chunk_index']+1} "
                    f"{'· p.' + str(ch['page_number']) if ch.get('page_number') else ''} "
                    f"{'· §' + ch['section_heading'][:40] if ch.get('section_heading') else ''}",
                    expanded=False,
                ):
                    st.text(ch["text"])
            if len(chunks) > 50:
                st.caption(f"… and {len(chunks) - 50} more chunks")

    with tab_statements:
        stmts = get_policy_statements_for_version(selected_ver_id)
        if not stmts:
            st.info("No policy statements extracted.")
        else:
            st.caption(f"{len(stmts)} policy statements extracted")

            mod_filter = st.multiselect(
                "Filter by modality",
                ["permission", "obligation", "prohibition", "recommendation"],
                key="stmt_mod_filter",
            )
            filtered_stmts = stmts
            if mod_filter:
                filtered_stmts = [s for s in stmts if s.get("modality") in mod_filter]

            for stmt in filtered_stmts:
                with st.expander(
                    f"{_modality_chip(stmt.get('modality', ''))} "
                    f"{stmt['original_text'][:80]}…",
                    expanded=False,
                ):
                    st.markdown(f"**Text:** {stmt['original_text']}")
                    st.markdown(
                        f"**Modality:** {_modality_chip(stmt.get('modality', ''))} | "
                        f"**Actor:** {stmt.get('actor') or 'unknown'} | "
                        f"**Page:** {stmt.get('source_page') or '—'} | "
                        f"**Section:** {stmt.get('source_section') or '—'}",
                        unsafe_allow_html=True,
                    )
                    if stmt.get("conditions"):
                        st.markdown(f"**Conditions:** {'; '.join(stmt['conditions'])}")
                    if stmt.get("exceptions"):
                        st.markdown(f"**Exceptions:** {'; '.join(stmt['exceptions'])}")
                    if stmt.get("temporal_constraints"):
                        st.markdown(f"**Temporal:** {json.dumps(stmt['temporal_constraints'])}")
                    if stmt.get("numerical_threshold"):
                        st.markdown(f"**Threshold:** {stmt['numerical_threshold']}")

                    st.caption(
                        f"Extraction: {stmt.get('extraction_method', 'rule_based')} | "
                        f"Verification: {stmt.get('verification_status', 'unverified')}"
                    )

                    # Inline correction
                    with st.expander("✏️ Correct This Statement", expanded=False):
                        new_actor = st.text_input(
                            "Actor", value=stmt.get("actor") or "",
                            key=f"edit_actor_{stmt['statement_id']}",
                        )
                        new_mod = st.selectbox(
                            "Modality",
                            ["permission", "obligation", "prohibition", "recommendation"],
                            index=["permission","obligation","prohibition","recommendation"]
                                  .index(stmt.get("modality") or "permission"),
                            key=f"edit_mod_{stmt['statement_id']}",
                        )
                        new_ver_status = st.selectbox(
                            "Verification Status",
                            ["unverified", "verified", "rejected"],
                            index=["unverified","verified","rejected"]
                                  .index(stmt.get("verification_status") or "unverified"),
                            key=f"edit_vs_{stmt['statement_id']}",
                        )
                        if st.button("💾 Save", key=f"save_stmt_{stmt['statement_id']}"):
                            ok = update_policy_statement(
                                stmt["statement_id"],
                                {"actor": new_actor, "modality": new_mod,
                                 "verification_status": new_ver_status},
                            )
                            if ok:
                                st.success("Statement updated.")
                                st.rerun()
                            else:
                                st.error("Update failed.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Semantic Search
# ─────────────────────────────────────────────────────────────────────────────

elif page == "search":
    st.markdown('<div class="section-header">🔍 Semantic Search</div>', unsafe_allow_html=True)
    st.caption(
        "Enter a natural-language query to find semantically related passages "
        "across all ingested documents."
    )

    docs = list_all_documents()
    if not docs:
        st.info("No documents yet. Upload documents to enable semantic search.")
        st.stop()

    query = st.text_input(
        "Search query",
        placeholder="e.g. Employees working from home requirements",
        key="sem_query",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        top_k = st.slider("Max results", 3, 30, SEMANTIC_TOP_K, key="sem_topk")
    with c2:
        threshold = st.slider("Min similarity", 0.0, 1.0, SEMANTIC_SIMILARITY_THRESHOLD,
                               step=0.05, key="sem_thresh")
    with c3:
        doc_filter_labels = ["All documents"] + [
            f"{d.get('doc_series_code') or d['title']} — {d['title']}" for d in docs
        ]
        doc_filter_sel = st.selectbox("Filter by document", doc_filter_labels, key="sem_docfilter")

    filter_vids = None
    if doc_filter_sel != "All documents":
        idx = doc_filter_labels.index(doc_filter_sel) - 1
        chosen_doc = docs[idx]
        det = get_document_detail(chosen_doc["document_id"])
        if det:
            filter_vids = [v["version_id"] for v in det["versions"]]

    if st.button("🔍 Search", type="primary", key="btn_search") and query.strip():
        with st.spinner("Searching…"):
            results = run_semantic_search(
                query=query.strip(),
                top_k=top_k,
                threshold=threshold,
                filter_version_ids=filter_vids,
            )

        if not results:
            st.warning(
                f"No results found above the {threshold:.2f} similarity threshold. "
                "Try a lower threshold or a different query."
            )
        else:
            st.success(f"Found {len(results)} result(s)")
            for i, res in enumerate(results, 1):
                sim_pct = int(res["similarity"] * 100)
                with st.container():
                    st.markdown(
                        f"**#{i}** — "
                        f"📄 **{res['document_title']}** (v{res['version_label']}) "
                        f"| Similarity: `{sim_pct}%`"
                    )
                    st.markdown(
                        f"> {res['source_text'][:500]}{'…' if len(res['source_text']) > 500 else ''}",
                    )
                    st.caption(
                        f"Type: {res.get('source_type', '—')} | "
                        f"Series: {res.get('doc_series_code', '—')} | "
                        f"Version ID: `{res['version_id'][:12]}…`"
                    )
                    st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — Knowledge Graph Explorer
# ─────────────────────────────────────────────────────────────────────────────

elif page == "graph":
    st.markdown('<div class="section-header">🕸️ Knowledge Graph Explorer</div>', unsafe_allow_html=True)

    g = get_graph()
    summary = get_graph_summary(g)

    col_stats1, col_stats2, col_stats3 = st.columns(3)
    col_stats1.metric("Total Nodes", summary["total_nodes"])
    col_stats2.metric("Total Edges", summary["total_edges"])
    col_stats3.metric("Node Types", len(summary["node_types"]))

    if g.number_of_nodes() == 0:
        st.info(
            "The knowledge graph is empty. Upload and process documents to populate it. "
            "You can also load the dataset relationships below."
        )

        if st.button("📂 Load Dataset Relationships", key="load_rels"):
            rel_path = DATASET_DIR / "relationships.json"
            if rel_path.exists():
                import json as _json
                rels = _json.loads(rel_path.read_text())
                g = get_graph()
                n = add_dataset_relationships(g, rels)
                save_graph(g)
                st.success(f"Loaded {n} relationships from dataset.")
                st.rerun()
            else:
                st.error(f"relationships.json not found at {rel_path}")
        st.stop()

    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────────────────────
    all_node_types = list(summary["node_types"].keys())
    fc1, fc2 = st.columns(2)
    with fc1:
        type_filter = st.multiselect(
            "Show node types", all_node_types, default=all_node_types, key="graph_types"
        )
    with fc2:
        node_search = st.text_input("🔎 Search node (title / ID)", key="graph_node_search")

    highlight = set()
    if node_search:
        for nid, attrs in g.nodes(data=True):
            if (node_search.lower() in nid.lower() or
                    node_search.lower() in str(attrs.get("title", "")).lower() or
                    node_search.lower() in str(attrs.get("doc_series_code", "")).lower()):
                highlight.add(nid)
        if highlight:
            st.info(f"Highlighted {len(highlight)} matching node(s).")
        else:
            st.warning("No nodes matched the search.")

    # ── Visualisation ─────────────────────────────────────────────────────────
    fig = build_plotly_figure(
        g,
        highlight_nodes=highlight if highlight else None,
        filter_node_types=type_filter if type_filter else None,
        title="AI-COS Knowledge Graph",
    )
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # ── Node detail ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Explore Node Neighbors**")
    all_node_ids = sorted(g.nodes())
    if all_node_ids:
        selected_node = st.selectbox("Select a node to explore", all_node_ids, key="graph_node_sel")
        depth = st.slider("Hop depth", 1, 3, 1, key="graph_depth")
        if st.button("🔍 Explore", key="graph_explore"):
            neighbors = get_neighbors(g, selected_node, depth=depth)
            node_attrs = dict(g.nodes[selected_node])
            st.markdown(f"**Node:** `{selected_node}`")
            for k, v in node_attrs.items():
                st.markdown(f"- **{k}**: {v}")

            if neighbors["edges"]:
                st.markdown(f"**{len(neighbors['edges'])} connected edge(s):**")
                for edge in neighbors["edges"][:30]:
                    st.markdown(
                        f"- `{edge['source']}` → `{edge['target']}` "
                        f"**[{edge.get('relationship', '?')}]** "
                        f"_(method: {edge.get('inference_method', '?')})_  "
                        f"{edge.get('evidence', '')[:120]}"
                    )
            else:
                st.info("No neighbors found within the selected depth.")

    # ── Relationship table ─────────────────────────────────────────────────────
    with st.expander("📋 All Graph Edges"):
        edge_data = [
            {
                "Source": u, "Target": v,
                "Relationship": d.get("relationship", "—"),
                "Method": d.get("inference_method", "—"),
                "Status": d.get("verification_status", "—"),
                "Evidence": str(d.get("evidence", ""))[:80],
            }
            for u, v, d in g.edges(data=True)
        ]
        if edge_data:
            import pandas as pd
            st.dataframe(pd.DataFrame(edge_data), use_container_width=True)
        else:
            st.info("No edges yet.")

    # ── Load dataset relationships button ─────────────────────────────────────
    with st.expander("📂 Load Dataset Relationships"):
        st.caption("Import pre-defined document relationships from the synthetic dataset.")
        if st.button("Import relationships.json", key="import_rels"):
            rel_path = DATASET_DIR / "relationships.json"
            if rel_path.exists():
                import json as _json
                rels = _json.loads(rel_path.read_text())
                n = add_dataset_relationships(g, rels)
                save_graph(g)
                st.success(f"Imported {n} relationships.")
                st.rerun()
            else:
                st.error(f"Not found: {rel_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — Evaluation (Phase 1 stub)
# ─────────────────────────────────────────────────────────────────────────────

elif page == "evaluation":
    st.markdown('<div class="section-header">📊 Evaluation</div>', unsafe_allow_html=True)
    st.info(
        "**Phase 1 Status:** Evaluation metrics for SIPA and ASCD are computed in Phase 2. "
        "The ground-truth dataset (SIPA scenarios and ASCD conflict cases) has been loaded below."
    )

    docs = list_all_documents()
    stats = get_system_stats()

    st.markdown("### Dataset Overview")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Documents in Library", stats.get("total_documents", 0))
    d2.metric("Versions Processed", stats.get("done_versions", 0))
    d3.metric("Policy Statements", stats.get("total_policy_statements", 0))
    d4.metric("Graph Edges", stats.get("graph_edges", 0))

    st.markdown("---")
    st.markdown("### Ground-Truth Scenarios (from dataset)")

    tab_sipa, tab_ascd = st.tabs(["SIPA Scenarios", "ASCD Conflict Cases"])

    with tab_sipa:
        sipa_dir = DATASET_DIR / "scenarios"
        if sipa_dir.exists():
            for f in sorted(sipa_dir.glob("SIPA-*.json")):
                import json as _json
                data = _json.loads(f.read_text())
                with st.expander(f"📋 {f.stem} — {data.get('change_type', '—')}"):
                    st.json(data)
        else:
            st.warning("SIPA scenarios directory not found.")

    with tab_ascd:
        ascd_dir = DATASET_DIR / "conflicts"
        if ascd_dir.exists():
            for f in sorted(ascd_dir.glob("ASCD-*.json")):
                import json as _json
                data = _json.loads(f.read_text())
                with st.expander(
                    f"⚔️ {f.stem} — {data.get('conflict_type', '—')} "
                    f"[severity: {data.get('severity', '—')}]"
                ):
                    st.json(data)
        else:
            st.warning("ASCD conflicts directory not found.")

    st.markdown("---")
    st.caption("Full evaluation metrics (precision, recall, F1) will be available in Phase 2.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 7 — Settings
# ─────────────────────────────────────────────────────────────────────────────

elif page == "settings":
    st.markdown('<div class="section-header">⚙️ Settings</div>', unsafe_allow_html=True)
    st.caption("Configuration is managed via environment variables (.env file). "
               "Values shown here are the current runtime configuration.")

    st.markdown("### Chunking")
    c1, c2 = st.columns(2)
    c1.metric("Chunk Size (chars)", CHUNK_SIZE)
    c2.metric("Chunk Overlap (chars)", CHUNK_OVERLAP)

    st.markdown("### Semantic Search")
    s1, s2 = st.columns(2)
    s1.metric("Default Top-K Results", SEMANTIC_TOP_K)
    s2.metric("Default Similarity Threshold", SEMANTIC_SIMILARITY_THRESHOLD)

    st.markdown("### Embedding Model")
    st.code(f"Model: sentence-transformers/all-MiniLM-L6-v2\nDimension: 384\nNormalisation: L2 (cosine via dot product)")

    st.markdown("### Knowledge Graph")
    st.metric("Semantic Edge Threshold", GRAPH_SEMANTIC_EDGE_THRESHOLD)

    st.markdown("### Supported File Types")
    st.code(".pdf  .docx  .txt")

    st.markdown("### Storage Paths")
    from config.settings import UPLOADS_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, GRAPHS_DIR
    st.code(f"""Uploads   : {UPLOADS_DIR}
Processed : {PROCESSED_DIR}
Embeddings: {EMBEDDINGS_DIR}
Graphs    : {GRAPHS_DIR}""")

    st.markdown("---")
    st.caption(
        "To change settings, edit `.env` (copy from `.env.example`) and restart the application."
    )
