"""
Causal Discovery Visualizer
----------------------------
Main Dash application.

Data flow
  dropdown change  ─┐
                     ├──► dcc.Store("dataset-store") ──► visualizations
  CSV upload       ─┘
"""

import base64
import io

import dash
import dash_cytoscape as cyto
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update

from causal import run_dummy_discovery, run_random_discovery
from chat import auto_summary, demo_response
from data import DATASETS, load_dataset

# ---------------------------------------------------------------------------
# Cytoscape stylesheet
# ---------------------------------------------------------------------------

CYTO_STYLESHEET = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "background-color": "#3b82f6",
            "border-color": "#1d4ed8",
            "border-width": "2px",
            "color": "#ffffff",
            "text-valign": "center",
            "text-halign": "center",
            "width": "72px",
            "height": "72px",
            "font-size": "13px",
            "font-weight": "bold",
            "shadow-blur": "10px",
            "shadow-color": "#93c5fd",
            "shadow-opacity": "0.5",
            "shadow-offset-x": "0px",
            "shadow-offset-y": "0px",
        },
    },
    {
        "selector": "edge",
        "style": {
            "width": "mapData(weight, 0, 1, 1.5, 6)",
            "line-color": "#94a3b8",
            "target-arrow-color": "#64748b",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "opacity": 0.85,
            "label": "data(weight_label)",
            "font-size": "10px",
            "color": "#475569",
            "text-rotation": "autorotate",
            "text-margin-y": "-8px",
            "text-background-color": "#f8fafc",
            "text-background-opacity": 0.85,
            "text-background-padding": "2px",
            "text-background-shape": "roundrectangle",
        },
    },
    {
        "selector": "node:selected",
        "style": {
            "background-color": "#f59e0b",
            "border-color": "#d97706",
            "shadow-color": "#fcd34d",
            "shadow-opacity": "0.7",
        },
    },
    {
        "selector": "edge:selected",
        "style": {
            "line-color": "#f59e0b",
            "target-arrow-color": "#f59e0b",
            "opacity": 1.0,
        },
    },
    {
        # Guaranteed (forced) edges are highlighted in green
        "selector": "edge[type = 'guaranteed']",
        "style": {
            "line-color": "#16a34a",
            "target-arrow-color": "#16a34a",
            "width": 4,
            "opacity": 1,
        },
    },
    {
        # Structural ambiguity: direction is undetermined (Markov-equivalent).
        # Shown as a bidirected orange edge so neither direction is implied.
        "selector": "edge[ambiguity = 'structural']",
        "style": {
            "line-color": "#f59e0b",
            "source-arrow-shape": "triangle",
            "source-arrow-color": "#f59e0b",
            "target-arrow-shape": "triangle",
            "target-arrow-color": "#f59e0b",
            "line-style": "dashed",
            "width": 2.5,
            "opacity": 0.9,
        },
    },
    {
        # Empirical ambiguity: confidence is below the user threshold.
        # Shown as a thin dashed grey edge — connection is uncertain.
        "selector": "edge[ambiguity = 'empirical']",
        "style": {
            "line-color": "#aaa",
            "target-arrow-color": "#aaa",
            "line-style": "dashed",
            "width": 1.5,
            "opacity": 0.55,
        },
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cyto_elements(edges):
    node_ids: set[str] = set()
    for e in edges:
        node_ids.add(e["source"])
        node_ids.add(e["target"])
    return (
        [{"data": {"id": n, "label": n}} for n in sorted(node_ids)]
        + [
            {
                "data": {
                    "id": f"{e['source']}__{e['target']}",
                    "source": e["source"],
                    "target": e["target"],
                    "weight": e["weight"],
                    "weight_label": f"{e['weight']:.2f}",
                    "type": e.get("type", "discovered"),
                    "ambiguity": e.get("ambiguity") or "",
                }
            }
            for e in edges
        ]
    )


def _timeseries_figure(df):
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(
            go.Scatter(y=df[col], name=col, mode="lines", line={"width": 1.5})
        )
    fig.update_layout(
        margin={"l": 40, "r": 10, "t": 20, "b": 40},
        xaxis_title="Time step",
        yaxis_title="Value",
        legend={"orientation": "h", "y": -0.22},
        template="plotly_white",
        hovermode="x unified",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    return fig


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, title="Causal Discovery Visualizer")
server = app.server  # Expose the Flask server for deployment

app.layout = html.Div(
    className="app-container",
    children=[
        # Shared state stores
        dcc.Store(id="dataset-store"),
        dcc.Store(id="constraints-store", data={"forbidden": [], "guaranteed": []}),
        dcc.Store(id="chat-store", data=[]),
        dcc.Store(id="_chat-scroll"),

        # ── Header ──────────────────────────────────────────────────────────
        html.Header(
            className="app-header",
            children=[
                html.H1("Causal Discovery Visualizer"),
                html.P("Discover and explore causal relationships in time-series data"),
            ],
        ),

        # ── Controls bar ────────────────────────────────────────────────────
        html.Div(
            className="controls-bar",
            children=[
                # Predefined dataset picker
                html.Div(
                    className="control-group",
                    children=[
                        html.Label("Dataset", className="control-label"),
                        dcc.Dropdown(
                            id="dataset-selector",
                            options=[
                                {"label": meta["label"], "value": key}
                                for key, meta in DATASETS.items()
                            ],
                            value=list(DATASETS.keys())[0],
                            clearable=False,
                            style={"minWidth": "240px"},
                        ),
                    ],
                ),

                # Divider + "or"
                html.Div(className="controls-divider"),
                html.Span("or", className="controls-or"),
                html.Div(className="controls-divider"),

                # Upload
                html.Div(
                    className="control-group",
                    children=[
                        dcc.Upload(
                            id="upload-data",
                            children=html.Div(
                                className="upload-label",
                                children=[
                                    html.Span("Upload CSV"),
                                    html.Span(" (drag & drop or click)", className="upload-hint"),
                                ],
                            ),
                            className="upload-area",
                            accept=".csv",
                            max_size=50 * 1024 * 1024,
                        ),
                        html.Div(id="upload-status", className="upload-status"),
                    ],
                ),

                # Active description (right-aligned)
                html.Div(id="dataset-description", className="dataset-description"),
            ],
        ),

        # ── Visualizations ──────────────────────────────────────────────────
        html.Div(
            className="viz-container",
            children=[
                # Time series panel (left)
                html.Div(
                    className="panel panel-timeseries",
                    children=[
                        html.H3("Time Series"),
                        dcc.Graph(
                            id="timeseries-plot",
                            config={"displayModeBar": False},
                            style={"height": "460px"},
                        ),
                    ],
                ),

                # Causal graph panel (center)
                html.Div(
                    className="panel panel-causal",
                    children=[
                        html.Div(
                            className="panel-causal-header",
                            children=[
                                html.H3("Discovered Causal Graph"),
                                html.Div(
                                    className="threshold-control",
                                    children=[
                                        html.Label(
                                            "Min. confidence",
                                            className="control-label",
                                            style={"whiteSpace": "nowrap"},
                                        ),
                                        dcc.Slider(
                                            id="confidence-threshold",
                                            min=0.0,
                                            max=1.0,
                                            step=0.05,
                                            value=0.3,
                                            marks={0: "0", 0.5: "0.5", 1: "1"},
                                            tooltip={"placement": "bottom", "always_visible": True},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            className="cyto-wrapper",
                            children=[
                                cyto.Cytoscape(
                                    id="causal-graph",
                                    layout={
                                        "name": "breadthfirst",
                                        "directed": True,
                                        "padding": 24,
                                        "spacingFactor": 1.4,
                                    },
                                    style={"width": "100%", "height": "420px", "background": "#f8fafc"},
                                    stylesheet=CYTO_STYLESHEET,
                                    elements=[],
                                    userZoomingEnabled=True,
                                    userPanningEnabled=True,
                                ),
                            ],
                        ),
                        html.Div(id="edge-info", className="edge-info"),
                    ],
                ),

                # Expert knowledge panel (right)
                html.Div(
                    className="panel expert-knowledge-panel",
                    children=[
                        html.H3("Expert Knowledge"),
                        html.Div(
                            className="constraint-form",
                            children=[
                                html.Label("Source", className="control-label"),
                                dcc.Dropdown(
                                    id="constraint-source",
                                    placeholder="Source variable…",
                                    clearable=True,
                                ),
                                html.Span("→", className="constraint-arrow"),
                                html.Label("Target", className="control-label"),
                                dcc.Dropdown(
                                    id="constraint-target",
                                    placeholder="Target variable…",
                                    clearable=True,
                                ),
                                dcc.RadioItems(
                                    id="constraint-type",
                                    options=[
                                        {"label": "Guaranteed", "value": "guaranteed"},
                                        {"label": "Impossible", "value": "forbidden"},
                                    ],
                                    value="guaranteed",
                                    inline=True,
                                    className="constraint-type-radio",
                                    inputStyle={"marginRight": "4px"},
                                    labelStyle={"marginRight": "16px"},
                                ),
                                html.Button(
                                    "Add constraint",
                                    id="add-constraint-btn",
                                    className="btn-add-constraint",
                                    n_clicks=0,
                                ),
                            ],
                        ),
                        html.Div(id="constraints-list", className="constraints-list"),
                    ],
                ),
            ],
        ),

        # ── AI Assistant chat ─────────────────────────────────────────────────
        html.Div(
            className="panel chat-panel",
            children=[
                html.Div(
                    className="chat-header",
                    children=[
                        html.H3("AI Assistant"),
                        html.Span("demo mode", className="chat-mode-badge"),
                    ],
                ),
                html.Div(id="chat-messages", className="chat-messages"),
                html.Div(
                    className="chat-input-row",
                    children=[
                        dcc.Input(
                            id="chat-input",
                            placeholder="Ask about the causal graph…",
                            type="text",
                            debounce=False,
                            n_submit=0,
                            value="",
                            className="chat-input",
                        ),
                        html.Button(
                            "Send",
                            id="chat-send-btn",
                            className="btn-chat-send",
                            n_clicks=0,
                        ),
                    ],
                ),
            ],
        ),
    ],
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("dataset-store", "data"),
    Input("dataset-selector", "value"),
)
def store_from_dropdown(dataset_key: str):
    df, true_edges = load_dataset(dataset_key)
    edges = run_dummy_discovery(df, true_edges)
    return {
        "df_json": df.to_json(orient="split"),
        "edges": edges,
        "description": DATASETS[dataset_key]["description"],
    }


@callback(
    Output("dataset-store", "data", allow_duplicate=True),
    Output("upload-status", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def store_from_upload(contents, filename):
    if contents is None:
        return no_update, no_update

    _type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    except Exception as exc:
        return no_update, html.Span(f"Error: {exc}", className="upload-error")

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return no_update, html.Span("No numeric columns found.", className="upload-error")

    edges = run_random_discovery(list(numeric_df.columns))
    store = {
        "df_json": numeric_df.to_json(orient="split"),
        "edges": edges,
        "description": (
            f"Uploaded: {filename}  —  {len(numeric_df):,} rows, {len(numeric_df.columns)} columns"
        ),
    }
    status = html.Span(
        f"✓ {filename}  ({len(numeric_df):,} rows, {len(numeric_df.columns)} columns)",
        className="upload-ok",
    )
    return store, status


@callback(
    Output("constraint-source", "options"),
    Output("constraint-target", "options"),
    Input("dataset-store", "data"),
)
def update_constraint_options(store_data):
    if not store_data:
        return [], []
    df = pd.read_json(io.StringIO(store_data["df_json"]), orient="split")
    opts = [{"label": c, "value": c} for c in df.columns]
    return opts, opts


@callback(
    Output("constraints-store", "data"),
    Input("add-constraint-btn", "n_clicks"),
    Input({"type": "remove-constraint-btn", "index": ALL}, "n_clicks"),
    Input("dataset-selector", "value"),
    Input("upload-data", "contents"),
    State("constraint-source", "value"),
    State("constraint-target", "value"),
    State("constraint-type", "value"),
    State("constraints-store", "data"),
    prevent_initial_call=True,
)
def manage_constraints(
    _add, _removes, _dataset_key, _upload,
    source, target, ctype, store,
):
    triggered_id = ctx.triggered_id
    empty = {"forbidden": [], "guaranteed": []}

    # Reset when the data source changes
    if triggered_id in ("dataset-selector", "upload-data"):
        return empty

    if store is None:
        store = empty

    # Add new constraint
    if triggered_id == "add-constraint-btn":
        if not source or not target or source == target:
            return store
        # Ignore duplicate in either category
        for cat in ("forbidden", "guaranteed"):
            for existing in store[cat]:
                if existing["source"] == source and existing["target"] == target:
                    return store
        return {**store, ctype: store[ctype] + [{"source": source, "target": target}]}

    # Remove an individual constraint chip
    if isinstance(triggered_id, dict) and triggered_id.get("type") == "remove-constraint-btn":
        cat, i_str = triggered_id["index"].rsplit("_", 1)
        i = int(i_str)
        return {**store, cat: [e for j, e in enumerate(store[cat]) if j != i]}

    return store


@callback(
    Output("constraints-list", "children"),
    Input("constraints-store", "data"),
)
def render_constraints(store):
    if not store:
        return []
    items = []
    for cat, label, css_class in [
        ("guaranteed", "Guaranteed", "chip-guaranteed"),
        ("forbidden",  "Impossible", "chip-forbidden"),
    ]:
        for i, e in enumerate(store[cat]):
            items.append(
                html.Div(
                    className="constraint-chip",
                    children=[
                        html.Span(
                            f"{e['source']} → {e['target']}",
                            className="constraint-chip-edge",
                        ),
                        html.Span(label, className=f"constraint-chip-label {css_class}"),
                        html.Button(
                            "×",
                            id={"type": "remove-constraint-btn", "index": f"{cat}_{i}"},
                            className="constraint-chip-remove",
                            n_clicks=0,
                        ),
                    ],
                )
            )
    if not items:
        return html.Span(
            "No constraints added yet.",
            style={"color": "#999", "fontStyle": "italic", "fontSize": "0.85rem"},
        )
    return items


@callback(
    Output("timeseries-plot", "figure"),
    Output("causal-graph", "elements"),
    Output("dataset-description", "children"),
    Input("dataset-store", "data"),
    Input("constraints-store", "data"),
    Input("confidence-threshold", "value"),
)
def update_visualizations(store_data, constraints, confidence_threshold):
    if store_data is None:
        return go.Figure(), [], ""

    df = pd.read_json(io.StringIO(store_data["df_json"]), orient="split")
    raw_edges = store_data["edges"]
    description = store_data.get("description", "")
    threshold = confidence_threshold if confidence_threshold is not None else 0.3

    # Apply expert-knowledge constraints
    forbidden_set = {
        (e["source"], e["target"]) for e in (constraints or {}).get("forbidden", [])
    }
    required_set = {
        (e["source"], e["target"]) for e in (constraints or {}).get("guaranteed", [])
    }

    edges = []
    for edge in raw_edges:
        key = (edge["source"], edge["target"])
        if key in forbidden_set:
            continue

        is_guaranteed = key in required_set
        edge_type = "guaranteed" if is_guaranteed else "discovered"

        if is_guaranteed:
            # Expert knowledge overrides any ambiguity
            ambiguity = None
        elif edge["weight"] < threshold:
            # Empirical ambiguity: confidence too low to trust this link
            ambiguity = "empirical"
        else:
            # Preserve structural ambiguity from discovery (direction unknown)
            ambiguity = edge.get("ambiguity")  # None or "structural"

        edges.append({**edge, "type": edge_type, "ambiguity": ambiguity})

    # Inject guaranteed edges not present in the raw discovery output
    existing_keys = {(e["source"], e["target"]) for e in edges}
    for req in (constraints or {}).get("guaranteed", []):
        key = (req["source"], req["target"])
        if key not in existing_keys:
            edges.append({
                "source": req["source"],
                "target": req["target"],
                "weight": 1.0,
                "type": "guaranteed",
                "ambiguity": None,
            })

    return _timeseries_figure(df), _build_cyto_elements(edges), description


@callback(
    Output("edge-info", "children"),
    Input("causal-graph", "tapEdgeData"),
)
def show_edge_info(edge_data):
    if not edge_data:
        return html.Span(
            "Click an edge to see details.",
            style={"color": "#888", "fontStyle": "italic"},
        )

    ambiguity = edge_data.get("ambiguity", "")
    ambiguity_labels = {
        "structural": ("Structural ambiguity", "#f59e0b"),
        "empirical":  ("Low confidence",       "#999"),
    }
    edge_type = edge_data.get("type", "discovered")

    parts = [
        html.Strong(f"{edge_data['source']}  →  {edge_data['target']}"),
        html.Span(f"  |  weight: {float(edge_data['weight']):.3f}"),
    ]
    if edge_type == "guaranteed":
        parts.append(html.Span(
            "  ·  Guaranteed",
            style={"color": "#16a34a", "fontWeight": "600"},
        ))
    if ambiguity in ambiguity_labels:
        label, color = ambiguity_labels[ambiguity]
        parts.append(html.Span(
            f"  ·  {label}",
            style={"color": color, "fontWeight": "600"},
        ))
    return html.Span(parts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Auto-scroll the chat window to the bottom whenever a message is added.
app.clientside_callback(
    """
    function(messages) {
        setTimeout(function () {
            var el = document.getElementById('chat-messages');
            if (el) el.scrollTop = el.scrollHeight;
        }, 40);
        return window.dash_clientside.no_update;
    }
    """,
    Output("_chat-scroll", "data"),
    Input("chat-store", "data"),
)


@callback(
    Output("chat-store", "data"),
    Input("dataset-store", "data"),
)
def chat_on_new_discovery(store_data):
    """Reset the chat and post an auto-summary whenever the graph changes."""
    if not store_data:
        return []
    summary = auto_summary(
        store_data.get("description", ""),
        store_data.get("edges", []),
    )
    return [{"role": "assistant", "content": summary}]


@callback(
    Output("chat-store", "data", allow_duplicate=True),
    Output("chat-input", "value"),
    Input("chat-send-btn", "n_clicks"),
    Input("chat-input", "n_submit"),
    State("chat-input", "value"),
    State("chat-store", "data"),
    State("dataset-store", "data"),
    State("confidence-threshold", "value"),
    prevent_initial_call=True,
)
def handle_user_message(_clicks, _submit, user_text, chat_data, store_data, threshold):
    if not user_text or not user_text.strip():
        return no_update, no_update

    user_text = user_text.strip()
    chat_data = chat_data or []

    # Reconstruct the currently-displayed edges so the response is context-aware
    raw_edges = (store_data or {}).get("edges", [])
    t = threshold if threshold is not None else 0.3
    edges = []
    for e in raw_edges:
        edge = dict(e)
        if edge.get("type") != "guaranteed" and edge["weight"] < t:
            edge["ambiguity"] = "empirical"
        edges.append(edge)

    response = demo_response(
        user_text,
        edges,
        (store_data or {}).get("description", ""),
    )
    new_history = chat_data + [
        {"role": "user",      "content": user_text},
        {"role": "assistant", "content": response},
    ]
    return new_history, ""


@callback(
    Output("chat-messages", "children"),
    Input("chat-store", "data"),
)
def render_chat(messages):
    if not messages:
        return html.Span(
            "Select a dataset to start the analysis.",
            style={"color": "#999", "fontStyle": "italic", "fontSize": "0.85rem",
                   "padding": "8px"},
        )
    items = []
    for msg in messages:
        if msg["role"] == "assistant":
            items.append(html.Div(
                className="chat-message chat-bot",
                children=[
                    html.Span("AI", className="chat-avatar chat-avatar-bot"),
                    dcc.Markdown(msg["content"], className="chat-content"),
                ],
            ))
        else:
            items.append(html.Div(
                className="chat-message chat-user",
                children=[
                    html.Span("You", className="chat-avatar chat-avatar-user"),
                    html.P(msg["content"], className="chat-content"),
                ],
            ))
    return items


if __name__ == "__main__":
    app.run(debug=False)
