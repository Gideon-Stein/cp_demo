"""
Chat module – demo response generation for the Causal Discovery Visualizer.

To integrate a real LLM, replace ``demo_response`` with an API call.
The function signature and the ``edges`` / ``description`` context are
designed to map directly onto a system-prompt + user-message pattern.
"""

from typing import Any, Dict, List

EdgeList = List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _suggest_intervention(edge: Dict, description: str) -> str:
    """
    Return a domain-aware intervention suggestion for a structurally ambiguous
    edge.  The do-calculus principle: P(tgt | do(src)) ≠ P(tgt) ⇒ src → tgt.
    """
    src, tgt = edge["source"], edge["target"]
    desc = description.lower()

    if any(k in desc for k in ("climate", "co2", "sunshine", "sea", "temperature")):
        setup = (
            f"artificially control **{src}** (e.g. via a climate-chamber experiment "
            f"or a carbon-capture pilot) while leaving **{tgt}** free to respond"
        )
    elif any(k in desc for k in ("gdp", "economic", "inflation", "unemployment")):
        setup = (
            f"apply a targeted fiscal or monetary policy that directly shifts "
            f"**{src}** (e.g. a stimulus package) without simultaneously acting on "
            f"**{tgt}**"
        )
    elif any(k in desc for k in ("region", "brain", "neuro")):
        setup = (
            f"apply transcranial magnetic stimulation (TMS) or optogenetics to "
            f"transiently suppress or activate **{src}** while recording "
            f"**{tgt}**"
        )
    elif any(k in desc for k in ("weather", "pressure", "wind")):
        setup = (
            f"fix **{src}** at different levels inside a meteorological simulation "
            f"and observe the resulting distribution of **{tgt}**"
        )
    else:
        setup = (
            f"directly manipulate **{src}** (hold it at a fixed value or apply an "
            f"external forcing) and observe whether **{tgt}** responds"
        )

    return (
        f"**Intervention to resolve {src} ↔ {tgt}**\n\n"
        f"Design an experiment that will {setup}.\n\n"
        f"| What you observe | Conclusion |\n"
        f"|---|---|\n"
        f"| **{tgt}** changes when **{src}** is manipulated | **{src} → {tgt}** ✓ |"
        f"\n"
        f"| **{tgt}** does not change | **{tgt} → {src}** (or no direct link) |\n\n"
        f"*Formally*: **P({tgt} \u2223 do({src})) ≠ P({tgt})** ⇒ {src} → {tgt}.  "
        f"This is the core idea behind randomised controlled trials in causal inference."
    )


def _suggest_data_collection(empirical_edges: EdgeList, description: str) -> str:
    """Return variable-specific advice for resolving low-confidence links."""
    if not empirical_edges:
        return (
            "No links are currently below the confidence threshold. "
            "Lower the slider to reveal weaker associations."
        )

    by_weight = sorted(empirical_edges, key=lambda e: e["weight"])
    # Unique variables, priority order (weakest edge first)
    variables: List[str] = list(dict.fromkeys(
        v for e in by_weight for v in (e["source"], e["target"])
    ))
    weakest = by_weight[0]
    desc = description.lower()

    if any(k in desc for k in ("climate", "co2", "sunshine", "sea")):
        instrument = "higher-resolution sensors (e.g. satellite radiometers or ocean buoys)"
    elif any(k in desc for k in ("gdp", "economic", "inflation", "unemployment")):
        instrument = "higher-frequency economic indicators (monthly instead of quarterly)"
    elif any(k in desc for k in ("region", "brain", "neuro")):
        instrument = "longer fMRI or EEG recording sessions with more trials per condition"
    elif any(k in desc for k in ("weather", "pressure", "wind")):
        instrument = "denser sensor networks and longer observation windows"
    else:
        instrument = "higher-frequency or higher-precision measurement equipment"

    edge_list = ", ".join(
        f"**{e['source']} → {e['target']}** (confidence {e['weight']:.2f})"
        for e in by_weight
    )
    lines = [
        f"**Data collection plan** — {len(by_weight)} uncertain link(s):\n"
        f"{edge_list}\n",
        "**Priority variables to instrument**: "
        + ", ".join(f"**{v}**" for v in variables[:4]),
        "",
        "Concrete steps:",
        f"1. **More samples** — statistical power scales with \u221an, so doubling "
        f"your observations roughly halves the standard error. "
        f"Start with **{weakest['source']}** and **{weakest['target']}** "
        f"(weakest link at {weakest['weight']:.2f}).",
        f"2. **Better instruments** — use {instrument} to reduce "
        f"measurement noise in the key variables.",
        "3. **Wider conditions** — collect data across different regimes "
        "(seasons, economic cycles, experimental settings) to increase "
        "the observed variance and improve identifiability.",
        "4. **Targeted experiment** — if a full intervention is feasible, "
        "that would resolve both empirical *and* structural ambiguity at once. "
        "Ask me *\"How do I intervene?\"* for details.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-summary (fired when a new graph is discovered)
# ---------------------------------------------------------------------------

def auto_summary(description: str, edges: EdgeList) -> str:
    """Return a brief analysis message when the dataset/graph changes."""
    if not edges:
        return (
            "No causal links were found for this dataset. "
            "Try uploading a different CSV or adjusting the discovery parameters."
        )

    n_total      = len(edges)
    structural   = [e for e in edges if e.get("ambiguity") == "structural"]
    n_certain    = n_total - len(structural)
    strongest    = max(edges, key=lambda e: e["weight"])

    lines = [
        f"**New causal graph ready.** *{description}*",
        "",
        f"Found **{n_total} link(s)**:",
    ]
    if n_certain > 0:
        lines.append(f"- **{n_certain}** with a determined direction")
    if structural:
        e = structural[0]
        lines.append(
            f"- **1** structurally ambiguous: "
            f"**{e['source']} \u2194 {e['target']}** (weight {e['weight']:.2f}) \u2014 "
            "direction cannot be resolved from the data alone"
        )
    lines += [
        "",
        f"Strongest link: **{strongest['source']} \u2192 {strongest['target']}** "
        f"(weight {strongest['weight']:.2f})",
        "",
        "Ask me:",
        "- *\"How do I resolve the structural ambiguity?\"* \u2014 get an intervention design",
        "- *\"What data should I collect?\"* \u2014 get a targeted measurement plan",
        "- *\"What are the strongest links?\"* \u2014 see the top-ranked edges",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo / template response (replace body with an LLM call when ready)
# ---------------------------------------------------------------------------

def demo_response(
    user_message: str,
    edges: EdgeList,
    description: str,
) -> str:
    """
    Generate a response to a user message (template-based demo).

    Parameters
    ----------
    user_message : Text typed by the user.
    edges        : Currently displayed edges, with ``ambiguity`` and ``type``
                   fields already applied by the visualisation layer.
    description  : Active dataset description string.

    Notes
    -----
    To switch to a real LLM replace everything below the dashed line with
    an API call, e.g.::

        import openai
        context = _build_context(edges, description)
        completion = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": context},
                {"role": "user",   "content": user_message},
            ],
        )
        return completion.choices[0].message.content
    """
    # ------------------------------------------------------------------
    msg = user_message.lower().strip()

    structural = [e for e in edges if e.get("ambiguity") == "structural"]
    empirical  = [e for e in edges if e.get("ambiguity") == "empirical"]
    guaranteed = [e for e in edges if e.get("type") == "guaranteed"]
    certain    = [e for e in edges
                  if not e.get("ambiguity") and e.get("type") != "guaranteed"]

    def _fmt(e: Dict) -> str:
        return f"**{e['source']} → {e['target']}** (weight {e['weight']:.2f})"

    # ── Intervention / resolve structural ambiguity ────────────────────
    if any(k in msg for k in (
        "intervene", "intervention", "experiment", "resolve", "fix direction",
        "which direction", "determine direction", "rct", "do calculus", "do("
    )):
        if structural:
            return _suggest_intervention(structural[0], description)
        return (
            "There are no structurally ambiguous edges right now, so no "
            "intervention is needed to determine directions. "
            "Lower the confidence threshold to see if any weak links surface."
        )

    # ── Data collection / resolve empirical ambiguity ──────────────────
    if any(k in msg for k in (
        "collect", "more data", "data collection", "gather", "sample",
        "sampling", "statistical power", "power", "instrument", "measure",
        "resolve empirical", "more observations"
    )):
        return _suggest_data_collection(empirical, description)

    # ── Structural ambiguity (explanation) ────────────────────────────
    if any(k in msg for k in (
        "structural", "direction", "bidirect", "undirected", "markov", "equivalent"
    )):
        if structural:
            e = structural[0]
            return (
                "**Structural ambiguity** arises from *Markov equivalence*: the "
                "observed distribution is compatible with multiple directed graphs "
                "that share the same skeleton but differ in edge orientation. "
                "The algorithm identifies *that* a link exists, but not *which way* "
                "it points.\n\n"
                f"Ambiguous edge: **{e['source']} \u2194 {e['target']}** "
                f"(weight {e['weight']:.2f})\n\n"
                "**Options to resolve it:**\n"
                "- Ask me *\"How do I resolve the structural ambiguity?\"* for a "
                "tailored intervention design\n"
                "- Use the **Expert Knowledge** panel to mark the direction as "
                "*Guaranteed* based on domain knowledge"
            )
        return (
            "No structurally ambiguous edges are visible right now. "
            "They appear as **orange bidirected dashes** when present."
        )

    # ── Empirical / low-confidence ─────────────────────────────────────
    if any(k in msg for k in (
        "empirical", "confidence", "noise", "threshold", "low confidence",
        "low", "dashed", "grey", "gray", "uncertain"
    )):
        if empirical:
            edge_list = ", ".join(_fmt(e) for e in empirical)
            return (
                "**Empirical ambiguity** (grey dashed edges) means the estimated "
                "effect is weaker than the confidence threshold \u2014 the signal may "
                "be real, but the data is too noisy to be certain.\n\n"
                f"Currently below threshold: {edge_list}.\n\n"
                "Ask me *\"What data should I collect?\"* for a targeted measurement "
                "plan, or raise the threshold slider to hide these links."
            )
        return (
            "No links are currently flagged as low-confidence. "
            "Lower the **confidence threshold** slider to reveal weaker associations."
        )

    # ── Strongest links ────────────────────────────────────────────────
    if any(k in msg for k in ("strong", "strongest", "best", "top", "highest")):
        if edges:
            top = sorted(edges, key=lambda e: e["weight"], reverse=True)[:3]
            lines = ["The **strongest links** are:"]
            for e in top:
                note = (" *(structural ambiguity)*"
                        if e.get("ambiguity") == "structural" else "")
                lines.append(f"- {_fmt(e)}{note}")
            return "\n".join(lines)

    # ── Weakest links ──────────────────────────────────────────────────
    if any(k in msg for k in ("weak", "weakest", "bottom", "lowest")):
        if edges:
            bottom = sorted(edges, key=lambda e: e["weight"])[:3]
            lines = ["The **weakest links** are:"]
            for e in bottom:
                note = (" *(low confidence)*"
                        if e.get("ambiguity") == "empirical" else "")
                lines.append(f"- {_fmt(e)}{note}")
            return "\n".join(lines)

    # ── Expert constraints ─────────────────────────────────────────────
    if any(k in msg for k in (
        "guaranteed", "expert", "constraint", "forced", "impossible", "forbid"
    )):
        if guaranteed:
            edge_list = ", ".join(_fmt(e) for e in guaranteed)
            return (
                f"You have **{len(guaranteed)} expert-guaranteed link(s)**: "
                f"{edge_list}.\n\n"
                "These green edges override the algorithm output — they are always "
                "included in the graph regardless of what the data says."
            )
        return (
            "No expert constraints are currently active. "
            "Use the **Expert Knowledge** panel to add *Guaranteed* or *Impossible* "
            "links based on your domain knowledge."
        )

    # ── Count / overview ───────────────────────────────────────────────
    if any(k in msg for k in (
        "how many", "count", "total", "summary", "overview", "all"
    )):
        return (
            f"The current graph has **{len(edges)} link(s)**:\n"
            f"- {len(certain)} certain (directed)\n"
            f"- {len(structural)} structurally ambiguous (orange, bidirected)\n"
            f"- {len(empirical)} below the confidence threshold (grey dashed)\n"
            f"- {len(guaranteed)} expert-guaranteed (green)"
        )

    # ── Help ───────────────────────────────────────────────────────────
    if any(k in msg for k in ("help", "what can", "commands", "example", "?")):
        return (
            "I can help you interpret the causal graph. Try:\n"
            "- *\"How do I resolve the structural ambiguity?\"* \u2014 intervention design\n"
            "- *\"What data should I collect?\"* \u2014 measurement plan\n"
            "- *\"What are the strongest links?\"*\n"
            "- *\"Which links are below the confidence threshold?\"*\n"
            "- *\"How many links are in the graph?\"*\n"
            "- *\"Tell me about [VariableName]\"*\n\n"
            "*(Demo mode — replace `chat.demo_response` with an LLM call to enable "
            "free-form reasoning.)*"
        )

    # ── Greeting ───────────────────────────────────────────────────────
    if any(k in msg for k in ("hi", "hello", "hey")):
        return (
            "Hello! I can help you interpret the causal discovery results. "
            "Ask me about specific links, ambiguities, or the graph structure. "
            "Type **help** for example questions."
        )

    # ── Variable-specific ──────────────────────────────────────────────
    mentioned = [
        e for e in edges
        if e["source"].lower() in msg or e["target"].lower() in msg
    ]
    if mentioned:
        lines = ["Here are the links involving the variables you mentioned:"]
        for e in mentioned:
            status = e.get("ambiguity") or e.get("type") or "discovered"
            lines.append(f"- {_fmt(e)} — *{status}*")
        return "\n".join(lines)

    # ── Fallback ───────────────────────────────────────────────────────
    return (
        f"The current graph has **{len(edges)} link(s)** "
        f"({len(structural)} structurally ambiguous, {len(empirical)} below threshold). "
        "Type **help** for example questions, or ask about specific variables."
    )
