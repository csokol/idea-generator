"""Report generation — markdown output from scored opportunities."""

from __future__ import annotations

from insights.models import Opportunity


def generate_report(
    opportunities: list[Opportunity],
    goal: str,
    queries_explored: list[str],
    format: str = "md",
    token_usage: dict | None = None,
) -> str:
    lines = [
        f"# Insight Report: {goal}",
        "",
        f"**Opportunities found:** {len(opportunities)}",
        f"**Queries explored:** {len(queries_explored)}",
        "",
        "---",
        "",
    ]

    if not opportunities:
        lines.append("No opportunities met the minimum evidence threshold.")
        return "\n".join(lines)

    for i, opp in enumerate(opportunities, 1):
        lines.append(f"## {i}. {opp.title}")
        lines.append("")
        lines.append(f"**Score:** {opp.score:.2f} | **Confidence:** {opp.confidence} | "
                      f"**Evidence:** {opp.evidence_count} signals from {len(opp.unique_urls)} sources")
        lines.append("")
        lines.append(f"### Summary")
        lines.append(opp.summary)
        lines.append("")

        lines.append("### Key Signals")
        lines.append("")
        for s in opp.signals[:10]:
            lines.append(f"- **Pain:** {s.pain}")
            lines.append(f"  - Workaround: {s.workaround}")
            lines.append(f"  - Desired outcome: {s.desired_outcome}")
            lines.append(f"  - Segment: {s.segment} | Severity: {s.severity}/5 | WTP: {s.willingness_to_pay}/5")
            lines.append(f"  - Source: [{s.url}]({s.url})")
            lines.append("")

        if opp.evidence_count > 10:
            lines.append(f"*... and {opp.evidence_count - 10} more signals*")
            lines.append("")

        lines.append("### Metrics")
        lines.append(f"- Avg severity: {opp.avg_severity:.1f}/5")
        lines.append(f"- Avg willingness to pay: {opp.avg_willingness_to_pay:.1f}/5")
        lines.append(f"- Unique sources: {len(opp.unique_urls)}")
        lines.append("")

        if opp.market_context:
            lines.append("### Market Landscape")
            lines.append(opp.market_context)
            lines.append("")

        lines.append("---")
        lines.append("")

    if token_usage and token_usage.get("rows"):
        lines.append("## Token Usage & Cost")
        lines.append("")
        lines.append("| Provider/Model | Input Tokens | Output Tokens | Est. Cost |")
        lines.append("|---|---:|---:|---:|")
        for row in token_usage["rows"]:
            label = f"{row['provider']}/{row['model']}"
            lines.append(f"| {label} | {row['input_tokens']:,} | {row['output_tokens']:,} | ${row['cost']:.3f} |")
        lines.append(
            f"| **Total** | **{token_usage['total_input']:,}** "
            f"| **{token_usage['total_output']:,}** "
            f"| **${token_usage['total_cost']:.3f}** |"
        )
        lines.append("")

    lines.append("## Appendix: Queries Explored")
    lines.append("")
    for q in queries_explored:
        lines.append(f"- {q}")
    lines.append("")

    return "\n".join(lines)
