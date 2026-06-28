from __future__ import annotations

from collections import Counter
from html import escape
from pathlib import Path
from statistics import mean

from wc_forecast.storage.database import DEFAULT_DATABASE_PATH, initialize_database
from wc_forecast.storage.prediction_store import (
    read_latest_prediction_ledger,
    read_prediction_ledger,
)


def build_dashboard_from_prediction_ledger(
    output_path: str | Path,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    latest_only: bool = True,
    limit: int = 100,
) -> list[dict[str, object]]:
    """Build a static HTML dashboard from the prediction ledger."""

    initialize_database(database_path)

    rows = (
        read_latest_prediction_ledger(database_path)
        if latest_only
        else read_prediction_ledger(database_path)
    )
    rows = rows[:limit]

    html = build_dashboard_html(rows, latest_only=latest_only)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")

    return rows


def build_dashboard_html(
    rows: list[dict[str, object]],
    latest_only: bool = True,
) -> str:
    """Build dashboard HTML for match and portfolio review."""

    summary = _portfolio_summary(rows)
    prediction_counts = Counter(str(row.get("predicted_outcome", "")) for row in rows)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>World Cup Forecasting Dashboard</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #1f2937;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --border: #374151;
      --accent: #93c5fd;
      --good: #86efac;
      --warn: #fde68a;
    }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui,
        -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    header {{
      margin-bottom: 28px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 32px;
      letter-spacing: -0.03em;
    }}
    h2 {{
      margin: 28px 0 12px;
      font-size: 20px;
    }}
    p {{
      color: var(--muted);
      margin: 0;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
    }}
    .card {{
      background: linear-gradient(180deg, var(--panel), var(--panel-2));
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 12px 32px rgba(0,0,0,.22);
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .08em;
      margin-bottom: 8px;
    }}
    .metric-value {{
      font-size: 26px;
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      font-size: 14px;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .08em;
      background: #0b1220;
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    .pill {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      background: #1e3a8a;
      color: #dbeafe;
      font-size: 12px;
      font-weight: 600;
    }}
    .muted {{
      color: var(--muted);
    }}
    .dist {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .dist span {{
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 8px 10px;
      color: var(--text);
      background: var(--panel);
    }}
    @media (max-width: 840px) {{
      .grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      table {{
        display: block;
        overflow-x: auto;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>World Cup Forecasting Dashboard</h1>
      <p>
        Static dashboard generated from the database-backed prediction ledger.
        Latest-only view: {escape(str(latest_only))}.
      </p>
    </header>

    <section class="grid">
      {_metric_card("Predictions", summary["prediction_count"])}
      {_metric_card("Open", summary["open_count"])}
      {_metric_card("Settled", summary["settled_count"])}
      {_metric_card("Avg confidence", _format_percent(summary["average_confidence"]))}
    </section>

    <h2>Portfolio view</h2>
    <div class="dist">
      {_prediction_distribution_html(prediction_counts)}
    </div>

    <h2>Match view</h2>
    {_match_table_html(rows)}
  </main>
</body>
</html>
"""


def _portfolio_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    confidences = [
        float(row["model_confidence"])
        for row in rows
        if row.get("model_confidence") not in {None, ""}
    ]

    statuses = Counter(str(row.get("status", "")).lower() for row in rows)

    return {
        "prediction_count": len(rows),
        "open_count": statuses.get("open", 0),
        "settled_count": statuses.get("settled", 0),
        "average_confidence": mean(confidences) if confidences else None,
    }


def _metric_card(label: str, value: object) -> str:
    return f"""
      <div class="card">
        <div class="metric-label">{escape(label)}</div>
        <div class="metric-value">{escape(str(value))}</div>
      </div>
    """


def _prediction_distribution_html(counts: Counter[str]) -> str:
    if not counts:
        return "<span>No predictions available</span>"

    return "\n".join(
        f"<span>{escape(outcome or 'unknown')}: {count}</span>"
        for outcome, count in sorted(counts.items())
    )


def _match_table_html(rows: list[dict[str, object]]) -> str:
    if not rows:
        return '<div class="card muted">No prediction rows available.</div>'

    body = "\n".join(_match_table_row(row) for row in rows)

    return f"""
    <table>
      <thead>
        <tr>
          <th>Date</th>
          <th>Match</th>
          <th>Prediction</th>
          <th>Confidence</th>
          <th>Probabilities</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {body}
      </tbody>
    </table>
    """


def _match_table_row(row: dict[str, object]) -> str:
    home = str(row.get("home_team", ""))
    away = str(row.get("away_team", ""))
    confidence = _format_percent(row.get("model_confidence"))

    probabilities = (
        f"H {_format_percent(row.get('prob_home_win'))} / "
        f"D {_format_percent(row.get('prob_draw'))} / "
        f"A {_format_percent(row.get('prob_away_win'))}"
    )

    return f"""
      <tr>
        <td>{escape(str(row.get("match_date", "")))}</td>
        <td>{escape(home)} vs {escape(away)}</td>
        <td><span class="pill">{escape(str(row.get("predicted_winner", "")))}</span></td>
        <td>{escape(confidence)}</td>
        <td>{escape(probabilities)}</td>
        <td>{escape(str(row.get("status", "")))}</td>
      </tr>
    """


def _format_percent(value: object) -> str:
    if value is None or value == "":
        return "n/a"

    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"
