from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go


COLORS = {
    "teal": "#2dd4bf",
    "mint": "#79e7cd",
    "amber": "#f59e0b",
    "coral": "#fb7185",
    "blue": "#60a5fa",
    "muted": "#8fb0b4",
}


def polish(fig, height: int = 390):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,20,27,.25)",
        font=dict(family="DM Sans", color="#cfe5e5"),
        height=height,
        margin=dict(l=25, r=20, t=45, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="#0a2028"),
    )
    fig.update_xaxes(gridcolor="rgba(148,210,200,.08)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,210,200,.08)", zeroline=False)
    return fig
