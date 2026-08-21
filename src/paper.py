from dataclasses import dataclass
from typing import Optional

from plotly import graph_objects as go


@dataclass
class PaperStyle:
    font_family: str = "Open Sans"
    font_size: int = 26
    legend_font_family: str = "Open Sans"
    legend_font_size: int = 24
    plot_bgcolor = "rgba(0,0,0,0)"
    gridwidth: int = 1
    gridcolor: str = "rgba(0,0,0,0.1)"
    legend_bgcolor: str = "rgba(1.0,1.0,1.0,0.3)"


def styler(font_size, legend_font_size):
    PaperStyle.font_size = font_size
    PaperStyle.legend_font_size = legend_font_size
    return PaperStyle


def with_paper_style(
    fig: go.Figure,
    config: PaperStyle = PaperStyle(),
    *,
    new_legend: Optional[dict] = None,
    # Top left corner of the plot
    legend_pos: Optional[tuple[float, float]] = (0.8,1.2),
) -> go.Figure:
    if legend_pos is None:
        show_legend = False
        legend = None
    elif new_legend:
        show_legend = True
        legend = new_legend
    else:
        show_legend = True
        legend = dict(
            x=legend_pos[0],
            y=legend_pos[1],
            xanchor="right",
            yanchor="top",
            orientation="h",  # Set the legend orientation to horizontal
            traceorder="normal",
            font=dict(
                family=config.legend_font_family,
                size=config.legend_font_size,
                color="black"
            ),
            bgcolor=config.legend_bgcolor,
            # bordercolor=config.gridcolor,
            # borderwidth=config.gridwidth,
        )
    axis_config = dict(
        showgrid=True,
        gridwidth=config.gridwidth,
        gridcolor=config.gridcolor,
        ticks="outside",
        tickwidth=config.gridwidth,
        tickcolor=config.gridcolor,
        zeroline=False,
        showline=False,
    )

    fig.update_layout(
        font_family=config.font_family,
        font_size=config.font_size,
        font_color="black",
        plot_bgcolor=config.plot_bgcolor,
        title_text="",
        showlegend=show_legend,
        legend=legend,
        # xaxis=axis_config,
        # yaxis=axis_config,
    )
    fig.update_xaxes(**axis_config)
    fig.update_yaxes(**axis_config)
    return fig
