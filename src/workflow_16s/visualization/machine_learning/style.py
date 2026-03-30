# workflow_16s/visualization/machine_learning/style.py
import plotly.graph_objects as go

DEFAULT_HEIGHT = 1100
DEFAULT_WIDTH_SQUARE = 1200
DEFAULT_WIDTH_RECTANGLE = 1600
DEFAULT_TITLE_FONT_SIZE = 24
DEFAULT_AXIS_TITLE_FONT_SIZE = 20
DEFAULT_TICKS_LABEL_FONT_SIZE = 16

def update_font_sizes(
    fig: go.Figure,
    title_font_size: int = DEFAULT_TITLE_FONT_SIZE,
    axis_title_font_size: int = DEFAULT_AXIS_TITLE_FONT_SIZE,
    ticks_label_font_size: int = DEFAULT_TICKS_LABEL_FONT_SIZE
):
    fig.update_layout(
        title=dict(font=dict(size=title_font_size)),
        xaxis=dict(
            title=dict(font=dict(size=axis_title_font_size)),
            tickfont=dict(size=ticks_label_font_size), 
        ),
        yaxis=dict(
            title=dict(font=dict(size=axis_title_font_size)),
            tickfont=dict(size=ticks_label_font_size), 
        )
    )    
    return fig