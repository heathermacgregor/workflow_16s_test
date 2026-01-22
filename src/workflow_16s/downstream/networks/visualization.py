import logging
from pathlib import Path
from typing import Dict, Optional
import networkx as nx
import plotly.graph_objects as go

logger = logging.getLogger('workflow_16s')

def plot_network(
    network_result: Dict,
    layout: str = 'spring',
    node_color_by: Optional[str] = None,
    output_path: Optional[Path] = None
) -> go.Figure:
    """Create interactive network visualization using Plotly."""
    G = network_result['network']
    
    # Layouts
    if layout == 'spring': pos = nx.spring_layout(G, k=0.5)
    elif layout == 'circular': pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai': pos = nx.kamada_kawai_layout(G)
    else: pos = nx.spring_layout(G)

    # Edges
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'),
        hoverinfo='none', mode='lines'
    )

    # Nodes
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    
    # Node Colors (Default to Degree)
    node_degrees = [G.degree(n) for n in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=list(G.nodes()), textposition='top center', hoverinfo='text',
        marker=dict(
            showscale=True, colorscale='YlGnBu', size=10,
            color=node_degrees,
            colorbar=dict(thickness=15, title='Degree')
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f"Network: {network_result.get('method', 'Unknown')}",
                        showlegend=False, hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        template='plotly_white'
                    ))
    
    if output_path:
        fig.write_html(str(output_path))
        logger.info(f"Network plot saved to {output_path}")

    return fig