from pathlib import Path
import re
import numpy as np
import pandas as pd
import plotly.express as px
from workflow_16s.utils.logger import get_logger
def plot_facility_match_pie_chart(
    df: pd.DataFrame, 
    output_dir: Path,
    file_stem: str = "facility_match_distribution"
):
    if 'facility_match' in df.columns:
        match_counts = df['facility_match'].astype(str).value_counts()
        fig = px.pie(
            names=match_counts.index, values=match_counts.values,
            title='Distribution of Samples Matching a Nearby Facility',
            hole=0.3,
            color_discrete_map={'True': '#1f77b4', 'False': '#d62728'}
        )
        fig.update_traces(textinfo='percent+label', pull=[0.05, 0])
        fig.update_layout(legend_title_text='Facility Match Status')
        output_path = output_dir / file_stem
        fig.write_html(f"{output_path}.html")
        fig.write_image(f"{output_path}.png", scale=2)
        

def plot_global_facilities_map(nfc_facilities_df, output_dir, output_filename: str = "global_facilities_map.html"):
    """
    Generates an interactive Plotly map of ALL aggregated facilities.
    Colors points by 'facility_category' (Nuclear vs Analog) and shapes by 'facility_type'.
    """
    logger = get_logger("workflow_16s")
    if nfc_facilities_df.empty:
        logger.warning(" ⚠️ No facilities to plot.")
        return

    df = nfc_facilities_df.copy()
    df['Category'] = df['facility_category'].fillna('Unknown')
    df['Type'] = df['facility_type_standard'].fillna('Other')
    plot_df = df.dropna(subset=['lat', 'lon'])
        
    logger.info(f" 🗺️  Plotting global map for {len(plot_df)} facilities...")

    try:
        fig = px.scatter_geo(
            plot_df,
            lat='lat',
            lon='lon',
            color='Category',
            symbol='Category',  # Different shapes for Nuclear vs Analog
            hover_name='facility',
            hover_data={
                    'lat': False, 
                    'lon': False, 
                    'Category': False,
                    'Type': True,
                    'country': True,
                    'data_source': True
            },
            title=f"Global Nuclear Fuel Cycle & Analog Facilities (n={len(plot_df)})",
            projection="natural earth",
            color_discrete_map={
                    'Nuclear Fuel Cycle': '#d62728',  # Red
                    'Contamination Analog': '#1f77b4' # Blue
            }
        )

        fig.update_geos(
                showcountries=True, 
                countrycolor="RebeccaPurple",
                showcoastlines=True, 
                coastlinecolor="RebeccaPurple",
                showland=True, 
                landcolor="LightGreen",
                showocean=True, 
                oceancolor="LightBlue"
        )
            
        fig.update_layout(legend_title_text='Facility Class')
        output_path = output_dir / output_filename
        fig.write_html(str(output_path))
        logger.info(f" 💾 Interactive map saved to: {output_path}")
            
    except Exception as e:
        logger.error(f" 🚫 Failed to plot global facilities map: {e}")