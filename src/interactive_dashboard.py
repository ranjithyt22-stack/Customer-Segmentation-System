import pandas as pd
import joblib
import plotly.express as px
import plotly.io as pio
import os

def generate_interactive_dashboard(rfm):
    os.makedirs("static", exist_ok=True)

    # Cluster scatter plot (interactive)
    fig1 = px.scatter(
        rfm,
        x='Frequency', y='Monetary', color='Segment',
        hover_data=['Recency'],
        title='Customer Segmentation Clusters (Interactive)'
    )
    pio.write_html(fig1, file="static/cluster_plot_interactive.html", auto_open=False, include_plotlyjs=True)

    # Segment distribution pie chart (interactive)
    seg_counts = rfm['Segment'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Count']
    fig2 = px.pie(seg_counts, names='Segment', values='Count', title='Segment Distribution (Interactive)')
    pio.write_html(fig2, file="static/segment_pie_interactive.html", auto_open=False, include_plotlyjs=True)

    # Average spending per segment (interactive)
    avg_spending = rfm.groupby('Segment')['Monetary'].mean().reset_index()
    fig3 = px.bar(avg_spending, x='Segment', y='Monetary', title='Average Spending per Segment (Interactive)')
    pio.write_html(fig3, file="static/avg_spending_interactive.html", auto_open=False, include_plotlyjs=True)
