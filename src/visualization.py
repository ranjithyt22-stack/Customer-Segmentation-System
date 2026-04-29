import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_dashboard_plots(rfm):
    # Ensure static dir exists
    os.makedirs("static", exist_ok=True)

    # Cluster scatter plot
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=rfm['Frequency'], y=rfm['Monetary'], hue=rfm['Segment'], palette='Set2')
    plt.title('Customer Segmentation Clusters')
    plt.xlabel('Frequency')
    plt.ylabel('Monetary')
    plt.legend(title='Segment')
    plt.tight_layout()
    plt.savefig('static/cluster_plot.png')
    plt.close()

    # Segment distribution pie chart
    plt.figure(figsize=(5,5))
    seg_counts = rfm['Segment'].value_counts()
    seg_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2'))
    plt.title('Segment Distribution')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('static/segment_pie.png')
    plt.close()

    # Average spending per segment
    plt.figure(figsize=(7,5))
    avg_spending = rfm.groupby('Segment')['Monetary'].mean().sort_values(ascending=False)
    sns.barplot(x=avg_spending.index, y=avg_spending.values, palette='Set2')
    plt.title('Average Spending per Segment')
    plt.ylabel('Average Monetary Value')
    plt.xlabel('Segment')
    plt.tight_layout()
    plt.savefig('static/avg_spending.png')
    plt.close()
