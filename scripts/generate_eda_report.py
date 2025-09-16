"""
Generate comprehensive EDA PDF report with figures
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_eda_report():
    """Create comprehensive EDA PDF report"""
    
    # Load data
    print("Loading NYC taxi dataset...")
    df = pd.read_csv('data/raw/nyc_taxi.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Create time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day'] = df.index.day
    
    # Create PDF report
    with PdfPages('reports/NYC_Taxi_EDA_Report.pdf') as pdf:
        
        # Title Page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        fig.text(0.5, 0.85, 'NYC Taxi Demand Analysis', 
                ha='center', va='center', fontsize=28, fontweight='bold')
        fig.text(0.5, 0.80, 'Exploratory Data Analysis Report', 
                ha='center', va='center', fontsize=20)
        
        # Dataset info
        fig.text(0.5, 0.70, f'Dataset Period: {df.index.min().strftime("%B %d, %Y")} - {df.index.max().strftime("%B %d, %Y")}', 
                ha='center', va='center', fontsize=14)
        fig.text(0.5, 0.67, f'Total Observations: {len(df):,}', 
                ha='center', va='center', fontsize=14)
        fig.text(0.5, 0.64, f'Frequency: 30-minute intervals', 
                ha='center', va='center', fontsize=14)
        
        # Key statistics
        fig.text(0.5, 0.55, 'Key Statistics', 
                ha='center', va='center', fontsize=18, fontweight='bold')
        
        stats_text = f"""
        Average trips per 30-min: {df['value'].mean():,.0f}
        Maximum trips (30-min): {df['value'].max():,}
        Minimum trips (30-min): {df['value'].min():,}
        Standard deviation: {df['value'].std():,.0f}
        
        Data Quality:
        • Missing values: {df['value'].isnull().sum()}
        • Duplicate timestamps: {df.index.duplicated().sum()}
        • Negative values: {(df['value'] < 0).sum()}
        """
        
        fig.text(0.5, 0.42, stats_text, 
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # Report info
        fig.text(0.5, 0.15, f'Generated on: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}', 
                ha='center', va='center', fontsize=10, style='italic')
        fig.text(0.5, 0.12, 'Time Series Analysis Project', 
                ha='center', va='center', fontsize=10, style='italic')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Time Series Overview
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
        
        # Complete time series
        ax1.plot(df.index, df['value'], alpha=0.7, linewidth=0.5, color='navy')
        ax1.set_title('NYC Taxi Trips Over Time (Complete Dataset)', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Number of Taxi Trips (30-min intervals)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # First week detail
        first_week = df.iloc[:336]  # 7 days * 48 half-hours
        ax2.plot(first_week.index, first_week['value'], linewidth=1.5, color='darkblue')
        ax2.set_title('First Week Detail - Daily Patterns Visible', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Date (July 1-7, 2014)', fontsize=12)
        ax2.set_ylabel('Number of Taxi Trips', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 3: Hourly Patterns
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
        
        # Hourly average with error bars
        hourly_stats = df.groupby('hour')['value'].agg(['mean', 'std', 'min', 'max'])
        ax1.plot(hourly_stats.index, hourly_stats['mean'], 'o-', linewidth=3, markersize=8, color='red', label='Average')
        ax1.fill_between(hourly_stats.index, 
                        hourly_stats['mean'] - hourly_stats['std'],
                        hourly_stats['mean'] + hourly_stats['std'], 
                        alpha=0.3, color='red', label='±1 Standard Deviation')
        ax1.set_title('Average Taxi Trips by Hour of Day', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Average Trips per 30-min', fontsize=12)
        ax1.set_xticks(range(0, 24))
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Box plot by hour
        hour_data = [df[df['hour'] == h]['value'].values for h in range(24)]
        bp = ax2.boxplot(hour_data, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        ax2.set_title('Distribution of Trips by Hour (Boxplot)', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Hour of Day', fontsize=12)
        ax2.set_ylabel('Number of Trips', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 4: Weekly Patterns  
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
        
        # Day of week pattern
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_stats = df.groupby('day_of_week')['value'].agg(['mean', 'std'])
        
        bars = ax1.bar(days, dow_stats['mean'], yerr=dow_stats['std'], 
                      capsize=5, alpha=0.8, color='skyblue', edgecolor='navy', linewidth=1.5)
        ax1.set_title('Average Taxi Trips by Day of Week', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Average Trips per 30-min', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, avg in zip(bars, dow_stats['mean']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                    f'{avg:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Heatmap of hour vs day of week
        pivot_data = df.pivot_table(values='value', index='hour', columns='day_of_week', aggfunc='mean')
        im = ax2.imshow(pivot_data.T, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        ax2.set_title('Demand Heatmap: Hour vs Day of Week', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Hour of Day', fontsize=12)
        ax2.set_ylabel('Day of Week', fontsize=12)
        ax2.set_yticks(range(7))
        ax2.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        ax2.set_xticks(range(0, 24, 4))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
        cbar.set_label('Average Trips', fontsize=10)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 5: Monthly and Statistical Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        
        # Monthly pattern
        monthly_stats = df.groupby('month')['value'].agg(['mean', 'count'])
        months = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
        ax1.plot(months, monthly_stats['mean'], 'o-', linewidth=2, markersize=8, color='green')
        ax1.set_title('Monthly Trends', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Average Trips per 30-min', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Distribution histogram
        ax2.hist(df['value'], bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax2.axvline(df['value'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["value"].mean():.0f}')
        ax2.axvline(df['value'].median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {df["value"].median():.0f}')
        ax2.set_title('Trip Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Trips', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Peak periods
        peak_periods = df.nlargest(10, 'value')
        peak_hours = [t.strftime('%m/%d %H:%M') for t in peak_periods.index]
        ax3.barh(range(len(peak_periods)), peak_periods['value'], color='red', alpha=0.7)
        ax3.set_title('Top 10 Peak Periods', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Number of Trips', fontsize=10)
        ax3.set_yticks(range(len(peak_periods)))
        ax3.set_yticklabels(peak_hours, fontsize=8)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Summary statistics table
        ax4.axis('off')
        stats_data = [
            ['Statistic', 'Value'],
            ['Total Trips', f'{df["value"].sum():,}'],
            ['Average per 30min', f'{df["value"].mean():,.0f}'],
            ['Peak Period', f'{peak_periods.index[0].strftime("%B %d, %Y %H:%M")}'],
            ['Peak Trips', f'{peak_periods.iloc[0]["value"]:,}'],
            ['Coefficient of Variation', f'{df["value"].std()/df["value"].mean():.2f}'],
            ['Weekday Average', f'{df[df["day_of_week"] < 5]["value"].mean():.0f}'],
            ['Weekend Average', f'{df[df["day_of_week"] >= 5]["value"].mean():.0f}']
        ]
        
        table = ax4.table(cellText=stats_data[1:], colLabels=stats_data[0], 
                         cellLoc='center', loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax4.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 6: Key Insights Summary
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        fig.text(0.5, 0.95, 'Key Insights from Exploratory Data Analysis', 
                ha='center', va='center', fontsize=20, fontweight='bold')
        
        insights_text = f"""
TEMPORAL PATTERNS DISCOVERED:

🕐 Daily Patterns:
• Peak Hour: 7:00 PM ({hourly_stats.loc[19, 'mean']:,.0f} avg trips)
• Rush Hours: 6-8 PM consistently highest demand  
• Quiet Period: 3-6 AM (lowest demand)
• Rush Hour Ratio: {hourly_stats.loc[19, 'mean']/hourly_stats.loc[5, 'mean']:.1f}x higher than minimum

📅 Weekly Patterns:
• Busiest Day: Saturday ({dow_stats.loc[5, 'mean']:,.0f} avg trips)
• Quietest Day: Monday ({dow_stats.loc[0, 'mean']:,.0f} avg trips)  
• Weekend Effect: {((dow_stats.loc[5, 'mean'] + dow_stats.loc[6, 'mean'])/2) / dow_stats.loc[:4, 'mean'].mean() - 1:.1%} higher than weekdays
• Business Days: Steady increase Thu-Fri

📈 Seasonal Trends:
• Peak Month: October ({monthly_stats.loc[10, 'mean']:,.0f} avg trips)
• Holiday Spikes: New Year's Eve/Day highest peaks
• Seasonal Variation: {monthly_stats['mean'].max()/monthly_stats['mean'].min() - 1:.1%} difference

⚡ Demand Characteristics:
• High Variability: CV = {df['value'].std()/df['value'].mean():.2f}
• Strong Predictability: Clear daily and weekly cycles
• Peak Demand: {peak_periods.iloc[0]['value']:,} trips on {peak_periods.index[0].strftime('%B %d, %Y')}
• Business Impact: Predictable patterns enable optimization

🎯 MODELING IMPLICATIONS:
• Strong seasonality suggests seasonal models (SARIMA, Holt-Winters)
• Time-of-day features critical for ML models  
• Lag features will be highly predictive
• Weekend/weekday effects should be captured
• Holiday effects need special handling
        """
        
        fig.text(0.05, 0.80, insights_text, 
                ha='left', va='top', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        fig.text(0.5, 0.05, f'Report generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}', 
                ha='center', va='center', fontsize=8, style='italic')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print("EDA Report generated: reports/NYC_Taxi_EDA_Report.pdf")

if __name__ == "__main__":
    # Create reports directory
    os.makedirs('reports', exist_ok=True)
    create_eda_report()