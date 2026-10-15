#!/usr/bin/env python3
"""
Create a PDF summary of the time series forecasting project
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def create_project_summary_pdf():
    """Create a PDF with the project summary paragraph"""

    # Create the PDF document
    doc = SimpleDocTemplate("C:/Astreon/TimeSeries_Project_Summary.pdf", pagesize=letter, author="DataRoamer")
    story = []

    # Get styles
    styles = getSampleStyleSheet()

    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=1,  # Center alignment
        textColor='#2E4057'
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=20,
        alignment=4,  # Justify alignment
        leftIndent=36,
        rightIndent=36,
        leading=18
    )

    # Add title
    title = Paragraph("NYC Taxi Demand Forecasting Project Summary", title_style)
    story.append(title)
    story.append(Spacer(1, 0.5*inch))

    # Add the summary paragraph
    summary_text = """
    This project develops a predictive forecasting system for NYC taxi demand using advanced machine learning and deep learning models (SARIMA, Random Forest, LSTM) to transform taxi operations from reactive to proactive fleet management. By accurately predicting passenger demand 30 minutes to 24 hours in advance, the system enables dynamic driver positioning and resource allocation, with the potential to reduce average passenger wait times by 15-25% and decrease operational costs through optimized fleet utilization. The implications extend beyond operational efficiency to enhancing customer satisfaction, reducing urban congestion through better demand-supply matching, and establishing a data-driven foundation for smart city transportation systems that can adapt in real-time to changing demand patterns and improve overall urban mobility infrastructure.
    """

    summary_paragraph = Paragraph(summary_text, body_style)
    story.append(summary_paragraph)

    # Add some spacing
    story.append(Spacer(1, 0.5*inch))

    # Add project details section
    details_style = ParagraphStyle(
        'Details',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=10,
        leftIndent=36,
        rightIndent=36,
        textColor='#666666'
    )

    details_text = """
    <b>Key Technologies:</b> Python, TensorFlow/Keras, Scikit-learn, Pandas, Time Series Analysis<br/>
    <b>Models Evaluated:</b> Naive Forecasting, SARIMA, Random Forest, LSTM Neural Networks<br/>
    <b>Dataset:</b> NYC Taxi Trip Counts (Hourly frequency, ~10,320 data points)<br/>
    <b>Best Performance:</b> LSTM model with MAE of ~6,151 trips
    """

    details_paragraph = Paragraph(details_text, details_style)
    story.append(details_paragraph)

    # Build the PDF
    doc.build(story)
    print("Project summary PDF created: C:/Astreon/TimeSeries_Project_Summary.pdf")

if __name__ == "__main__":
    create_project_summary_pdf()