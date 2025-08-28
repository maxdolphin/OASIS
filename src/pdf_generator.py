"""
PDF Report Generator
Generates professional PDF reports without requiring LaTeX installation.
Uses HTML/CSS for formatting and converts to PDF.
"""

import base64
from io import BytesIO
import plotly.graph_objects as go
import plotly.io as pio

def generate_pdf_report(report_generator, calculator, metrics, charts=None):
    """
    Generate a professional PDF report.
    
    Args:
        report_generator: PublicationReportGenerator instance
        calculator: RegenerativeMetricsCalculator instance  
        metrics: Dictionary of calculated metrics
        charts: Optional dictionary of plotly figures to include
        
    Returns:
        bytes: PDF file content
    """
    
    # Generate HTML content with embedded CSS
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{
                font-family: 'Times New Roman', serif;
                font-size: 12pt;
                line-height: 1.6;
                color: #333;
                text-align: justify;
            }}
            h1 {{
                font-size: 24pt;
                font-weight: bold;
                text-align: center;
                margin-bottom: 20px;
                page-break-after: avoid;
            }}
            h2 {{
                font-size: 18pt;
                font-weight: bold;
                margin-top: 20px;
                margin-bottom: 10px;
                page-break-after: avoid;
            }}
            h3 {{
                font-size: 14pt;
                font-weight: bold;
                margin-top: 15px;
                margin-bottom: 8px;
                page-break-after: avoid;
            }}
            .abstract {{
                background-color: #f5f5f5;
                padding: 15px;
                border-left: 4px solid #2c3e50;
                margin: 20px 0;
                font-style: italic;
            }}
            .metric-box {{
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
            }}
            .chart-container {{
                margin: 20px 0;
                text-align: center;
                page-break-inside: avoid;
            }}
            .chart-image {{
                max-width: 100%;
                height: auto;
            }}
            pre {{
                white-space: pre-wrap;
                word-wrap: break-word;
                font-family: 'Times New Roman', serif;
            }}
            .page-break {{
                page-break-before: always;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            .footer {{
                position: fixed;
                bottom: 0;
                width: 100%;
                text-align: center;
                font-size: 10pt;
                color: #666;
            }}
        </style>
    </head>
    <body>
    """
    
    # Title page
    html_content += f"""
    <h1>ORGANIZATIONAL NETWORK ANALYSIS REPORT</h1>
    <h2 style="text-align: center; font-weight: normal;">{report_generator.org_name}</h2>
    <p style="text-align: center; margin-top: 30px;">
        Based on Ulanowicz-Fath Regenerative Economics Framework<br>
        Generated: {report_generator._get_timestamp()}
    </p>
    
    <div class="page-break"></div>
    """
    
    # Abstract
    html_content += f"""
    <h2>ABSTRACT</h2>
    <div class="abstract">
        <pre>{report_generator.generate_abstract()}</pre>
    </div>
    """
    
    # Introduction
    html_content += f"""
    <div class="page-break"></div>
    <h2>1. INTRODUCTION</h2>
    <pre>{report_generator.generate_introduction()}</pre>
    """
    
    # Methodology
    html_content += f"""
    <div class="page-break"></div>
    <h2>2. METHODOLOGY</h2>
    <pre>{report_generator.generate_methodology()}</pre>
    """
    
    # Results with charts
    html_content += f"""
    <div class="page-break"></div>
    <h2>3. RESULTS</h2>
    <pre>{report_generator.generate_results()}</pre>
    """
    
    # Add charts if provided
    if charts:
        for chart_name, fig in charts.items():
            if fig:
                # Convert Plotly figure to static image (base64 encoded)
                img_bytes = pio.to_image(fig, format='png', width=700, height=500)
                img_base64 = base64.b64encode(img_bytes).decode()
                
                html_content += f"""
                <div class="chart-container">
                    <h3>{chart_name}</h3>
                    <img src="data:image/png;base64,{img_base64}" class="chart-image">
                </div>
                """
    
    # Discussion
    html_content += f"""
    <div class="page-break"></div>
    <h2>4. DISCUSSION</h2>
    <pre>{report_generator.generate_discussion()}</pre>
    """
    
    # Conclusions
    html_content += f"""
    <div class="page-break"></div>
    <h2>5. CONCLUSIONS AND RECOMMENDATIONS</h2>
    <pre>{report_generator.generate_conclusions()}</pre>
    """
    
    # References
    html_content += f"""
    <div class="page-break"></div>
    <h2>REFERENCES</h2>
    <pre>{report_generator.generate_references()}</pre>
    """
    
    # Appendix
    html_content += f"""
    <div class="page-break"></div>
    <h2>APPENDIX: DETAILED DATA</h2>
    <pre>{report_generator.generate_appendix()}</pre>
    """
    
    # Close HTML
    html_content += """
    </body>
    </html>
    """
    
    # Try to use different PDF generation methods
    try:
        # Try weasyprint first (best quality)
        from weasyprint import HTML, CSS
        pdf_bytes = HTML(string=html_content).write_pdf()
        return pdf_bytes
    except ImportError:
        pass
    
    try:
        # Try pdfkit (requires wkhtmltopdf)
        import pdfkit
        pdf_bytes = pdfkit.from_string(html_content, False)
        return pdf_bytes
    except:
        pass
    
    # Fallback: Return HTML for user to convert
    # Create a simple HTML file that can be opened and printed to PDF
    return html_content.encode('utf-8')


def create_simple_pdf(report_text, org_name):
    """
    Create a simple text-based PDF using reportlab.
    Fallback method that doesn't require external dependencies.
    
    Args:
        report_text: Full text of the report
        org_name: Organization name
        
    Returns:
        bytes: PDF content
    """
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
        
        # Create BytesIO buffer
        buffer = BytesIO()
        
        # Create PDF
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=12
        )
        
        # Build content
        story = []
        
        # Title
        story.append(Paragraph(f"Organizational Network Analysis Report", title_style))
        story.append(Paragraph(f"{org_name}", heading_style))
        story.append(Spacer(1, 0.5*inch))
        
        # Split report into sections and format
        sections = report_text.split('\n\n')
        for section in sections:
            if section.strip():
                # Check if it's a heading (all caps or starts with number)
                if section.strip().isupper() or (len(section.strip()) > 0 and section.strip()[0].isdigit()):
                    story.append(PageBreak())
                    story.append(Paragraph(section.strip(), heading_style))
                else:
                    # Clean the text for reportlab
                    clean_text = section.strip().replace('\n', '<br/>')
                    story.append(Paragraph(clean_text, body_style))
                story.append(Spacer(1, 0.1*inch))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
        
    except ImportError:
        # If reportlab is not available, return None
        return None