"""
LaTeX-based PDF Report Generator for Professional Publications
Generates publication-quality PDF reports using LaTeX formatting
"""

import os
import tempfile
import subprocess
import shutil
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import re


class LaTeXReportGenerator:
    """
    Generates professional LaTeX-formatted reports that can be compiled to PDF.
    Creates publication-quality documents suitable for academic journals.
    """
    
    def __init__(self, calculator, metrics: Dict[str, Any], assessments: Dict[str, str], 
                 org_name: str, flow_matrix: np.ndarray, node_names: List[str]):
        """Initialize LaTeX report generator."""
        self.calculator = calculator
        self.metrics = metrics
        self.assessments = assessments
        self.org_name = self._escape_latex(org_name)
        self.flow_matrix = flow_matrix
        self.node_names = node_names
        self.timestamp = datetime.now()
        
    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters."""
        if not text:
            return ""
        # Escape special LaTeX characters
        special_chars = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}',
            '\\': r'\textbackslash{}',
        }
        for char, escaped in special_chars.items():
            text = text.replace(char, escaped)
        return text
    
    def generate_latex_document(self) -> str:
        """Generate complete LaTeX document."""
        
        # Determine viability status
        viable = "viable" if self.metrics['is_viable'] else "non-viable"
        
        # Format metrics for display
        alpha = f"{self.metrics['ascendency_ratio']:.3f}"
        robustness = f"{self.metrics['robustness']:.3f}"
        efficiency = f"{self.metrics['network_efficiency']:.3f}"
        tst = f"{self.metrics['total_system_throughput']:.1f}"
        
        latex_doc = r"""\documentclass[11pt,a4paper]{article}

% Packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{times}  % Times font for professional look
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{float}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{fancyhdr}
\usepackage{abstract}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\usepackage{array}
\usepackage{multirow}
\usepackage{longtable}

% Colors
\definecolor{viable}{RGB}{34,139,34}
\definecolor{nonviable}{RGB}{220,20,60}
\definecolor{windowgreen}{RGB}{144,238,144}

% Header and footer
\pagestyle{fancy}
\fancyhf{}
\rhead{\small Network Analysis Report}
\lhead{\small """ + self.org_name + r"""}
\cfoot{\thepage}

% Title and authors
\title{\Large \textbf{Network Analysis of """ + self.org_name + r""":\\
A Quantitative Assessment Using Regenerative Economics Principles}}
\author{Adaptive Organization Analysis System\\
\small Generated: """ + self.timestamp.strftime('%B %d, %Y') + r"""}
\date{}

\begin{document}

\maketitle

% Abstract
\begin{abstract}
\noindent
This study presents a comprehensive network analysis of """ + self.org_name + r""" using the Ulanowicz-Fath 
regenerative economics framework. The analysis examines """ + str(len(self.node_names)) + r""" organizational units 
connected through """ + str(np.count_nonzero(self.flow_matrix)) + r""" directed flow relationships, 
representing a total system throughput of """ + tst + r""" units. 
Key findings indicate that the system exhibits a relative ascendency of $\alpha = """ + alpha + r"""$ 
and robustness of $R = """ + robustness + r"""$, positioning it as \textbf{""" + viable + r"""} 
within the theoretical window of viability (0.2 < $\alpha$ < 0.6). 
The analysis provides quantitative evidence for organizational sustainability assessment and 
strategic recommendations for system optimization.

\vspace{0.5em}
\noindent
\textbf{Keywords:} network analysis, organizational sustainability, information theory, 
regenerative economics, system resilience, adaptive capacity
\end{abstract}

% Main content
\section{Introduction}

The application of ecological network analysis to organizational systems represents a paradigm shift 
in understanding organizational dynamics and sustainability. This approach, pioneered by Ulanowicz 
(1986, 1997) and extended by Fath et al. (2019) to economic systems, provides quantitative measures 
of system health, efficiency, and resilience based on information-theoretic principles.

\subsection{Theoretical Framework}

The Ulanowicz framework quantifies organizational sustainability through information theory, 
treating organizations as flow networks. Central to this framework is the concept of 
ascendency ($A$), calculated as:

\begin{equation}
A = TST \times AMI
\end{equation}

where TST represents Total System Throughput and AMI is the Average Mutual Information. 
This is balanced against development capacity ($C$):

\begin{equation}
C = TST \times H
\end{equation}

where $H$ is the flow diversity (Shannon entropy).

\section{Methodology}

\subsection{Data Collection}

The analysis is based on a flow matrix representing """ + self.org_name + r""", consisting of:

\begin{itemize}
\item Number of nodes (organizational units): """ + str(len(self.node_names)) + r"""
\item Number of active connections: """ + str(np.count_nonzero(self.flow_matrix)) + r"""
\item Total system throughput: """ + tst + r""" units
\item Network density: """ + f"{np.count_nonzero(self.flow_matrix)/(len(self.node_names)**2):.3f}" + r"""
\end{itemize}

\subsection{Analytical Measures}

The following measures were calculated according to Ulanowicz (1986, 1997) and Fath et al. (2019):

\subsubsection{Information-Theoretic Measures}
\begin{itemize}
\item Total System Throughput: $TST = \sum_{ij} F_{ij}$
\item Average Mutual Information: $AMI = \sum_{ij} \frac{F_{ij}}{TST} \log_2\left(\frac{F_{ij} \cdot TST}{T_i \cdot T_j}\right)$
\item Flow Diversity: $H = -\sum_{ij} \frac{F_{ij}}{TST} \log_2\left(\frac{F_{ij}}{TST}\right)$
\end{itemize}

\section{Results}

\subsection{Key Performance Indicators}

\begin{table}[H]
\centering
\caption{Key System Metrics}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{Status} \\
\midrule
Viability Status & $\alpha = """ + alpha + r"""$ & """ + ("Viable" if self.metrics['is_viable'] else "Non-Viable") + r""" \\
Robustness & """ + robustness + r""" & """ + self._categorize_robustness() + r""" \\
Network Efficiency & """ + efficiency + r""" & """ + self._categorize_efficiency() + r""" \\
Total Throughput & """ + tst + r""" & """ + str(len(self.node_names)) + r""" nodes \\
\bottomrule
\end{tabular}
\end{table}

\subsection{System Organization Metrics}

\begin{table}[H]
\centering
\caption{Ulanowicz Framework Metrics}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{\% of Capacity} \\
\midrule
Ascendency ($A$) & """ + f"{self.metrics['ascendency']:.3f}" + r""" & """ + f"{self.metrics['ascendency_ratio']*100:.1f}" + r"""\% \\
Development Capacity ($C$) & """ + f"{self.metrics['development_capacity']:.3f}" + r""" & 100.0\% \\
Overhead ($\Phi$) & """ + f"{self.metrics['overhead']:.3f}" + r""" & """ + f"{self.metrics['overhead_ratio']*100:.1f}" + r"""\% \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Window of Viability Analysis}

The system's position relative to the window of viability is illustrated in Figure \ref{fig:viability}.

\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[
    width=\textwidth,
    height=6cm,
    xlabel={Relative Ascendency ($\alpha$)},
    ylabel={Robustness},
    xmin=0, xmax=1,
    ymin=0, ymax=0.6,
    grid=major,
    legend pos=north east,
]

% Window of viability
\fill[windowgreen, opacity=0.3] (axis cs:0.2,0) rectangle (axis cs:0.6,0.6);

% Theoretical robustness curve
\addplot[blue, thick, dashed, domain=0.01:0.99, samples=100] 
    {-x*ln(x)/ln(2)};

% Current position
\addplot[red, mark=*, mark size=5pt, only marks] 
    coordinates {(""" + alpha + r""",""" + robustness + r""")};

% Optimal point
\addplot[green, mark=star, mark size=5pt, only marks] 
    coordinates {(0.37, 0.531)};

\legend{Window of Viability, Theoretical Curve, Current Position, Optimal Point}

\end{axis}
\end{tikzpicture}
\caption{System position within the window of viability}
\label{fig:viability}
\end{figure}

\section{Discussion}

\subsection{Interpretation of Findings}

The relative ascendency of $\alpha = """ + alpha + r"""$ positions the system 
""" + self._interpret_efficiency_resilience_balance() + r""". 
The robustness value of $R = """ + robustness + r"""$ """ + ('exceeds' if self.metrics['robustness'] > 0.25 else 'approaches' if self.metrics['robustness'] > 0.15 else 'falls below') + r""" 
the threshold for high resilience ($R > 0.25$).

""" + self._generate_viability_discussion_latex() + r"""

\subsection{Comparison with Reference Systems}

When compared to sustainable reference systems:
\begin{itemize}
\item Ecological food webs: $\alpha \in [0.20, 0.50]$ (Ulanowicz, 2009)
\item High-performing organizations: $\alpha \in [0.30, 0.45]$ (Fath et al., 2019)
\item Current system: $\alpha = """ + alpha + r"""$ """ + ('aligns with' if 0.30 <= self.metrics['ascendency_ratio'] <= 0.45 else 'deviates from') + r""" benchmarks
\end{itemize}

\section{Conclusions and Recommendations}

\subsection{Key Findings}

\begin{enumerate}
\item The organization """ + ('operates within' if self.metrics['is_viable'] else 'falls outside') + r""" the window of viability
\item System exhibits """ + self._categorize_robustness().lower() + r""" resilience to perturbations
\item Network efficiency of """ + efficiency + r""" indicates """ + self._categorize_efficiency().lower() + r""" operational streamlining
\item Overhead ratio of """ + f"{self.metrics['overhead_ratio']:.3f}" + r""" suggests """ + ('adequate' if self.metrics['overhead_ratio'] > 0.4 else 'limited') + r""" adaptive capacity
\end{enumerate}

\subsection{Strategic Recommendations}

""" + self._generate_recommendations_latex() + r"""

\section*{References}

\small
\begin{itemize}
\item[$\bullet$] Fath, B. D., Fiscus, D. A., Goerner, S. J., Berea, A., \& Ulanowicz, R. E. (2019). 
    Measuring regenerative economics: 10 principles and measures undergirding systemic economic health. 
    \textit{Global Transitions}, 1, 15-27.

\item[$\bullet$] Ulanowicz, R. E. (1986). \textit{Growth and Development: Ecosystems Phenomenology}. 
    Springer-Verlag, New York.

\item[$\bullet$] Ulanowicz, R. E. (1997). \textit{Ecology, the Ascendent Perspective}. 
    Columbia University Press, New York.

\item[$\bullet$] Ulanowicz, R. E. (2009). \textit{A Third Window: Natural Life beyond Newton and Darwin}. 
    Templeton Foundation Press, West Conshohocken, PA.
\end{itemize}

\end{document}
"""
        return latex_doc
    
    def save_latex_file(self, filepath: str) -> bool:
        """Save LaTeX document to file."""
        try:
            latex_content = self.generate_latex_document()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            return True
        except Exception as e:
            print(f"Error saving LaTeX file: {e}")
            return False
    
    def compile_to_pdf(self, output_path: str = None) -> tuple:
        """
        Compile LaTeX document to PDF.
        Returns (success: bool, pdf_path: str or None, error_message: str or None)
        """
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Generate LaTeX file
                tex_file = os.path.join(temp_dir, "report.tex")
                self.save_latex_file(tex_file)
                
                # Try to compile with pdflatex
                try:
                    # Run pdflatex twice for references
                    for _ in range(2):
                        result = subprocess.run(
                            ["pdflatex", "-interaction=nonstopmode", "-output-directory", temp_dir, tex_file],
                            capture_output=True,
                            text=True,
                            cwd=temp_dir
                        )
                    
                    # Check if PDF was created
                    pdf_file = os.path.join(temp_dir, "report.pdf")
                    if os.path.exists(pdf_file):
                        # Copy to output path or return bytes
                        if output_path:
                            shutil.copy(pdf_file, output_path)
                            return (True, output_path, None)
                        else:
                            with open(pdf_file, 'rb') as f:
                                pdf_bytes = f.read()
                            return (True, pdf_bytes, None)
                    else:
                        return (False, None, "PDF compilation failed")
                        
                except FileNotFoundError:
                    # pdflatex not installed, return LaTeX source instead
                    return (False, None, "LaTeX not installed. Please install TeX distribution (e.g., TeX Live, MiKTeX)")
                    
        except Exception as e:
            return (False, None, str(e))
    
    # Helper methods for LaTeX generation
    
    def _categorize_efficiency(self) -> str:
        """Categorize network efficiency level."""
        eff = self.metrics['network_efficiency']
        if eff < 0.2:
            return "Low"
        elif eff < 0.4:
            return "Moderate"
        elif eff < 0.6:
            return "High"
        else:
            return "Very High"
    
    def _categorize_robustness(self) -> str:
        """Categorize robustness level."""
        rob = self.metrics['robustness']
        if rob < 0.1:
            return "Very Low"
        elif rob < 0.15:
            return "Low"
        elif rob < 0.2:
            return "Moderate"
        elif rob < 0.25:
            return "High"
        else:
            return "Very High"
    
    def _interpret_efficiency_resilience_balance(self) -> str:
        """Interpret the efficiency-resilience balance."""
        alpha = self.metrics['ascendency_ratio']
        if abs(alpha - 0.37) < 0.05:
            return "near the theoretical optimum for robustness"
        elif alpha < 0.37:
            return f"in the resilience-favoring regime ({(0.37 - alpha)*100:.1f}\\% below optimum)"
        else:
            return f"in the efficiency-favoring regime ({(alpha - 0.37)*100:.1f}\\% above optimum)"
    
    def _generate_viability_discussion_latex(self) -> str:
        """Generate LaTeX discussion about viability status."""
        if self.metrics['is_viable']:
            return """
The position within the window suggests functional balance between efficiency and flexibility, 
critical for long-term sustainability."""
        elif self.metrics['ascendency_ratio'] < self.metrics['viability_lower_bound']:
            return f"""
The sub-viable position ($\\alpha = {self.metrics['ascendency_ratio']:.3f} < {self.metrics['viability_lower_bound']:.3f}$) 
indicates insufficient organization, leading to inefficient resource utilization and reduced coherence."""
        else:
            return f"""
The supra-viable position ($\\alpha = {self.metrics['ascendency_ratio']:.3f} > {self.metrics['viability_upper_bound']:.3f}$) 
indicates over-organization, resulting in brittleness and limited adaptive capacity."""
    
    def _generate_recommendations_latex(self) -> str:
        """Generate strategic recommendations in LaTeX format."""
        recs = []
        
        if self.metrics['network_efficiency'] < 0.2:
            recs.append("\\item \\textbf{Increase Organizational Efficiency}: Streamline processes and improve coordination")
        elif self.metrics['network_efficiency'] > 0.6:
            recs.append("\\item \\textbf{Reduce Over-Optimization}: Introduce strategic redundancies")
        
        if self.metrics['robustness'] < 0.2:
            recs.append("\\item \\textbf{Enhance System Robustness}: Build reserve capacity and alternative pathways")
        
        if not self.metrics['is_viable']:
            if self.metrics['ascendency_ratio'] < self.metrics['viability_lower_bound']:
                recs.append("\\item \\textbf{Increase Organization}: Strengthen coordination and clarify structures")
            else:
                recs.append("\\item \\textbf{Increase Flexibility}: Decentralize decision-making and reduce constraints")
        
        if recs:
            return "\\begin{itemize}\n" + "\n".join(recs) + "\n\\end{itemize}"
        else:
            return "\\begin{itemize}\n\\item \\textbf{Maintain Current Configuration}: Continue monitoring key metrics\n\\end{itemize}"