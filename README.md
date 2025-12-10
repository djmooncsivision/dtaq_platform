# Data Analysis and Quality Platform

This repository contains tools and prototypes for data analysis, PDF processing, and reliability analysis.

## Repository Structure

```
dtaq_platform/
├── Antigravity/                    # Missile reliability analysis prototypes
│   ├── missile_reliability_proto_v0/
│   ├── missile_reliability_proto_v1/
│   └── missile_reliability_proto_v2/
├── dtaq_func_pdf_to_csv/          # PDF to CSV conversion tools
├── 2025_R_conference/             # R conference materials
├── dtaq_analysis/                 # Data analysis utilities
└── requirements.txt               # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Virtual environment recommended

### Installation

1. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Main Components

### Antigravity

Contains multiple versions of missile reliability analysis tools with various features:
- **proto_v0**: Initial PDF analysis and OCR extraction
- **proto_v1**: Enhanced with Streamlit GUI (vgui 1.0, 2.0, 3.0) for data acquisition, configuration, and analysis
- **proto_v2**: Advanced Bayesian modeling with PyMC for reliability estimation

### dtaq_func_pdf_to_csv

PDF table extraction and conversion pipeline for processing structured data from PDF documents.

## Usage

Refer to individual component directories for specific usage instructions and documentation.
