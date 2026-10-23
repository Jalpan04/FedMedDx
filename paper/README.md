# FedMedDx IEEE Research Paper

This directory contains the complete, publication-ready research paper for the FedMedDx project formatted for standard 6-page IEEE conference proceedings (`IEEEtran`).

## Files

1. [fedmeddx_ieee_paper.tex](file:///d:/python%20projects/FedMedDx/paper/fedmeddx_ieee_paper.tex): Complete LaTeX source with embedded bibliography, mathematical formulations, algorithms, and comparative empirical tables.
2. [references.bib](file:///d:/python%20projects/FedMedDx/paper/references.bib): BibTeX citations for external reference managers.
3. **`figs/`**: High-resolution 300-DPI publication figures:
   - `figs/architecture.png`: Full-width system architecture banner.
   - `figs/convergence.png`: Loss convergence and multi-disease macro F1 progression.
   - `figs/gradcam.png`: Edge Grad-CAM explainability, saliency overlay, and class distribution.

## Compilation Instructions

### Option 1: Overleaf (Recommended)
1. Go to [Overleaf](https://www.overleaf.com).
2. Create a "New Project" -> "Blank Project".
3. Upload `fedmeddx_ieee_paper.tex`, `references.bib`, and the entire `figs/` folder.
4. Ensure the compiler is set to `pdfLaTeX` and click **Recompile**.

### Option 2: Local Command Line (pdflatex)
If TeX Live or MiKTeX is installed:
```bash
pdflatex fedmeddx_ieee_paper.tex
pdflatex fedmeddx_ieee_paper.tex
```

## Paper Highlights

- **Title**: *FedMedDx: Decoupled Federated Representation Learning with Adaptive Normalization for Cross-Institutional Multi-Disease Chest Radiography*
- **Primary Novelty**: Solving dual heterogeneity in medical FL (task heterogeneity across 4 different diagnostic disease heads, and domain/scanner shift across hospital sites via FedBN + FedRep).
- **Format**: Standard 2-column IEEE conference format (`\documentclass[conference]{IEEEtran}`), 6 pages with math equations, Algorithm 1 block, and 3 structured benchmark tables.
