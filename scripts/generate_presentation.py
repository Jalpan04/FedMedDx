"""
FedMedDx Professional Presentation Generator.
Generates a 17-slide widescreen (16:9) PowerPoint deck for the progress review.
Adheres strictly to professional styling: custom color palette, cards, tables,
architectural blocks, and speaker notes. No emojis used.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def build_presentation(output_path: str = "FedMedDx_Progress_Review.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Color Palette
    COLOR_PRIMARY_DARK = RGBColor(15, 23, 42)     # Slate 900 (Deep Navy)
    COLOR_SECONDARY_DARK = RGBColor(30, 41, 59)   # Slate 800
    COLOR_ACCENT_BLUE = RGBColor(2, 132, 199)     # Sky 600
    COLOR_ACCENT_CYAN = RGBColor(14, 165, 233)    # Sky 500
    COLOR_TEAL = RGBColor(13, 148, 136)           # Teal 600
    COLOR_BG_LIGHT = RGBColor(248, 250, 252)      # Slate 50
    COLOR_CARD_BG = RGBColor(255, 255, 255)       # White
    COLOR_CARD_BORDER = RGBColor(226, 232, 240)   # Slate 200
    COLOR_CARD_BORDER_ALT = RGBColor(203, 213, 225)
    COLOR_TEXT_MAIN = RGBColor(30, 41, 59)        # Slate 800
    COLOR_TEXT_MUTED = RGBColor(100, 116, 139)    # Slate 500
    COLOR_TEXT_WHITE = RGBColor(255, 255, 255)
    COLOR_GREEN = RGBColor(22, 101, 52)           # Green 800
    COLOR_GREEN_BG = RGBColor(240, 253, 244)      # Green 50
    COLOR_RED = RGBColor(153, 27, 27)             # Red 800
    COLOR_RED_BG = RGBColor(254, 242, 242)        # Red 50
    COLOR_HEADER_BG = RGBColor(241, 245, 249)     # Slate 100

    blank_slide_layout = prs.slide_layouts[6]

    def set_slide_background(slide, color):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = color

    def add_header(slide, title_text, category_text="FEDMEDDX PROGRESS REVIEW", dark=False):
        # Category Pill / Tag
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(10), Inches(0.3))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = COLOR_ACCENT_CYAN if dark else COLOR_ACCENT_BLUE

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.5), Inches(0.6))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_TEXT_WHITE if dark else COLOR_PRIMARY_DARK

    def add_card(slide, left, top, width, height, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1)
        return shape

    def add_footer(slide, current_idx, total_count=17, dark=False):
        footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.0), Inches(11.733), Inches(0.3))
        tf = footer_box.text_frame
        p = tf.paragraphs[0]
        p.text = f"FedMedDx -- Multi-Disease Representation Learning on Unified CXR  |  Slide {current_idx} of {total_count}"
        p.font.size = Pt(9)
        p.font.color.rgb = COLOR_TEXT_MUTED if not dark else RGBColor(148, 163, 184)

    def add_notes(slide, text):
        notes_slide = slide.notes_slide
        text_frame = notes_slide.notes_text_frame
        text_frame.text = text

    # ==========================================
    # SLIDE 1: Title Slide (Dark Theme)
    # ==========================================
    s1 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(s1, COLOR_PRIMARY_DARK)

    # Accent decorative bar
    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.8), Inches(0.15), Inches(2.2))
    bar.fill.solid()
    bar.fill.fore_color.rgb = COLOR_ACCENT_CYAN
    bar.line.fill.background()

    # Title box
    tb = s1.shapes.add_textbox(Inches(1.2), Inches(1.7), Inches(11.0), Inches(2.5))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = "FedMedDx"
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEXT_WHITE

    p2 = tf.add_paragraph()
    p2.text = "Multi-Disease Representation Learning on Unified Chest Radiography"
    p2.font.size = Pt(20)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_ACCENT_CYAN
    p2.space_before = Pt(8)

    p3 = tf.add_paragraph()
    p3.text = "Privacy-Preserving Federated Diagnostic AI Across Heterogeneous Hospital Datasets"
    p3.font.size = Pt(14)
    p3.font.color.rgb = RGBColor(203, 213, 225)
    p3.space_before = Pt(6)

    # Meta card (Team details)
    meta_card = add_card(s1, Inches(0.8), Inches(4.5), Inches(11.733), Inches(2.0), bg_color=COLOR_SECONDARY_DARK, border_color=RGBColor(51, 65, 85))
    tb_meta = s1.shapes.add_textbox(Inches(1.1), Inches(4.7), Inches(11.1), Inches(1.6))
    tf_m = tb_meta.text_frame
    tf_m.word_wrap = True

    p = tf_m.paragraphs[0]
    p.text = "PROJECT PROGRESS REVIEW -- STAGE 2"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_CYAN

    p = tf_m.add_paragraph()
    p.text = "Core Lead: Jalpan (Federated Engine, FedRep + FedBN Coordinator)"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEXT_WHITE
    p.space_before = Pt(4)

    p = tf_m.add_paragraph()
    p.text = "Disease Module Developers: Priyanka (COVID-19), Gargee (Pneumonia), Smit (Tuberculosis), Hirva (Pediatric Pneumonia)"
    p.font.size = Pt(11)
    p.font.color.rgb = RGBColor(226, 232, 240)
    p.space_before = Pt(3)

    add_footer(s1, 1, 17, dark=True)
    add_notes(s1, "Verbal cue: FedMedDx enables multiple hospitals to collaboratively train a diagnostic AI without ever sharing a single patient scan.")

    # ==========================================
    # SLIDE 2: Problem Statement
    # ==========================================
    s2 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(s2, COLOR_BG_LIGHT)
    add_header(s2, "The Clinical Data Silo and Diagnostic Gap")

    cards_data_s2 = [
        ("Data Silos & Privacy Regulations", 
         "Strict health data regulations (HIPAA, GDPR) strictly prohibit pooling raw patient scans into centralized cloud repositories, isolating valuable diagnostic data inside local hospital firewalls.",
         COLOR_ACCENT_BLUE),
        ("High Radiology Workload & Misdiagnosis", 
         "Chest Radiography accounts for ~40% of all diagnostic imaging worldwide. High clinical caseloads contribute to an estimated 3-5% diagnostic error rate even among experienced radiologists.",
         COLOR_ACCENT_BLUE),
        ("Small, Biased Local Datasets", 
         "Isolated hospital AI models overfit to local scanner hardware and patient demographics, causing severe generalization failures when evaluated on external clinical centers.",
         COLOR_ACCENT_BLUE),
        ("Heterogeneous Task Problem", 
         "Different clinical centers specialize in different conditions (4-class COVID-19, binary Pneumonia, binary TB). No conventional system allows federating across mismatched label spaces.",
         COLOR_ACCENT_BLUE),
    ]

    for i, (title, desc, accent) in enumerate(cards_data_s2):
        col = i % 2
        row = i // 2
        left = Inches(0.8 + col * 5.95)
        top = Inches(1.4 + row * 2.65)
        
        add_card(s2, left, top, Inches(5.75), Inches(2.45))
        
        # Color bar top
        b = s2.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, Inches(5.75), Inches(0.08))
        b.fill.solid()
        b.fill.fore_color.rgb = accent
        b.line.fill.background()

        tb = s2.shapes.add_textbox(left + Inches(0.2), top + Inches(0.2), Inches(5.35), Inches(2.0))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = f"0{i+1}. {title}"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY_DARK
        
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(11)
        p2.font.color.rgb = COLOR_TEXT_MAIN
        p2.space_before = Pt(8)

    add_footer(s2, 2, 17)
    add_notes(s2, "Verbal cue: Hospitals sit on clinically valuable data they cannot share. FedMedDx turns that constraint into a collaborative design principle.")

    # ==========================================
    # SLIDE 3: Existing Systems / Limitations
    # ==========================================
    s3 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(s3, COLOR_BG_LIGHT)
    add_header(s3, "Limitations of Current Diagnostic & FL Approaches")

    # Table comparing architectures
    table_shape = s3.shapes.add_table(5, 5, Inches(0.8), Inches(1.5), Inches(11.733), Inches(4.8))
    table = table_shape.table

    headers = ["Approach", "Data Privacy (HIPAA)", "Heterogeneous Labels", "Hardware Invariance", "Cross-Hospital Benefit"]
    for col_idx, text in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY_DARK
        p = cell.text_frame.paragraphs[0]
        p.text = text
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.alignment = PP_ALIGN.CENTER

    rows_data = [
        ("Centralized ML Cloud", "Failed (Raw data pooled externally)", "Supported (if unified multi-task)", "Poor (Prone to batch effects)", "High (Full access to all images)"),
        ("Isolated Local Training", "Compliant (Data stays on-premise)", "Supported (Only local labels)", "Poor (Overfits to local scanner)", "Zero (No shared intelligence)"),
        ("Standard FedAvg (McMahan 2017)", "Compliant (Only gradients shared)", "Failed (Requires identical output dim)", "Poor (Batch norm drift across sites)", "Moderate (Crashes on label mismatch)"),
        ("FedMedDx (FedRep + FedBN)", "Compliant (Strict zero-image sharing)", "Supported (Decoupled private heads)", "High (Local BN absorbs site noise)", "High (Universal pulmonary features)"),
    ]

    for row_idx, row in enumerate(rows_data):
        is_ours = (row_idx == 3)
        for col_idx, val in enumerate(row):
            cell = table.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            if is_ours:
                cell.fill.fore_color.rgb = RGBColor(238, 242, 255)
            else:
                cell.fill.fore_color.rgb = COLOR_CARD_BG if row_idx % 2 == 0 else COLOR_HEADER_BG
            
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            if is_ours:
                p.font.bold = (col_idx == 0)
                p.font.color.rgb = COLOR_ACCENT_BLUE if col_idx == 0 else COLOR_PRIMARY_DARK
            else:
                p.font.bold = (col_idx == 0)
                p.font.color.rgb = COLOR_PRIMARY_DARK

    add_footer(s3, 3, 17)
    add_notes(s3, "Verbal cue: Standard federated averaging crashes when clients have different output dimensions. That is exactly the engineering gap we solve.")

    # ==========================================
    # SLIDE 4: Proposed Solution
    # ==========================================
    s4 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(s4, COLOR_BG_LIGHT)
    add_header(s4, "Proposed Solution: Decoupled Personalized Federated Learning")

    # 3 Column cards
    solution_cols = [
        ("Shared ResNet-18 Backbone",
         "Convolutional layers (conv1 through layer4) are collaboratively trained across all hospitals to learn universal pulmonary visual representations: infiltrates, consolidations, opacities, and pleural line patterns.",
         COLOR_ACCENT_BLUE),
        ("Persistent Local Heads",
         "Classification layers (W_fc) remain 100% private to each hospital client. Fully resolves the dimension mismatch between 4-class COVID-19 and binary screening models. Head weights are cached across rounds.",
         COLOR_TEAL),
        ("Federated Batch Normalization",
         "Implements FedBN (Li et al., ICLR 2021). Batch normalization running mean and variance are kept strictly local on each client PC, preventing domain drift caused by different X-ray scanner calibrations.",
         COLOR_PRIMARY_DARK),
    ]

    for i, (title, desc, color) in enumerate(solution_cols):
        left = Inches(0.8 + i * 4.0)
        top = Inches(1.5)
        
        add_card(s4, left, top, Inches(3.75), Inches(5.1))
        
        # Header banner
        b = s4.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, Inches(3.75), Inches(0.6))
        b.fill.solid()
        b.fill.fore_color.rgb = color
        b.line.fill.background()
        
        tb_h = s4.shapes.add_textbox(left, top, Inches(3.75), Inches(0.6))
        p = tb_h.text_frame.paragraphs[0]
        p.text = f"Pillar 0{i+1}"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.alignment = PP_ALIGN.CENTER
        
        tb = s4.shapes.add_textbox(left + Inches(0.25), top + Inches(0.8), Inches(3.25), Inches(4.0))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY_DARK
        
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(11)
        p2.font.color.rgb = COLOR_TEXT_MAIN
        p2.space_before = Pt(12)

    add_footer(s4, 4, 17)
    add_notes(s4, "Verbal cue: The shared backbone learns universal visual primitives; each hospital's private head translates those primitives into its own diagnostic prediction.")

    # ==========================================
    # SLIDE 5: Objectives
    # ==========================================
    s5 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(s5, COLOR_BG_LIGHT)
    add_header(s5, "Measurable Project Objectives")

    objectives = [
        ("Objective 1: Core Federated Engine",
         "Implement and verify a decoupled FedRep + FedBN aggregation engine using Flower NumPy client, supporting non-IID Dirichlet partitioning and checkpoint persistence.",
         "Metric: End-to-end simulation passing 20 rounds with zero tensor dimension mismatch errors."),
        ("Objective 2: 4 Verified Disease Modules",
         "Develop four modular CXR disease pipelines (COVID-19 4-class, Pneumonia binary, TB binary, Pediatric binary) adhering strictly to the CONTRACT.md standard.",
         "Metric: 100% modular compliance with get_model, train_one_round, evaluate, explain methods."),
        ("Objective 3: Accuracy Improvement over Local-Only",
         "Demonstrate that personalized federated training yields higher classification accuracy and F1-score compared to standalone local training on small hospital partitions.",
         "Metric: Benchmark accuracy improvement target > 5-10% across heterogeneous partitions."),
        ("Objective 4: Distributed Multi-Node LAN Execution",
         "Execute federated training across 4 physical participant computers connected over local Wi-Fi gRPC without centralized data leakage.",
         "Metric: Zero raw image transmission; only filtered 11.1M backbone weights exchanged."),
        ("Objective 5: Clinical Explainability Integration",
         "Integrate Grad-CAM visualization into each module to provide visual heatmaps confirming the model attends to pathological lung regions rather than background artifacts.",
         "Metric: High-resolution class-discriminative heatmap generated per scan in < 200ms."),
    ]

    for i, (title, desc, metric) in enumerate(objectives):
        top = Inches(1.4 + i * 1.05)
        add_card(s5, Inches(0.8), top, Inches(11.733), Inches(0.95))
        
        # Pill on left
        pill = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), top + Inches(0.18), Inches(0.6), Inches(0.6))
        pill.fill.solid()
        pill.fill.fore_color.rgb = COLOR_ACCENT_BLUE
        pill.line.fill.background()
        p = pill.text_frame.paragraphs[0]
        p.text = f"O{i+1}"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.alignment = PP_ALIGN.CENTER
        
        tb = s5.shapes.add_textbox(Inches(1.8), top + Inches(0.08), Inches(10.5), Inches(0.8))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title + "  |  " + desc
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY_DARK
        
        p2 = tf.add_paragraph()
        p2.text = metric
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = COLOR_TEAL
        p2.space_before = Pt(2)

    add_footer(s5, 5, 17)
    add_notes(s5, "Verbal cue: Every objective has a quantifiable target. We measure convergence, modular compliance, accuracy gain, and explainability latency.")

    # ==========================================
    # SLIDE 6: Scope Boundaries
    # ==========================================
    s6 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(s6, COLOR_BG_LIGHT)
    add_header(s6, "Project Scope and Explicit Boundaries")

    # In Scope Card
    add_card(s6, Inches(0.8), Inches(1.4), Inches(5.75), Inches(5.3), bg_color=COLOR_GREEN_BG, border_color=RGBColor(187, 247, 208))
    tb_in = s6.shapes.add_textbox(Inches(1.0), Inches(1.55), Inches(5.35), Inches(5.0))
    tf_in = tb_in.text_frame
    tf_in.word_wrap = True
    
    p = tf_in.paragraphs[0]
    p.text = "IN SCOPE (Delivered in 14-Day Sprint)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_GREEN

    in_scope_items = [
        "Unified 2D Chest Radiography (224x224 RGB, PNG/JPEG)",
        "4 verified public datasets (COVID, Pneumonia, TB, Pediatric)",
        "ResNet-18 backbone with task-specific classification heads",
        "FedRep alternating local training + FedAvg backbone aggregation",
        "FedBN local batch normalization to absorb scanner drift",
        "Non-IID Dirichlet partitioning (alpha = 0.5) simulation",
        "Flower gRPC LAN distributed network protocol (Port 8080)",
        "SQLite experiment database logging round metrics and runs",
        "Grad-CAM visual heatmap overlay generation",
    ]
    for item in in_scope_items:
        p = tf_in.add_paragraph()
        p.text = "- " + item
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_PRIMARY_DARK
        p.space_before = Pt(4)

    # Out of Scope Card
    add_card(s6, Inches(6.75), Inches(1.4), Inches(5.75), Inches(5.3), bg_color=COLOR_RED_BG, border_color=RGBColor(254, 202, 202))
    tb_out = s6.shapes.add_textbox(Inches(6.95), Inches(1.55), Inches(5.35), Inches(5.0))
    tf_out = tb_out.text_frame
    tf_out.word_wrap = True
    
    p = tf_out.paragraphs[0]
    p.text = "OUT OF SCOPE (Deferred / Future Work)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_RED

    out_scope_items = [
        "DICOM header parsing and 3D volumetric CT/MRI reconstruction",
        "Multi-organ cross-modality learning (Brain MRI, Retinal, Skin)",
        "Cryptographic Secure Aggregation (SecAgg) protocol",
        "Formal Differential Privacy (DP-SGD) with noise injection",
        "Production cloud deployment (AWS/GCP Kubernetes clusters)",
        "Live hospital EHR/PACS integration and IRB patient data",
        "Medical device regulatory compliance (FDA 510(k), CE mark)",
        "Mobile/embedded on-device inference optimization",
    ]
    for item in out_scope_items:
        p = tf_out.add_paragraph()
        p.text = "- " + item
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_PRIMARY_DARK
        p.space_before = Pt(4)

    add_footer(s6, 6, 17)
    add_notes(s6, "Verbal cue: We scoped tightly to a 14-day student sprint. We eliminated DICOM parsing and cloud infrastructure to focus 100% on the federated learning contribution.")

    # ==========================================
    # SLIDE 7: System Architecture
    # ==========================================
    s7 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(s7, COLOR_BG_LIGHT)
    add_header(s7, "System Architecture: 3-Tier Layered Design")

    tiers = [
        ("Tier 1: Decentralized Data Layer", 
         "Four independent, unshared Chest X-Ray datasets located on client hospital PCs.\n- Client 1: COVID-19 Radiography (4 classes, PNG)\n- Client 2: RSNA / Mooney Adult Pneumonia (2 classes, JPEG)\n- Client 3: TB Chest X-Ray Database (2 classes, PNG)\n- Client 4: Pediatric Pneumonia Database (2 classes, JPEG)",
         COLOR_ACCENT_BLUE),
        ("Tier 2: Client Node Execution Layer (Local Hospital PCs)",
         "Standardized CONTRACT.md interface: 224x224 RGB image transformation.\n- ResNet-18 Shared Backbone (Parameters extracted for aggregation)\n- Local Head Checkpoint: client_{id}_head.pth (Never transmitted)\n- Local Batch Normalization Statistics (FedBN: Absorbs scanner drift)\n- Grad-CAM Engine: layer4[-1] gradient activations for heatmaps",
         COLOR_TEAL),
        ("Tier 3: Central Aggregation & Logging Layer (Coordinator PC)",
         "Flower gRPC Server (0.0.0.0:8080) orchestrating rounds.\n- Parameter Filter: Filters out 'fc', 'bn', and 'downsample.1' weights\n- Federated Averaging (FedAvg) on shared convolutional backbone\n- SQLite Database (results/fedmeddx_experiments.db) for experiment auditing",
         COLOR_PRIMARY_DARK),
    ]

    for i, (title, desc, color) in enumerate(tiers):
        top = Inches(1.4 + i * 1.8)
        add_card(s7, Inches(0.8), top, Inches(11.733), Inches(1.65))
        
        # Left color strip
        b = s7.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), top, Inches(0.15), Inches(1.65))
        b.fill.solid()
        b.fill.fore_color.rgb = color
        b.line.fill.background()
        
        tb = s7.shapes.add_textbox(Inches(1.1), top + Inches(0.1), Inches(11.2), Inches(1.45))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = color
        
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(10)
        p2.font.color.rgb = COLOR_TEXT_MAIN
        p2.space_before = Pt(3)

    add_footer(s7, 7, 17)
    add_notes(s7, "Verbal cue: Patient data never leaves Tier 1. Only filtered 11.1M backbone weight tensors travel between Tier 2 and Tier 3 over gRPC.")

    # ==========================================
    # SLIDE 8: Technology Stack
    # ==========================================
    s8 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(s8, COLOR_BG_LIGHT)
    add_header(s8, "Technology Stack & Technical Justification")

    tech_table_shape = s8.shapes.add_table(7, 3, Inches(0.8), Inches(1.4), Inches(11.733), Inches(5.1))
    t_tech = tech_table_shape.table

    tech_headers = ["Component / Layer", "Technology Selected", "Technical Rationale & Justification"]
    for col_idx, text in enumerate(tech_headers):
        cell = t_tech.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY_DARK
        p = cell.text_frame.paragraphs[0]
        p.text = text
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT_WHITE

    tech_rows = [
        ("Language & Runtime", "Python 3.10+", "Broad library ecosystem for scientific computing, PyTorch, and Flower compatibility."),
        ("Deep Learning Framework", "PyTorch 2.0+ & Torchvision", "Dynamic computation graph, native CUDA acceleration, industry standard for medical imaging."),
        ("Federated Learning Core", "Flower (flwr 1.5+)", "Lightweight gRPC client-server transport; NumPy client API allows clean parameter filtering."),
        ("Feature Backbone", "ResNet-18 (ImageNet Pretrained)", "11.7M parameters; optimal balance between feature capacity and low VRAM footprint on consumer GPUs."),
        ("Explainability Engine", "Grad-CAM (grad-cam 1.4+)", "Extracts activation maps from layer4 to highlight anatomical pathology regions for radiologists."),
        ("Data & Experiment Logging", "SQLite 3 & Pandas", "Zero-dependency embedded SQL database; logs run metadata, hospital metrics, and loss curves."),
    ]

    for row_idx, (col0, col1, col2) in enumerate(tech_rows):
        for col_idx, val in enumerate([col0, col1, col2]):
            cell = t_tech.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_CARD_BG if row_idx % 2 == 0 else COLOR_HEADER_BG
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            p.font.bold = (col_idx < 2)
            p.font.color.rgb = COLOR_PRIMARY_DARK if col_idx != 1 else COLOR_ACCENT_BLUE

    add_footer(s8, 8, 17)
    add_notes(s8, "Verbal cue: Every library in our stack is open-source, reproducible, and runnable on consumer-grade hardware with GPU acceleration.")

    # ==========================================
    # SLIDE 9: Module Breakdown & Ownership
    # ==========================================
    s9 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(s9, COLOR_BG_LIGHT)
    add_header(s9, "Module Breakdown and Team Responsibilities")

    modules_table_shape = s9.shapes.add_table(6, 4, Inches(0.8), Inches(1.4), Inches(11.733), Inches(5.1))
    t_mod = modules_table_shape.table

    mod_headers = ["Module Path", "Assigned Owner", "Primary Functionality", "Output Dimension"]
    for col_idx, text in enumerate(mod_headers):
        cell = t_mod.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY_DARK
        p = cell.text_frame.paragraphs[0]
        p.text = text
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT_WHITE

    mod_rows = [
        ("federated/ (server, client, wrapper, db)", "Jalpan (Core Lead)", "Orchestrates Flower server, FedRep aggregation, SQLite logging, head checkpointing.", "Shared 11.1M Backbone"),
        ("modules/covid_module.py", "Priyanka (Client 1)", "COVID-19 Radiography pipeline, 4-class head, Dirichlet non-IID data partitioning.", "512 x 4 Classes"),
        ("modules/pneumonia_module.py", "Gargee (Client 2)", "RSNA Adult Pneumonia binary classification pipeline and local training loop.", "512 x 2 Classes"),
        ("modules/tb_module.py", "Smit (Client 3)", "Tuberculosis screening pipeline, class-weighted cross-entropy loss calculation.", "512 x 2 Classes"),
        ("modules/pediatric_module.py", "Hirva (Client 4)", "Pediatric Pneumonia binary classification (5,856 CXR images from Guangzhou Women's Hospital).", "512 x 2 Classes"),
    ]

    for row_idx, row in enumerate(mod_rows):
        for col_idx, val in enumerate(row):
            cell = t_mod.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_CARD_BG if row_idx % 2 == 0 else COLOR_HEADER_BG
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            p.font.bold = (col_idx <= 1)
            p.font.color.rgb = COLOR_PRIMARY_DARK if col_idx != 1 else COLOR_ACCENT_BLUE

    add_footer(s9, 9, 17)
    add_notes(s9, "Verbal cue: CONTRACT.md acts as our team contract. Each disease module developer adheres to the exact same 5-method interface.")

    # ==========================================
    # SLIDE 10: Algorithm & Parameter Filtering
    # ==========================================
    s10 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(s10, COLOR_BG_LIGHT)
    add_header(s10, "Algorithm: FedRep Alternating Optimization + Parameter Filtering")

    # Left: Steps
    add_card(s10, Inches(0.8), Inches(1.4), Inches(5.75), Inches(5.3))
    tb_alg = s10.shapes.add_textbox(Inches(1.0), Inches(1.55), Inches(5.35), Inches(5.0))
    tf_alg = tb_alg.text_frame
    tf_alg.word_wrap = True
    
    p = tf_alg.paragraphs[0]
    p.text = "FedRep Alternating Optimization (Round t)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_DARK

    alg_steps = [
        "1. Server Broadcast: Central server sends current aggregated backbone theta(t) to all clients.",
        "2. Phase 1 (Local Head Update): Client freezes the backbone theta. Trains local head w_i on local data for E_head epochs using Adam optimizer.",
        "3. Phase 2 (Local Representation Update): Client unfreezes backbone theta. Computes gradients on local data for E_rep epochs to adapt representation.",
        "4. Parameter Extraction: Client extracts ONLY shared backbone parameters, filtering out fc and bn weights.",
        "5. Global FedAvg Aggregation: Server computes weighted average of backbone weights: theta(t+1) = Sum((n_i / n) * theta_i).",
        "6. Checkpoint Persistence: Local head w_i is saved to checkpoints/client_{id}_head.pth on disk for next round.",
    ]
    for step in alg_steps:
        p = tf_alg.add_paragraph()
        p.text = step
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_TEXT_MAIN
        p.space_before = Pt(5)

    # Right: Parameter Filtering Code & Rule
    add_card(s10, Inches(6.75), Inches(1.4), Inches(5.75), Inches(5.3), bg_color=COLOR_SECONDARY_DARK, border_color=RGBColor(51, 65, 85))
    tb_code = s10.shapes.add_textbox(Inches(6.95), Inches(1.55), Inches(5.35), Inches(5.0))
    tf_code = tb_code.text_frame
    tf_code.word_wrap = True

    p = tf_code.paragraphs[0]
    p.text = "Strict Parameter Filtering Rule"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_CYAN

    p = tf_code.add_paragraph()
    p.text = "Implemented in federated/client_wrapper.py:"
    p.font.size = Pt(10)
    p.font.color.rgb = RGBColor(148, 163, 184)
    p.space_before = Pt(4)

    code_lines = [
        "def _is_shared_param(name: str) -> bool:",
        "    # Exclude classification head & batch norm",
        "    if 'fc' in name: return False",
        "    if 'bn' in name: return False",
        "    if 'downsample.1' in name: return False",
        "    return True",
        "",
        "# Uploaded to Server (11.1M Params):",
        "conv1, layer1.*.conv*, layer2.*.conv*,",
        "layer3.*.conv*, layer4.*.conv*",
        "",
        "# Kept 100% Local & Private:",
        "fc.weight, fc.bias (Different per hospital)",
        "bn*.running_mean, bn*.running_var (FedBN)",
    ]
    for line in code_lines:
        p = tf_code.add_paragraph()
        p.text = line
        p.font.size = Pt(9)
        p.font.color.rgb = RGBColor(226, 232, 240)
        p.space_before = Pt(2)

    add_footer(s10, 10, 17)
    add_notes(s10, "Verbal cue: This parameter filter solves both problems: the head exclusion handles dimension mismatch, while the batch norm exclusion eliminates domain drift.")

    # ==========================================
    # SLIDE 11: Work Completed Since Review 1
    # ==========================================
    s11 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(s11, COLOR_BG_LIGHT)
    add_header(s11, "Work Completed Since 1st Review")

    cards_data_s11 = [
        ("Federated Core Engine Built & Verified",
         "- federated/server.py: Standalone Flower server on port 8080 with metric aggregation.\n- federated/client_wrapper.py: FedRep client with head checkpointing & CUDA cache clearing.\n- federated/run_fedavg.py: Sequential simulation engine running without Ray overhead.\n- federated/db.py: SQLite experiment schema logging runs and round-by-round metrics.",
         COLOR_TEAL),
        ("4 Verified Disease Modules Implemented",
         "- modules/covid_module.py (Priyanka): 4-class COVID-19 Radiography pipeline.\n- modules/pneumonia_module.py (Gargee): Binary Adult Pneumonia detection pipeline.\n- modules/tb_module.py (Smit): Binary Tuberculosis screening pipeline.\n- modules/pediatric_module.py (Hirva): Pediatric Pneumonia (tolgadincer Kaggle dataset).",
         COLOR_ACCENT_BLUE),
        ("Comprehensive Technical Documentation",
         "- CONTRACT.md: Standardized interface specification for disease modules.\n- FEDERATED_LEARNING_PROPOSAL.md: Research proposal with mathematical proofs.\n- DATASET_GUIDE.md & MODALITY_MODULES.md: Detailed developer walkthroughs.\n- GitHub Wiki: Fully synchronized documentation available online.",
         COLOR_PRIMARY_DARK),
        ("Agile Project Management Configured",
         "- 20 color-coded GitHub issues created and assigned to team members.\n- Tagged by domain (COVID-CXR, Pneumonia-CXR, TB-CXR, Pediatric-CXR, Core-Lead).\n- Phase labels assigned from Day 1-2 to Day 13-14 for milestone tracking.",
         COLOR_ACCENT_CYAN),
    ]

    for i, (title, desc, accent) in enumerate(cards_data_s11):
        col = i % 2
        row = i // 2
        left = Inches(0.8 + col * 5.95)
        top = Inches(1.4 + row * 2.65)
        
        add_card(s11, left, top, Inches(5.75), Inches(2.45))
        
        b = s11.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, Inches(5.75), Inches(0.08))
        b.fill.solid()
        b.fill.fore_color.rgb = accent
        b.line.fill.background()

        tb = s11.shapes.add_textbox(left + Inches(0.2), top + Inches(0.18), Inches(5.35), Inches(2.1))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY_DARK
        
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = COLOR_TEXT_MAIN
        p2.space_before = Pt(6)

    add_footer(s11, 11, 17)
    add_notes(s11, "Verbal cue: The entire federated engine, all 4 disease modules, and the documentation suite are built, verified, and committed to GitHub.")

    # ==========================================
    # SLIDE 12: Timeline & 14-Day Work Plan
    # ==========================================
    s12 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(s12, COLOR_BG_LIGHT)
    add_header(s12, "Timeline: 14-Day Sprint Schedule & Milestones")

    plan_table_shape = s12.shapes.add_table(8, 4, Inches(0.8), Inches(1.4), Inches(11.733), Inches(5.1))
    t_plan = plan_table_shape.table

    plan_headers = ["Phase & Timeline", "Key Tasks & Deliverables", "Assigned Team", "Current Status"]
    for col_idx, text in enumerate(plan_headers):
        cell = t_plan.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY_DARK
        p = cell.text_frame.paragraphs[0]
        p.text = text
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT_WHITE

    plan_rows = [
        ("Phase 1 (Day 1-2)", "Kaggle API setup, dataset download, data inspection, directory creation.", "All Team Members", "COMPLETED"),
        ("Phase 2 (Day 3-4)", "Image preprocessing pipeline (224x224 RGB), centralized PyTorch baseline.", "Priyanka, Gargee, Smit, Hirva", "COMPLETED"),
        ("Phase 3 (Day 5-6)", "CONTRACT.md implementation (get_model, train_one_round, evaluate, explain).", "Priyanka, Gargee, Smit, Hirva", "COMPLETED"),
        ("Phase 4 (Day 7-8)", "FedRep simulation engine, parameter filtering, SQLite logging, head checkpointing.", "Jalpan (Core Lead)", "COMPLETED"),
        ("Phase 5 (Day 9-10)", "Distributed LAN execution across 4 physical PCs via Flower gRPC (Port 8080).", "All Team Members", "IN PROGRESS"),
        ("Phase 6 (Day 11-12)", "Non-IID Dirichlet robustness sweep (alpha = 0.1, 0.5, 1.0) & Grad-CAM validation.", "All Team Members", "UPCOMING"),
        ("Phase 7 (Day 13-14)", "Streamlit demo dashboard, final report generation, presentation rehearsals.", "All Team Members", "UPCOMING"),
    ]

    for row_idx, (col0, col1, col2, status) in enumerate(plan_rows):
        for col_idx, val in enumerate([col0, col1, col2, status]):
            cell = t_plan.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_CARD_BG if row_idx % 2 == 0 else COLOR_HEADER_BG
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(9.5)
            if col_idx == 3:
                p.font.bold = True
                if status == "COMPLETED":
                    p.font.color.rgb = COLOR_GREEN
                elif status == "IN PROGRESS":
                    p.font.color.rgb = COLOR_ACCENT_BLUE
                else:
                    p.font.color.rgb = COLOR_TEXT_MUTED
            else:
                p.font.bold = (col_idx == 0)
                p.font.color.rgb = COLOR_PRIMARY_DARK

    add_footer(s12, 12, 17)
    add_notes(s12, "Verbal cue: We are currently on schedule at Day 8. The engine and modules are complete; distributed multi-PC testing begins immediately next.")

    # ==========================================
    # SLIDE 13: Next Phase of Development
    # ==========================================
    s13 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(s13, COLOR_BG_LIGHT)
    add_header(s13, "Next Phase of Development: Immediate Deliverables")

    next_cards = [
        ("1. Multi-Machine LAN Testing (Day 9-10)",
         "Deploy Flower server on Jalpan's machine. Connect 4 student laptops over local Wi-Fi. Verify client connection handshakes, round synchronization, and zero connection dropouts.",
         COLOR_ACCENT_BLUE),
        ("2. Non-IID Dirichlet Sweep (Day 11-12)",
         "Evaluate model convergence and robustness across extreme non-IID data distributions (Dirichlet concentration alpha = 0.1 vs 0.5 vs 1.0) and measure accuracy stability.",
         COLOR_TEAL),
        ("3. Grad-CAM Pathological Validation (Day 11-12)",
         "Generate and visually inspect Grad-CAM heatmaps across test scans to confirm the model attends to true radiological abnormalities (opacities, consolidations) rather than corner artifacts.",
         COLOR_PRIMARY_DARK),
        ("4. Interactive Streamlit Demo (Day 13-14)",
         "Build a clinician demo interface allowing users to upload a CXR image, select a hospital diagnostic model, view classification confidence, and inspect the Grad-CAM heatmap.",
         COLOR_ACCENT_CYAN),
    ]

    for i, (title, desc, color) in enumerate(next_cards):
        col = i % 2
        row = i // 2
        left = Inches(0.8 + col * 5.95)
        top = Inches(1.4 + row * 2.65)
        
        add_card(s13, left, top, Inches(5.75), Inches(2.45))
        
        b = s13.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, Inches(5.75), Inches(0.08))
        b.fill.solid()
        b.fill.fore_color.rgb = color
        b.line.fill.background()

        tb = s13.shapes.add_textbox(left + Inches(0.2), top + Inches(0.2), Inches(5.35), Inches(2.0))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY_DARK
        
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = COLOR_TEXT_MAIN
        p2.space_before = Pt(8)

    add_footer(s13, 13, 17)
    add_notes(s13, "Verbal cue: Our highest priority for the next phase is validating Flower gRPC communication across physical laptops on campus Wi-Fi.")

    # ==========================================
    # SLIDE 14: Preliminary Results & Verification
    # ==========================================
    s14 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(s14, COLOR_BG_LIGHT)
    add_header(s14, "Preliminary Verification & Pipeline Integrity")

    res_cards = [
        ("Sequential Simulation Verified",
         "Successfully executed multi-round sequential simulation on run_fedavg.py across virtual hospital partitions with zero runtime exceptions.",
         COLOR_GREEN),
        ("Standalone Module Tests Passing",
         "All 4 disease modules (covid_module, pneumonia_module, tb_module, pediatric_module) pass standalone CONTRACT.md verification with synthetic test batches.",
         COLOR_GREEN),
        ("Experiment Database Populated",
         "SQLite database results/fedmeddx_experiments.db confirmed active. Correctly records experiment_runs, round_metrics, and hospital_distributions.",
         COLOR_GREEN),
        ("GPU VRAM Management Verified",
         "torch.cuda.empty_cache() integration confirmed; prevents memory leak across sequential client evaluation rounds on consumer GPU hardware.",
         COLOR_GREEN),
    ]

    for i, (title, desc, color) in enumerate(res_cards):
        top = Inches(1.4 + i * 1.3)
        add_card(s14, Inches(0.8), top, Inches(11.733), Inches(1.15))
        
        # Checkmark box
        pill = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), top + Inches(0.25), Inches(0.65), Inches(0.65))
        pill.fill.solid()
        pill.fill.fore_color.rgb = COLOR_GREEN_BG
        pill.line.color.rgb = COLOR_GREEN
        p = pill.text_frame.paragraphs[0]
        p.text = "[PASS]"
        p.font.size = Pt(9)
        p.font.bold = True
        p.font.color.rgb = COLOR_GREEN
        p.alignment = PP_ALIGN.CENTER
        
        tb = s14.shapes.add_textbox(Inches(1.8), top + Inches(0.12), Inches(10.5), Inches(0.9))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY_DARK
        
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(10)
        p2.font.color.rgb = COLOR_TEXT_MAIN
        p2.space_before = Pt(2)

    add_footer(s14, 14, 17)
    add_notes(s14, "Verbal cue: The pipeline infrastructure is 100% proven end-to-end. We are ready to execute federated training on the full Kaggle datasets.")

    # ==========================================
    # SLIDE 15: Summary & Key Takeaways
    # ==========================================
    s15 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(s15, COLOR_BG_LIGHT)
    add_header(s15, "Summary and Key Takeaways")

    summary_items = [
        ("Architecture Solves Heterogeneity",
         "Decoupling the ResNet-18 backbone from private classification heads allows hospitals with completely different disease tasks to collaborate without tensor dimension crashes.",
         COLOR_ACCENT_BLUE),
        ("FedBN Eliminates Scanner Drift",
         "Keeping batch normalization layers local prevents image contrast and scanner noise differences across clinical centers from corrupting shared representation learning.",
         COLOR_TEAL),
        ("Strict Privacy & Zero Data Leakage",
         "Raw patient Chest X-Rays never leave hospital nodes. Only abstract convolutional feature weights are exchanged over the network.",
         COLOR_PRIMARY_DARK),
        ("Feasible 14-Day Delivery",
         "The federated core, modular pipelines, contract specifications, and issue tracking are fully established, ensuring on-time delivery for final demonstration.",
         COLOR_ACCENT_CYAN),
    ]

    for i, (title, desc, color) in enumerate(summary_items):
        top = Inches(1.4 + i * 1.3)
        add_card(s15, Inches(0.8), top, Inches(11.733), Inches(1.15))
        
        b = s15.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), top, Inches(0.15), Inches(1.15))
        b.fill.solid()
        b.fill.fore_color.rgb = color
        b.line.fill.background()
        
        tb = s15.shapes.add_textbox(Inches(1.1), top + Inches(0.12), Inches(11.2), Inches(0.9))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = f"Takeaway 0{i+1}: {title}"
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = color
        
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(10)
        p2.font.color.rgb = COLOR_TEXT_MAIN
        p2.space_before = Pt(2)

    add_footer(s15, 15, 17)
    add_notes(s15, "Verbal cue: We have moved from an abstract project concept to a working, verified federated diagnostic engine.")

    # ==========================================
    # SLIDE 16: Risks & Mitigations
    # ==========================================
    s16 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(s16, COLOR_BG_LIGHT)
    add_header(s16, "Project Risks and Engineering Mitigations")

    risk_table_shape = s16.shapes.add_table(6, 4, Inches(0.8), Inches(1.4), Inches(11.733), Inches(5.1))
    t_risk = risk_table_shape.table

    risk_headers = ["Identified Risk", "Impact / Severity", "Probability", "Actionable Engineering Mitigation"]
    for col_idx, text in enumerate(risk_headers):
        cell = t_risk.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY_DARK
        p = cell.text_frame.paragraphs[0]
        p.text = text
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT_WHITE

    risk_rows = [
        ("Campus Wi-Fi Firewall blocks gRPC port 8080", "HIGH", "MEDIUM", "Fallback to local mobile hotspot or single-machine multi-process simulation via run_fedavg.py."),
        ("Consumer GPU VRAM exhaustion during training", "MEDIUM", "MEDIUM", "Implemented torch.cuda.empty_cache() per round; batch size reduced to 16/32 with gradient accumulation."),
        ("Extreme Non-IID label skew causes divergence", "MEDIUM", "LOW", "FedRep alternating optimization separates head training from representation update, stabilizing training."),
        ("Dataset class imbalance (e.g. COVID 4-class)", "LOW", "HIGH", "Implemented class-weighted CrossEntropyLoss with inverse frequency weights in disease modules."),
        ("Client dropout / network latency spikes", "MEDIUM", "LOW", "Flower server configured with configurable min_fit_clients and fault-tolerant round aggregation."),
    ]

    for row_idx, (r0, r1, r2, r3) in enumerate(risk_rows):
        for col_idx, val in enumerate([r0, r1, r2, r3]):
            cell = t_risk.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_CARD_BG if row_idx % 2 == 0 else COLOR_HEADER_BG
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(9.5)
            if col_idx == 1:
                p.font.bold = True
                p.font.color.rgb = COLOR_RED if val == "HIGH" else (COLOR_ACCENT_BLUE if val == "MEDIUM" else COLOR_GREEN)
            elif col_idx == 0:
                p.font.bold = True
                p.font.color.rgb = COLOR_PRIMARY_DARK
            else:
                p.font.color.rgb = COLOR_PRIMARY_DARK

    add_footer(s16, 16, 17)
    add_notes(s16, "Verbal cue: Our primary risk is campus networking. We have already prepared a dedicated mobile hotspot and an offline simulation fallback.")

    # ==========================================
    # SLIDE 17: Q&A / Closing (Dark Theme)
    # ==========================================
    s17 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(s17, COLOR_PRIMARY_DARK)

    # Accent decorative bar
    bar = s17.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.5), Inches(0.15), Inches(1.8))
    bar.fill.solid()
    bar.fill.fore_color.rgb = COLOR_ACCENT_CYAN
    bar.line.fill.background()

    tb_q = s17.shapes.add_textbox(Inches(1.2), Inches(1.4), Inches(11.0), Inches(2.0))
    tf_q = tb_q.text_frame
    tf_q.word_wrap = True
    
    p = tf_q.paragraphs[0]
    p.text = "Questions & Discussion"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEXT_WHITE

    p2 = tf_q.add_paragraph()
    p2.text = "FedMedDx: Multi-Disease Representation Learning on Unified Chest Radiography"
    p2.font.size = Pt(16)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_ACCENT_CYAN
    p2.space_before = Pt(6)

    # Discussion Topics Card
    add_card(s17, Inches(0.8), Inches(3.6), Inches(11.733), Inches(3.0), bg_color=COLOR_SECONDARY_DARK, border_color=RGBColor(51, 65, 85))
    tb_disc = s17.shapes.add_textbox(Inches(1.1), Inches(3.8), Inches(11.1), Inches(2.6))
    tf_d = tb_disc.text_frame
    tf_d.word_wrap = True

    p = tf_d.paragraphs[0]
    p.text = "ANTICIPATED TECHNICAL DISCUSSION TOPICS:"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_CYAN

    topics = [
        "1. Mathematical convergence guarantees of FedRep vs standard FedAvg under high label skew.",
        "2. Why local batch normalization (FedBN) is necessary to eliminate scanner contrast drift.",
        "3. Network parameter filtering implementation in PyTorch and Flower NumPy client.",
        "4. Grad-CAM visual heatmap generation on layer4 activation maps for clinical verification.",
    ]
    for topic in topics:
        p = tf_d.add_paragraph()
        p.text = topic
        p.font.size = Pt(10.5)
        p.font.color.rgb = RGBColor(226, 232, 240)
        p.space_before = Pt(4)

    p_repo = tf_d.add_paragraph()
    p_repo.text = "GitHub Repository: https://github.com/Jalpan04/FedMedDx"
    p_repo.font.size = Pt(11)
    p_repo.font.bold = True
    p_repo.font.color.rgb = COLOR_ACCENT_BLUE
    p_repo.space_before = Pt(10)

    add_footer(s17, 17, 17, dark=True)
    add_notes(s17, "Verbal cue: Thank you for your time. We welcome questions on our architecture, algorithm, or experimental setup.")

    # Save presentation
    prs.save(output_path)
    print(f"Presentation saved successfully to: {output_path}")

if __name__ == "__main__":
    build_presentation()
