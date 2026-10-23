"""
Local High-Resolution Colorful Architecture Diagram Generator for FedMedDx.
Generates paper/figs/architecture.png with modern academic-tier styling,
vibrant categorical color palettes, nested component cards, badges, and zero text overflow.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper", "figs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "architecture.png")

# Canvas setup: 100 x 30 for spacious, perfectly padded layout
fig = plt.figure(figsize=(16.0, 4.4), dpi=300)
ax = fig.add_subplot(111)
ax.set_xlim(0, 100)
ax.set_ylim(0, 30)
ax.axis("off")

# Typography
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]

def draw_card_base(x, y, w, h, title, banner_color, bg_color, border_color):
    """Draws the main container card and header banner."""
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.3,rounding_size=1.2",
        facecolor=bg_color, edgecolor=border_color, linewidth=1.5, zorder=2
    )
    ax.add_patch(rect)

    banner_h = 4.4
    banner = patches.FancyBboxPatch(
        (x, y + h - banner_h), w, banner_h,
        boxstyle="round,pad=0.15,rounding_size=0.9",
        facecolor=banner_color, edgecolor="none", zorder=3
    )
    ax.add_patch(banner)

    ax.text(x + w / 2, y + h - banner_h / 2, title, color="white", fontsize=9.2,
            fontweight="bold", ha="center", va="center", zorder=4)

def draw_sub_box(x, y, w, h, title, items, fill_color, border_color, title_color="#0f172a"):
    """Draws a nested rounded box inside a card for rich visual hierarchy."""
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        facecolor=fill_color, edgecolor=border_color, linewidth=1.0, zorder=3
    )
    ax.add_patch(box)
    
    ax.text(x + w / 2, y + h - 1.8, title, color=title_color, fontsize=7.6,
            fontweight="bold", ha="center", va="center", zorder=4)
    
    cur_y = y + h - 3.8
    for it in items:
        ax.text(x + 1.2, cur_y, it, color="#334155", fontsize=6.8,
                ha="left", va="center", zorder=4)
        cur_y -= 1.85

def draw_pill(x, y, w, h, text, fill_color, border_color, text_color):
    """Draws a pill tag."""
    pill = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.5",
        facecolor=fill_color, edgecolor=border_color, linewidth=0.8, zorder=4
    )
    ax.add_patch(pill)
    ax.text(x + w / 2, y + h / 2, text, color=text_color, fontsize=6.9,
            fontweight="bold", ha="center", va="center", zorder=5)

card_y = 1.0
card_h = 28.0
card_w = 17.6

# -------------------------------------------------------------
# 1. Clinical Data Ingestion (Sky Blue Theme)
# -------------------------------------------------------------
draw_card_base(
    0.8, card_y, card_w, card_h,
    "1. Clinical Data Ingestion",
    "#0284c7", "#f0f9ff", "#7dd3fc"
)

# 4 Clinical Modalities as colored pills
pills = [
    ("COVID-19 Radiography (4 Cls)", "#e0f2fe", "#38bdf8", "#0369a1"),
    ("Adult Pneumonia CXR (Binary)", "#dbeafe", "#60a5fa", "#1d4ed8"),
    ("Tuberculosis CXR (Binary)", "#e0e7ff", "#818cf8", "#4338ca"),
    ("Pediatric CXR (Binary)", "#f1f5f9", "#94a3b8", "#334155"),
]
py = card_y + card_h - 7.2
for txt, fc, ec, tc in pills:
    draw_pill(1.6, py, card_w - 1.6, 2.5, txt, fc, ec, tc)
    py -= 3.1

# Preprocessing Sub-box
draw_sub_box(
    1.6, card_y + 1.8, card_w - 1.6, 9.8,
    "Non-IID Preprocessing",
    [
        "* Standardized RGB (3, 224, 224)",
        "* CLAHE Contrast Normalization",
        "* Dirichlet Skew (alpha = 0.5)",
        "* Strict Zero-PHI Ingestion"
    ],
    "#e0f2fe", "#bae6fd", "#0284c7"
)

# Arrow 1 -> 2
ax.annotate("", xy=(20.2, 15.0), xytext=(18.6, 15.0),
            arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#0284c7", mutation_scale=14))

# -------------------------------------------------------------
# 2. Uniform Modality Contract (Purple Theme)
# -------------------------------------------------------------
draw_card_base(
    20.5, card_y, card_w, card_h,
    "2. Uniform Modality Contract",
    "#7c3aed", "#faf5ff", "#c084fc"
)

# Sub-box: CONTRACT.md Standard API
draw_sub_box(
    21.3, card_y + 8.8, card_w - 1.6, 14.0,
    "Standardized 5-Function API",
    [
        "get_model() -> ResNet-18",
        "get_hospital_partitions()",
        "train_one_round() [FedRep]",
        "evaluate() -> Acc, F1, AUC",
        "explain() -> Grad-CAM Map",
        "* Unifies Heterogeneous Tasks"
    ],
    "#f3e8ff", "#d8b4fe", "#6b21a8"
)

# Pill: Decoupled Integration
draw_pill(21.3, card_y + 1.8, card_w - 1.6, 5.2, "Modular Modality Plug-in\nZero Cross-Dependency\nStrict Interface Typing", "#ede9fe", "#a855f7", "#581c87")

# Arrow 2 -> 3
ax.annotate("", xy=(39.9, 15.0), xytext=(38.3, 15.0),
            arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#7c3aed", mutation_scale=14))

# -------------------------------------------------------------
# 3. Decoupled Model Architecture (Teal/Emerald Theme)
# -------------------------------------------------------------
draw_card_base(
    40.2, card_y, card_w + 0.4, card_h,
    "3. Decoupled Architecture",
    "#0d9488", "#f0fdfa", "#5eead4"
)

# Sub-box A: Globally Shared Representation
draw_sub_box(
    41.0, card_y + 13.2, card_w - 1.2, 9.6,
    "Global Shared Backbone (f_theta)",
    [
        "* ResNet-18 (conv1 -> layer4)",
        "* 512-dim Feature Representation",
        "* Aggregated Across Hospitals"
    ],
    "#ccfbf1", "#2dd4bf", "#0f766e"
)

# Sub-box B: Locally Private Personalization
draw_sub_box(
    41.0, card_y + 1.8, card_w - 1.2, 10.0,
    "Local Personalization (FedBN)",
    [
        "* Local FedBN (beta_k): Scanner Norm",
        "* Private Task Head (g_phi_k): C_k classes",
        "* No Head Parameter Communication",
        "* Eliminates Task Mismatch"
    ],
    "#e6fffa", "#99f6e4", "#115e59"
)

# Arrow 3 -> 4
ax.annotate("", xy=(60.0, 15.0), xytext=(58.4, 15.0),
            arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#0d9488", mutation_scale=14))

# -------------------------------------------------------------
# 4. Flower Federation Core (Amber Theme)
# -------------------------------------------------------------
draw_card_base(
    60.3, card_y, card_w + 0.4, card_h,
    "4. Flower Federation Engine",
    "#d97706", "#fffbeb", "#fcd34d"
)

# Sub-box A: Hospital Clients
draw_sub_box(
    61.1, card_y + 13.2, card_w - 1.2, 9.6,
    "Hospital Clients (k = 1...4)",
    [
        "* Phase 1: Local Head SGD (Frozen theta)",
        "* Phase 2: Backbone SGD (Frozen phi_k)",
        "* Transmits only backbone delta_theta"
    ],
    "#fef3c7", "#fde68a", "#b45309"
)

# Sub-box B: Coordinator Server
draw_sub_box(
    61.1, card_y + 1.8, card_w - 1.2, 10.0,
    "Flower Central Coordinator",
    [
        "* Sample-Weighted FedAvg",
        "* Global Backbone Aggregation",
        "* Asynchronous gRPC Transport",
        "* SQLite Telemetry Logging"
    ],
    "#fff7ed", "#fed7aa", "#c2410c"
)

# Arrow 4 -> 5
ax.annotate("", xy=(80.1, 15.0), xytext=(78.5, 15.0),
            arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#d97706", mutation_scale=14))

# -------------------------------------------------------------
# 5. Diagnostic Triage & XAI (Crimson/Rose Theme)
# -------------------------------------------------------------
draw_card_base(
    80.4, card_y, card_w + 1.0, card_h,
    "5. Diagnostic Triage & XAI",
    "#e11d48", "#fff1f2", "#fda4af"
)

# Sub-box A: Clinician Dashboard
draw_sub_box(
    81.2, card_y + 13.8, card_w - 0.6, 9.0,
    "Clinician Web App (Streamlit)",
    [
        "* Real-Time Multi-Disease Triage",
        "* Sub-Second Local Inference",
        "* Zero-Cloud Upload (On-Premise)"
    ],
    "#ffe4e6", "#fecdd3", "#be123c"
)

# Sub-box B: Explainability & Compliance
draw_sub_box(
    81.2, card_y + 1.8, card_w - 0.6, 10.6,
    "Edge Explainability & Privacy",
    [
        "* Edge Grad-CAM Saliency Maps",
        "* layer4[-1] Feature Gradients",
        "* Alpha-blended Heatmap Overlays",
        "* HIPAA / GDPR Compliant (No PHI)"
    ],
    "#fff1f2", "#fda4af", "#9f1239"
)

plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Successfully generated colorful concise architecture at: {out_path}")
