import urllib.request
import base64
import json
import os

graph = """flowchart LR
    subgraph Data ["1. Clinical Data Ingestion"]
        direction TB
        D1["COVID-19 Radiography (4-Class)"]
        D2["Adult Pneumonia CXR (Binary)"]
        D3["Tuberculosis CXR (Binary)"]
        D4["Pediatric Pneumonia (Binary)"]
        Part["Dirichlet Partitioning (alpha=0.5)"]
        D1 --> Part
        D2 --> Part
        D3 --> Part
        D4 --> Part
    end

    subgraph Contract ["2. Modality Contract"]
        API["CONTRACT.md Interface<br/>- get_model()<br/>- train_one_round()<br/>- evaluate(), explain()"]
    end

    subgraph Model ["3. Decoupled ResNet-18"]
        direction TB
        BB["Global Backbone f_theta<br/>(Shared conv1 to layer4)"]
        BN["Local FedBN beta_k<br/>(Scanner Adaptation)"]
        Head["Local Head W_fc phi_k<br/>(Task Personalization)"]
        BB --- BN
        BB --> Head
    end

    subgraph Fed ["4. Flower Federation Core"]
        direction TB
        Clients["Distributed Hospital Clients<br/>Phase 1: Local Head Update<br/>Phase 2: Backbone Optimization"]
        Server["Central Flower Server<br/>Sample-Weighted FedAvg<br/>SQLite Telemetry Logging"]
        Clients <-->|Backbone Weights Only| Server
    end

    subgraph Edge ["5. Diagnostic Inference"]
        direction TB
        App["Streamlit Clinician Dashboard<br/>Sub-second Offline Triage"]
        XAI["Grad-CAM Saliency Maps<br/>(ResNet-18 layer4[-1] Heatmaps)"]
        App --> XAI
    end

    Data --> Contract
    Contract --> Model
    Model --> Clients
    Server -.->|Aggregated Backbone| App

    %% Professional Color Styling
    style Data fill:#f0f9ff,stroke:#0284c7,stroke-width:1.5px,color:#0369a1
    style Contract fill:#f5f3ff,stroke:#7c3aed,stroke-width:1.5px,color:#6d28d9
    style Model fill:#ecfdf5,stroke:#059669,stroke-width:1.5px,color:#047857
    style Fed fill:#fffbeb,stroke:#d97706,stroke-width:1.5px,color:#b45309
    style Edge fill:#fdf2f8,stroke:#db2777,stroke-width:1.5px,color:#be185d

    style BB fill:#0284c7,stroke:#0369a1,color:#ffffff,font-weight:bold
    style BN fill:#0d9488,stroke:#0f766e,color:#ffffff,font-weight:bold
    style Head fill:#d97706,stroke:#b45309,color:#ffffff,font-weight:bold
    style Server fill:#1e293b,stroke:#0f172a,color:#ffffff,font-weight:bold
    style XAI fill:#dc2626,stroke:#b91c1c,color:#ffffff,font-weight:bold
    style Part fill:#2563eb,stroke:#1d4ed8,color:#ffffff,font-weight:bold
    style API fill:#7c3aed,stroke:#6d28d9,color:#ffffff,font-weight:bold
"""

obj = {
    "code": graph,
    "mermaid": {
        "theme": "default",
        "themeVariables": {
            "fontSize": "13px",
            "fontFamily": "arial"
        }
    }
}

enc = base64.b64encode(json.dumps(obj).encode("utf-8")).decode("ascii")
url = f"https://mermaid.ink/img/{enc}"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper", "figs", "architecture.png"))
with urllib.request.urlopen(req, timeout=20) as res:
    data = res.read()
    with open(out_path, "wb") as f:
        f.write(data)

print(f"Successfully generated colorful concise Mermaid architecture at: {out_path} ({len(data)} bytes)")
