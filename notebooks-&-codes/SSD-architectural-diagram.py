import matplotlib.patches as patches
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 3.2), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 3.5)
ax.axis("off")

stages = [
    {
        "title": "Stage 1: Ingestion & Scrub",
        "file": "01_data_cleaning.ipynb\n03_feature_engineering.ipynb",
        "bullets": "• Temporal boundary enforcement\n• Purge post-origination leakage\n• Encode & isolate SMOTE balancing",
        "color": "#E3F2FD",
        "border": "#1976D2",
    },
    {
        "title": "Stage 2: Modeling Engine",
        "file": "04_model_training.ipynb\n08_Threshold_Optimization.ipynb",
        "bullets": "• LightGBM leaf-wise boosting\n• Probability calibration\n• Dynamic ~3.4% risk threshold",
        "color": "#E8F5E9",
        "border": "#388E3C",
    },
    {
        "title": "Stage 3: Explainability (XAI)",
        "file": "05_explainability_analysis.ipynb",
        "bullets": "• SHAP TreeExplainer attributions\n• Global risk-driver hierarchy\n• Adverse-action reason codes",
        "color": "#FFF3E0",
        "border": "#F57C00",
    },
    {
        "title": "Stage 4: Fairness & Audit",
        "file": "06_fairness_compliance.ipynb\n07_final_report_visuals.ipynb",
        "bullets": "• Disparate impact verification\n• Basel III probability calibration\n• Underwriter triage dashboard",
        "color": "#F3E5F5",
        "border": "#7B1FA2",
    },
]

box_width = 2.45
box_height = 2.6
y_pos = 0.45
gap = 0.45
x_start = 0.4

for i, stage in enumerate(stages):
  x = x_start + i * (box_width + gap)

  # Draw Stage Box
  rect = patches.FancyBboxPatch(
      (x, y_pos),
      box_width,
      box_height,
      boxstyle="round,pad=0.08,rounding_size=0.15",
      facecolor=stage["color"],
      edgecolor=stage["border"],
      linewidth=1.8,
  )
  ax.add_patch(rect)

  # Title
  ax.text(
      x + box_width / 2,
      y_pos + box_height - 0.35,
      stage["title"],
      ha="center",
      va="center",
      fontsize=10.5,
      fontweight="bold",
      color="#212121",
  )

  # Code mapping pill
  pill = patches.FancyBboxPatch(
      (x + 0.15, y_pos + box_height - 0.95),
      box_width - 0.3,
      0.45,
      boxstyle="round,pad=0.04,rounding_size=0.08",
      facecolor="white",
      edgecolor=stage["border"],
      linewidth=0.8,
  )
  ax.add_patch(pill)

  ax.text(
      x + box_width / 2,
      y_pos + box_height - 0.73,
      stage["file"],
      ha="center",
      va="center",
      fontsize=7.5,
      fontfamily="monospace",
      color="#37474F",
  )

  # Bullets
  ax.text(
      x + 0.18,
      y_pos + 0.75,
      stage["bullets"],
      ha="left",
      va="center",
      fontsize=8,
      color="#333333",
      linespacing=1.45,
  )

  # Transition Arrows
  if i < len(stages) - 1:
    arrow_start = x + box_width + 0.05
    arrow_end = arrow_start + gap - 0.1
    ax.annotate(
        "",
        xy=(arrow_end, y_pos + box_height / 2),
        xytext=(arrow_start, y_pos + box_height / 2),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#455A64",
            lw=2.2,
            mutation_scale=15,
        ),
    )

plt.title(
    "Credit Risk Architecture & Pipeline Code Mapping",
    fontsize=13,
    fontweight="bold",
    pad=15,
    color="#1A237E",
)
plt.tight_layout()
plt.savefig(
    "../plots/sdd_solution_architecture.png", dpi=300, bbox_inches="tight"
)
plt.show()