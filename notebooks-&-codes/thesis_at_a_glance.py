import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from textwrap import fill

# -------------------- Reported facts --------------------
TITLE = "MACHINE LEARNING FOR CREDIT RISK MODELING"
SUBTITLE = "A comparative study of traditional and explainable AI approaches"
DATASET = "OpenIntro Loans Full Schema"
N_LOANS, N_FEATURES, DEFAULT_RATE = "10,000", "55", "1.7%"
MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
AUC = {"Logistic Regression": "0.58", "Random Forest": "0.670", "XGBoost": "0.65", "LightGBM": "0.659"}
FINAL_THRESHOLD, PRECISION, RECALL = "~3.4%", "7.1%", "32.1%"
FPR_LOW, FPR_HIGH = "0.07%", "0.13%"
DRIVERS = ["interest rate", "debt-to-income", "loan grade"]

# -------------------- Style --------------------
plt.rcParams.update({"font.family":"DejaVu Sans", "svg.fonttype":"none", "pdf.fonttype":42,
                     "axes.titleweight":"bold"})
NAVY="#102A43"; BLUE="#1F6FEB"; TEAL="#0F766E"; GOLD="#D97706"; RED="#B42318"
INK="#172B4D"; MUTED="#52606D"; PALE="#F4F7FB"; GRID="#D9E2EC"
fig = plt.figure(figsize=(8.27, 11.69), dpi=220, facecolor="white")
ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

def box(x,y,w,h,title,body,color=BLUE, fs=8.0):
    patch=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.008,rounding_size=0.012",
                         linewidth=1.2,edgecolor=color,facecolor="white")
    ax.add_patch(patch)
    ax.text(x+0.014,y+h-0.026,title,fontsize=10,fontweight="bold",color=color,va="top")
    ax.text(x+0.014,y+h-0.060,fill(body, width=max(20,int(w*145))),fontsize=fs,color=INK,va="top",linespacing=1.25)

def arrow(x1,y1,x2,y2,color=BLUE):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=12,
                                 linewidth=1.5,color=color,shrinkA=5,shrinkB=5))

def metric(x,y,value,label,color):
    ax.text(x,y,value,ha="center",va="center",fontsize=17,fontweight="bold",color=color)
    ax.text(x,y-0.026,label,ha="center",va="top",fontsize=7.2,color=MUTED)

# Header
ax.add_patch(FancyBboxPatch((0.035,0.925),0.93,0.055,boxstyle="round,pad=0.01",facecolor=NAVY,edgecolor=NAVY))
ax.text(0.055,0.963,TITLE,fontsize=17,fontweight="bold",color="white",va="top")
ax.text(0.055,0.938,SUBTITLE,fontsize=8.5,color="#D9E2EC",va="top")
ax.text(0.95,0.938,"THESIS AT A GLANCE",fontsize=7.5,color="#B3D4FC",ha="right",va="top",fontweight="bold")

# Framing row
box(0.05,0.805,0.27,0.09,"QUESTION", "Can explainable ML improve minority-class default detection without sacrificing transparency?", TEAL, 7.5)
box(0.365,0.805,0.27,0.09,"DATA", f"{DATASET}: {N_LOANS} loans, {N_FEATURES} variables, {DEFAULT_RATE} default rate.", BLUE, 7.5)
box(0.68,0.805,0.27,0.09,"DESIGN", "70/30 hold-out split; random seed 42; SMOTE applied only to training data.", GOLD, 7.5)

# Pipeline
ax.text(0.05,0.775,"METHODS PIPELINE",fontsize=10.5,fontweight="bold",color=NAVY)
steps=[("1","Select\npre-origination\nvariables"),("2","Split\ntrain / test"),("3","SMOTE\ntraining only"),("4","Train 4\nmodels"),("5","Tune\nthreshold"),("6","Explain +\naudit")]
xs=[.06,.205,.35,.495,.64,.785]
for (n,lab),x in zip(steps,xs):
    ax.add_patch(FancyBboxPatch((x,.700),.105,.052,boxstyle="round,pad=0.006,rounding_size=.008",facecolor=PALE,edgecolor=BLUE,linewidth=1.1))
    ax.text(x+.012,.738,n,fontsize=9,fontweight="bold",color=BLUE,va="top")
    ax.text(x+.052,.726,lab,fontsize=7.0,color=INK,ha="center",va="top",linespacing=1.1)
for x in xs[:-1]: arrow(x+.108,.726,x+.139,.726)

# Experiments/results
ax.text(0.05,0.665,"EXPERIMENTS AND RESULTS",fontsize=10.5,fontweight="bold",color=NAVY)
box(0.05,0.525,0.43,0.115,"MODEL COMPARISON", "Four candidates: Logistic Regression, Random Forest, XGBoost, and LightGBM. Random Forest had the highest global ROC AUC, but it collapsed to the majority class at threshold 0.50. LightGBM retained useful minority-class sensitivity.", BLUE, 7.5)
box(0.52,0.525,0.43,0.115,"SELECTED OPERATING POINT", "LightGBM was selected for practical default detection. At the exploratory threshold ~3.4%, Precision = 7.1% and Recall = 32.1%. These figures are optimistic because the hold-out test set was used for threshold calibration.", GOLD, 7.5)

# metrics band
ax.add_patch(FancyBboxPatch((0.05,0.425),0.90,0.075,boxstyle="round,pad=0.008",facecolor=PALE,edgecolor=GRID))
metric(.17,.465,AUC["Random Forest"],"best ROC AUC",BLUE); metric(.38,.465,AUC["LightGBM"],"selected ROC AUC",TEAL)
metric(.59,.465,PRECISION,"precision @ ~3.4%",GOLD); metric(.80,.465,RECALL,"recall @ ~3.4%",RED)

# Explainability fairness
box(0.05,0.285,0.43,0.105,"EXPLAINABILITY", "SHAP identified the main risk drivers as interest rate, debt-to-income ratio, and loan grade. This connects model predictions to borrower-level and global explanations.", TEAL, 7.5)
box(0.52,0.285,0.43,0.105,"FAIRNESS CHECK", f"Using income as a proxy and False Positive Rate parity: low-income FPR = {FPR_LOW}; high-income FPR = {FPR_HIGH}. No evidence of disparate impact was found under the evaluated metric.", TEAL, 7.5)

# Conclusions
ax.text(0.05,0.255,"CONCLUSIONS AND LIMITATIONS",fontsize=10.5,fontweight="bold",color=NAVY)
box(0.05,0.105,0.58,0.125,"TAKEAWAY", "A constrained LightGBM framework can provide useful minority-class discrimination while remaining interpretable through SHAP. Deployment requires leakage controls, validation-based threshold selection, monitoring, and governance.", NAVY, 8.0)
box(0.665,0.105,0.285,0.125,"LIMITATIONS", "Single dataset; severe imbalance; income used only as a proxy; test-set threshold calibration introduces optimistic bias; results are not a definitive deployment estimate.", RED, 7.5)

# footer
ax.text(0.05,0.055,"Reproducibility: Python 3.13.5 | scikit-learn 1.8 | LightGBM 4.7 | imbalanced-learn 0.11 | seed = 42",fontsize=6.4,color=MUTED)
ax.text(0.95,0.055,"Sana Ur Rehman Arain | 2026",fontsize=6.4,color=MUTED,ha="right")

fig.savefig("thesis_at_a_glance.svg", bbox_inches="tight", pad_inches=0.08)
fig.savefig("thesis_at_a_glance.pdf", bbox_inches="tight", pad_inches=0.08)
fig.savefig("thesis_at_a_glance.png", bbox_inches="tight", pad_inches=0.08, dpi=300)
plt.close(fig)

print("Created thesis_at_a_glance.svg, thesis_at_a_glance.pdf, thesis_at_a_glance.png")
