"""
Generate the Credit Recourse Engine technical report as a PDF.
Run from project root:  python docs/generate_report.py
"""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
    Table, TableStyle, PageBreak, HRFlowable, KeepTogether
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.colors import HexColor

# ── Colour palette ─────────────────────────────────────────────────────────
C_BG        = HexColor("#0f172a")
C_SURFACE   = HexColor("#1e293b")
C_BORDER    = HexColor("#334155")
C_BRAND     = HexColor("#6366f1")
C_BRAND_LT  = HexColor("#818cf8")
C_GREEN     = HexColor("#22c55e")
C_AMBER     = HexColor("#f59e0b")
C_RED       = HexColor("#ef4444")
C_WHITE     = HexColor("#f8fafc")
C_GRAY_300  = HexColor("#cbd5e1")
C_GRAY_400  = HexColor("#94a3b8")
C_GRAY_500  = HexColor("#64748b")
C_CODE_BG   = HexColor("#0f172a")
C_CODE_FG   = HexColor("#e2e8f0")

OUTPUT_PATH = Path(__file__).parent / "credit_recourse_engine_report.pdf"

# ── Styles ─────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def make_styles():
    s = {}
    s["title"] = ParagraphStyle("title",
        fontSize=28, textColor=C_WHITE, spaceAfter=6, spaceBefore=4,
        fontName="Helvetica-Bold", alignment=TA_LEFT)
    s["subtitle"] = ParagraphStyle("subtitle",
        fontSize=13, textColor=C_GRAY_400, spaceAfter=20,
        fontName="Helvetica", alignment=TA_LEFT)
    s["h1"] = ParagraphStyle("h1",
        fontSize=18, textColor=C_BRAND_LT, spaceBefore=22, spaceAfter=8,
        fontName="Helvetica-Bold", borderPad=0)
    s["h2"] = ParagraphStyle("h2",
        fontSize=13, textColor=C_WHITE, spaceBefore=16, spaceAfter=6,
        fontName="Helvetica-Bold")
    s["h3"] = ParagraphStyle("h3",
        fontSize=11, textColor=C_BRAND_LT, spaceBefore=12, spaceAfter=4,
        fontName="Helvetica-Bold")
    s["body"] = ParagraphStyle("body",
        fontSize=9.5, textColor=C_GRAY_300, spaceAfter=6, spaceBefore=2,
        fontName="Helvetica", leading=15, alignment=TA_JUSTIFY)
    s["body_left"] = ParagraphStyle("body_left",
        fontSize=9.5, textColor=C_GRAY_300, spaceAfter=6, spaceBefore=2,
        fontName="Helvetica", leading=15, alignment=TA_LEFT)
    s["code"] = ParagraphStyle("code",
        fontSize=8, textColor=C_CODE_FG, spaceAfter=4, spaceBefore=4,
        fontName="Courier", leading=12, leftIndent=8, rightIndent=8,
        backColor=C_CODE_BG, borderPad=6, alignment=TA_LEFT)
    s["bullet"] = ParagraphStyle("bullet",
        fontSize=9.5, textColor=C_GRAY_300, spaceAfter=3, spaceBefore=1,
        fontName="Helvetica", leading=14, leftIndent=14, firstLineIndent=-10)
    s["qa_q"] = ParagraphStyle("qa_q",
        fontSize=10, textColor=C_WHITE, spaceAfter=4, spaceBefore=10,
        fontName="Helvetica-Bold", leftIndent=0)
    s["qa_a"] = ParagraphStyle("qa_a",
        fontSize=9.5, textColor=C_GRAY_300, spaceAfter=6, spaceBefore=2,
        fontName="Helvetica", leading=14, leftIndent=12, alignment=TA_JUSTIFY)
    s["caption"] = ParagraphStyle("caption",
        fontSize=8, textColor=C_GRAY_500, spaceAfter=4, spaceBefore=2,
        fontName="Helvetica-Oblique", alignment=TA_CENTER)
    s["toc"] = ParagraphStyle("toc",
        fontSize=10, textColor=C_GRAY_300, spaceAfter=3, leftIndent=0,
        fontName="Helvetica")
    s["toc2"] = ParagraphStyle("toc2",
        fontSize=9, textColor=C_GRAY_400, spaceAfter=2, leftIndent=16,
        fontName="Helvetica")
    return s

ST = make_styles()

def P(text, style="body"):         return Paragraph(text, ST[style])
def H1(text):                      return Paragraph(text, ST["h1"])
def H2(text):                      return Paragraph(text, ST["h2"])
def H3(text):                      return Paragraph(text, ST["h3"])
def B(text):                       return Paragraph(f"• {text}", ST["bullet"])
def Code(text):                    return Paragraph(text.replace("\n","<br/>").replace(" ","&nbsp;"), ST["code"])
def SP(n=6):                       return Spacer(1, n)
def HR():                          return HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=8, spaceBefore=8)

def info_box(text, color=C_BRAND):
    """Coloured callout box."""
    data = [[Paragraph(text, ST["body_left"])]]
    t = Table(data, colWidths=["100%"])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), HexColor("#1e293b")),
        ("LEFTPADDING",  (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
        ("TOPPADDING",   (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0), (-1,-1), 8),
        ("LINECOLOR",    (0,0), (-1,-1), color),
        ("BOX",  (0,0), (-1,-1), 1.5, color),
    ]))
    return t

def section_table(rows, col_widths=None):
    """Generic two-column info table."""
    t = Table(rows, colWidths=col_widths or [5*cm, 11*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (0,-1), HexColor("#1e293b")),
        ("BACKGROUND",   (1,0), (1,-1), HexColor("#0f172a")),
        ("TEXTCOLOR",    (0,0), (0,-1), C_BRAND_LT),
        ("TEXTCOLOR",    (1,0), (1,-1), C_GRAY_300),
        ("FONTNAME",     (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",     (1,0), (1,-1), "Helvetica"),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("GRID",         (0,0), (-1,-1), 0.5, C_BORDER),
    ]))
    return t

# ── Page template ──────────────────────────────────────────────────────────
def make_doc(path):
    doc = BaseDocTemplate(
        str(path), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm,
        title="Credit Recourse Engine — Technical Report",
        author="Credit Recourse Engine",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="main")

    def header_footer(canvas, doc):
        canvas.saveState()
        # Dark header strip
        canvas.setFillColor(C_SURFACE)
        canvas.rect(0, A4[1]-1.8*cm, A4[0], 1.8*cm, fill=1, stroke=0)
        canvas.setFillColor(C_BRAND)
        canvas.rect(0, A4[1]-1.8*cm, 4*mm, 1.8*cm, fill=1, stroke=0)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(C_WHITE)
        canvas.drawString(1.5*cm, A4[1]-1.1*cm, "Credit Recourse Engine — Technical Report")
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(C_GRAY_400)
        canvas.drawRightString(A4[0]-2*cm, A4[1]-1.1*cm, f"Page {doc.page}")
        # Footer
        canvas.setFillColor(C_BORDER)
        canvas.rect(0, 1.2*cm, A4[0], 0.4*mm, fill=1, stroke=0)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(C_GRAY_500)
        canvas.drawString(2*cm, 0.8*cm, "XGBoost + MAPIE + DiCE-ML + FastAPI | Home Credit Default Risk")
        canvas.restoreState()

    pt = PageTemplate(id="main", frames=[frame], onPage=header_footer)
    doc.addPageTemplates([pt])
    return doc

# ── Content builder ────────────────────────────────────────────────────────
def build_content():
    story = []

    # ── Cover page ──────────────────────────────────────────────────────────
    story += [SP(80)]
    story += [P("Credit Recourse Engine", "title")]
    story += [P("Complete Technical Reference — Theory, Code & Interview Q&A", "subtitle")]
    story += [HR()]
    story += [SP(8)]
    cover_data = [
        ["Stack",     "XGBoost · MAPIE · DiCE-ML · FastAPI · Alpine.js"],
        ["Dataset",   "Home Credit Default Risk (307,511 applicants, 6 tables)"],
        ["Pipeline",  "4-Layer: Classifier → Conformal → Counterfactual → Ranker"],
        ["Model AUC", "~0.79 (13/60 Optuna trials, tunable to 0.82+)"],
        ["Purpose",   "Actionable credit denial recourse with effort-ranked paths"],
    ]
    story += [section_table([[P(k, "h3"), P(v, "body_left")] for k, v in cover_data], [4*cm, 12*cm])]
    story += [PageBreak()]

    # ── Table of Contents ───────────────────────────────────────────────────
    story += [H1("Table of Contents")]
    toc = [
        ("1",  "Project Overview & Motivation"),
        ("2",  "Architecture: The 4-Layer Pipeline"),
        ("3",  "Dataset & Feature Engineering"),
        ("4",  "Layer 1 — XGBoost Classifier + Optuna"),
        ("5",  "Layer 2 — Conformal Prediction & Grey Zone (MAPIE)"),
        ("6",  "Layer 3 — Counterfactual Generation (DiCE-ML)"),
        ("7",  "Layer 4 — Effort Ranker & Recourse Paths"),
        ("8",  "Feature Constraints System"),
        ("9",  "Backend API (FastAPI)"),
        ("10", "Frontend (Alpine.js + Tailwind CSS)"),
        ("11", "Training Pipeline: End-to-End Walkthrough"),
        ("12", "Testing Strategy"),
        ("13", "Deployment (HuggingFace Spaces + Docker)"),
        ("14", "How All Components Talk to Each Other"),
        ("15", "Interview Panel Q&A (50 Questions)"),
    ]
    for num, title in toc:
        story += [P(f"<b>{num}.</b>&nbsp;&nbsp;{title}", "toc")]
    story += [PageBreak()]

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 1 — PROJECT OVERVIEW
    # ═══════════════════════════════════════════════════════════════════════
    story += [H1("1. Project Overview & Motivation")]
    story += [P("""
The Credit Recourse Engine is a full-stack machine-learning system that does two things most credit-scoring systems do not:
it not only predicts whether a loan application will default, but also generates <b>actionable, effort-ranked pathways</b>
that tell a denied applicant <i>exactly what to change</i> to get approved.
""")]
    story += [P("""
Traditional credit models output a single probability and a binary decision. A loan officer or applicant receives
"Denied — P(default)=0.74" with no explanation and no path forward. This project replaces that with:
<br/><br/>
<b>Deny → "If you increase your external credit score by 0.15 and reduce your loan amount by 20%, there is a 74% chance
of approval in roughly 4–6 months."</b>
""")]
    story += [H2("1.1 Why This Matters")]
    story += [P("""
Credit denial without explanation is a well-documented harm in consumer finance. Regulations like the EU AI Act and
the US Equal Credit Opportunity Act require that denied applicants receive actionable reasons for denial. This project
operationalises that requirement algorithmically: every denial comes with a ranked menu of concrete, feasible changes.
""")]
    story += [H2("1.2 Four-Layer Design Philosophy")]
    for row in [
        ("Layer 1: Classify",    "XGBoost answers: what is P(default) for this applicant?"),
        ("Layer 2: Calibrate",   "MAPIE conformal prediction answers: how confident are we? Flag borderline cases."),
        ("Layer 3: Explain",     "DiCE-ML counterfactuals answer: what minimal change flips the decision?"),
        ("Layer 4: Rank",        "Effort ranker answers: which path is easiest/fastest for this specific person?"),
    ]:
        story += [KeepTogether([
            P(f"<b>{row[0]}</b>", "body_left"),
            P(row[1], "body"),
            SP(3),
        ])]
    story += [PageBreak()]

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 2 — ARCHITECTURE
    # ═══════════════════════════════════════════════════════════════════════
    story += [H1("2. Architecture: The 4-Layer Pipeline")]
    story += [P("""
The system is a sequential pipeline where each layer builds on the output of the previous one.
Data flows from raw applicant input → engineered features → prediction → calibration → counterfactuals → ranked paths → API response.
""")]

    arch_data = [
        [P("Component", "h3"),       P("File", "h3"),                       P("Responsibility", "h3")],
        [P("Feature Eng.", "body"),  P("ml/feature_engineering.py", "code"), P("Merge 6 tables, derive 300+ features", "body")],
        [P("Classifier",  "body"),   P("ml/classifier.py", "code"),          P("XGBoost + Optuna HPO, AUC ~0.79", "body")],
        [P("Grey Zone",   "body"),   P("ml/grey_zone.py", "code"),           P("MAPIE LAC conformal calibration", "body")],
        [P("Counterfactuals","body"),P("ml/counterfactuals.py", "code"),     P("DiCE genetic, 15 recourse features", "body")],
        [P("Constraints", "body"),   P("ml/constraints.py", "code"),         P("Immutable/bounded/directional rules", "body")],
        [P("Effort Ranker","body"),  P("ml/effort_ranker.py", "code"),       P("Score = flip_prob×0.5 + feasibility×0.5", "body")],
        [P("API",         "body"),   P("api/main.py + routes/", "code"),     P("FastAPI: /predict, /recourse, /health", "body")],
        [P("Frontend",    "body"),   P("frontend/templates/index.html", "code"), P("Alpine.js + Tailwind CSS, CDN-only", "body")],
        [P("Train Script","body"),   P("scripts/train.py", "code"),          P("6-stage end-to-end training pipeline", "body")],
    ]
    t = Table(arch_data, colWidths=[3.5*cm, 5*cm, 7.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), C_BRAND),
        ("TEXTCOLOR",    (0,0), (-1,0), C_WHITE),
        ("BACKGROUND",   (0,1), (-1,-1), C_CODE_BG),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [HexColor("#0f172a"), HexColor("#1e293b")]),
        ("GRID",         (0,0), (-1,-1), 0.4, C_BORDER),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("LEFTPADDING",  (0,0), (-1,-1), 7),
    ]))
    story += [t, SP(10)]

    story += [H2("2.1 Data Flow: Request to Response")]
    story += [Code("""
Browser POST /api/predict  →  _build_feature_row()
                               ↓
                           XGBClassifier.predict_proba()    ← Layer 1
                               ↓
                           GreyZonePredictor.predict_single()  ← Layer 2
                               ↓
                           return {zone, p_default, shap_features}

Browser POST /api/recourse →  _build_feature_row()
                               ↓
                           CounterfactualGenerator.generate()  ← Layer 3
                               ↓
                           EffortRanker.rank()                 ← Layer 4
                               ↓
                           return {paths: [{rank, steps, flip_prob, time}]}
""")]
    story += [H2("2.2 Artifact Files")]
    story += [P("All trained objects are serialised with <b>joblib</b> to <code>backend/artifacts/</code>:")]
    for row in [
        ("xgb_model.pkl",      "Fitted XGBClassifier + metadata (AUC, feature_names)"),
        ("mapie_clf.pkl",      "Fitted MapieClassifier + raw estimator reference"),
        ("dice_data.pkl",      "CounterfactualGenerator (DiCE data + model + explainer)"),
        ("feature_names.pkl",  "Ordered list of feature columns (must match model input)"),
        ("feature_stats.pkl",  "mean/std/median/min/max per feature (for imputation + effort)"),
        ("label_encoders.pkl", "LabelEncoder per categorical column"),
        ("training_sample.pkl","2,000-row sample used for DiCE distribution estimation"),
    ]:
        story += [B(f"<b>{row[0]}</b> — {row[1]}")]
    story += [PageBreak()]

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 3 — DATASET & FEATURE ENGINEERING
    # ═══════════════════════════════════════════════════════════════════════
    story += [H1("3. Dataset & Feature Engineering")]
    story += [H2("3.1 The Home Credit Default Risk Dataset")]
    story += [P("""
Home Credit is a consumer finance company that serves people with little or no credit history.
The dataset (publicly available on Kaggle) contains 307,511 loan applications with a binary TARGET label:
0 = repaid, 1 = defaulted. The base positive rate (default rate) is approximately 8.07%, making this a
heavily imbalanced classification problem.
""")]
    story += [H2("3.2 The Six Tables")]
    table_data = [
        [P("Table", "h3"),              P("Rows", "h3"),  P("What It Contains", "h3"),          P("Key Features Generated", "h3")],
        [P("application_train.csv","body"), P("307k","body"), P("Core application data — income, loan amount, age, employment, external credit scores, region", "body"), P("Base 120 features + derived ratios", "body")],
        [P("bureau.csv", "body"),        P("1.7M","body"), P("Credit Bureau records — all previous credits at other institutions","body"), P("BUREAU_ACTIVE_COUNT, BUREAU_DAYS_CREDIT_MIN, BUREAU_AMT_CREDIT_SUM_DEBT_SUM","body")],
        [P("bureau_balance.csv","body"), P("27M","body"),  P("Monthly balance snapshots of bureau credits","body"), P("Aggregated into bureau.csv first, then joined","body")],
        [P("previous_application.csv","body"),P("1.7M","body"),P("All prior Home Credit applications (approved/refused/cancelled)","body"),P("PREV_APP_COUNT, approved/refused counts","body")],
        [P("installments_payments.csv","body"),P("13.6M","body"),P("Repayment history for previously disbursed credits","body"),P("INSTAL_INST_DPD_MEAN, INSTAL_LATE_RATE, payment ratio","body")],
        [P("credit_card_balance.csv","body"),P("3.8M","body"),P("Monthly credit card balance snapshots","body"),P("CC_CC_LIMIT_USE_RATIO_MEAN, CC_CC_PAYMENT_RATIO_MEAN","body")],
    ]
    t = Table(table_data, colWidths=[4*cm, 1.5*cm, 6.5*cm, 4*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), C_BRAND),
        ("TEXTCOLOR",    (0,0), (-1,0), C_WHITE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[HexColor("#0f172a"), HexColor("#1e293b")]),
        ("GRID",         (0,0), (-1,-1), 0.4, C_BORDER),
        ("FONTSIZE",     (0,0), (-1,-1), 8.5),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
    ]))
    story += [t, SP(10)]

    story += [H2("3.3 Feature Engineering Code Walkthrough")]
    story += [P("""
The entry point is <b>build_features(DATA_DIR, split='train', use_supplementary=True)</b>
in <code>backend/ml/feature_engineering.py</code>. It reads the CSVs, processes each supplementary table
into a per-applicant aggregation, then left-joins onto the application table.
""")]
    story += [H3("3.3.1 Processing Pattern (Same for All 5 Supplementary Tables)")]
    story += [Code("""
def process_bureau(bureau_df: pd.DataFrame) -> pd.DataFrame:
    # Step 1: Group by SK_ID_CURR — one row per applicant
    df = pd.DataFrame(index=pd.Index(bureau_df["SK_ID_CURR"].unique(),
                                     name="SK_ID_CURR"))
    g = bureau_df.groupby("SK_ID_CURR")

    # Step 2: Simple aggregations
    df["BUREAU_LOAN_COUNT"]  = g["SK_ID_BUREAU"].count()
    df["BUREAU_ACTIVE_COUNT"] = (bureau_df["CREDIT_ACTIVE"] == "Active") \\
                                  .groupby(bureau_df["SK_ID_CURR"]).sum()

    # Step 3: Numeric aggregations
    df["BUREAU_AMT_CREDIT_SUM_DEBT_MEAN"] = g["AMT_CREDIT_SUM_DEBT"].mean()
    df["BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM"] = g["AMT_CREDIT_SUM_OVERDUE"].sum()

    return df   # index = SK_ID_CURR, columns = engineered features
""")]
    story += [P("""
<b>Why <code>pd.Index(..., name="SK_ID_CURR")</code>?</b> When you do <code>main_df.join(bureau_df)</code>,
pandas merges on matching index names. Without naming the index, pandas raises a KeyError because
it cannot find "SK_ID_CURR" as a column in the aggregated DataFrame.
""")]
    story += [H3("3.3.2 Application-Level Derived Features")]
    story += [P("The <b>engineer_application_features()</b> function adds ratio-based features that are known strong predictors:")]
    story += [Code("""
df["CREDIT_INCOME_RATIO"]   = df["AMT_CREDIT"]  / (df["AMT_INCOME_TOTAL"] + 1)
df["ANNUITY_INCOME_RATIO"]  = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
df["CREDIT_TERM"]           = df["AMT_CREDIT"]  / (df["AMT_ANNUITY"] + 1)
df["EXT_SOURCE_MEAN"]       = df[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]].mean(axis=1)
df["EXT_SOURCE_PRODUCT"]    = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
df["DAYS_EMPLOYED_RATIO"]   = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1e-6)
df["AGE_YEARS"]             = -df["DAYS_BIRTH"]  / 365.25
df["INCOME_PER_PERSON"]     = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].clip(lower=1)
""")]
    story += [info_box(
        "<b>Why +1 in denominators?</b> Avoids division-by-zero. AMT_INCOME_TOTAL could theoretically be 0. "
        "The +1 causes a tiny distortion (off by &lt;0.001%) but eliminates inf values throughout the feature matrix.",
        C_AMBER)]
    story += [SP(8)]

    story += [H3("3.3.3 Categorical Encoding")]
    story += [P("""
After merging all tables, categorical columns (CODE_GENDER, NAME_EDUCATION_TYPE, etc.) are label-encoded
using sklearn's <b>LabelEncoder</b>. Each encoder is fitted on the training split and saved so that
inference uses identical mappings.
""")]
    story += [Code("""
def encode_categoricals(X, encoders=None):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    if encoders is None:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    else:
        for col in cat_cols:
            if col in encoders:
                X[col] = encoders[col].transform(X[col].astype(str))
    return X, encoders
""")]
    story += [P("""
<b>Important design decision:</b> After label-encoding, all categoricals become integers (0, 1, 2…).
This is intentional — it means we can pass the entire feature matrix to both XGBoost and DiCE
without any special handling. DiCE's "categorical" mode expects raw string categories; using
label-encoded integers as continuous features in DiCE avoids type-mismatch errors at inference time.
""")]
    story += [PageBreak()]

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 4 — XGBOOST
    # ═══════════════════════════════════════════════════════════════════════
    story += [H1("4. Layer 1 — XGBoost Classifier + Optuna")]
    story += [H2("4.1 Gradient Boosting Theory")]
    story += [P("""
<b>Gradient boosting</b> builds an ensemble of weak learners (shallow decision trees) sequentially.
Each new tree is trained to correct the residual errors of the previous ensemble.
""")]
    story += [P("""
Formally, at step <i>m</i>, we have an ensemble <i>F_m(x)</i> and we fit a new tree <i>h_m(x)</i>
to the negative gradient of the loss function:
""")]
    story += [Code("""
F_m(x) = F_{m-1}(x) + η · h_m(x)
where h_m(x) = argmin_h Σ L(y_i, F_{m-1}(x_i) + h(x_i))
""")]
    story += [P("""
For binary cross-entropy loss (used here), the negative gradient at each point is the residual
<i>y_i - p_i</i>, where <i>p_i</i> is the current probability estimate.
Each tree learns to push predictions toward the correct class.
""")]
    story += [H2("4.2 What XGBoost Adds to Standard Gradient Boosting")]
    for item in [
        "<b>Second-order Taylor approximation</b> — uses both gradient and Hessian of the loss, making tree splits more accurate than first-order methods (GBM, sklearn's GradientBoostingClassifier).",
        "<b>Regularisation terms</b> — adds L1 (alpha) and L2 (lambda) penalties on leaf weights, preventing overfitting without reducing tree count.",
        "<b>Column subsampling</b> — each tree sees only a random subset of features (colsample_bytree), reducing correlation between trees and variance.",
        "<b>tree_method='hist'</b> — histogram-based approximate split finding: bins continuous values into buckets, dramatically faster than exact split finding on large datasets (307k rows here).",
        "<b>scale_pos_weight</b> — adjusts class weights for imbalanced datasets. With 8% default rate: neg/pos = 92/8 ≈ 11.5. Tells XGBoost to up-weight positive (default) examples.",
    ]:
        story += [B(item)]
    story += [SP(6)]

    story += [H2("4.3 Optuna Hyperparameter Search")]
    story += [P("""
<b>Optuna</b> is a define-by-run Bayesian optimisation framework. Instead of exhaustively searching a grid,
it uses <b>Tree-structured Parzen Estimator (TPE)</b> to model which hyperparameter regions are promising
and samples from those regions preferentially.
""")]
    story += [P("The search space in <code>classifier.py</code>:")]
    story += [Code("""
def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 800),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":         trial.suggest_float("reg_alpha",  1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
    }
    clf = XGBClassifier(**params, scale_pos_weight=neg/pos,
                        tree_method="hist", eval_metric="auc",
                        early_stopping_rounds=50)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return roc_auc_score(y_val, clf.predict_proba(X_val)[:,1])
""")]
    story += [P("""
<b>Early stopping</b> (50 rounds) means each trial fits until validation AUC stops improving for 50 consecutive trees.
This means the n_estimators parameter is an upper bound — the actual number of trees used is determined by when
performance plateaues. This prevents wasting time on trials that are clearly not improving.
""")]
    story += [H2("4.4 CreditClassifier Class Design")]
    story += [Code("""
class CreditClassifier:
    def train(self, X_train, y_train, X_val, y_val, n_trials=60, timeout=3600):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: self._objective(t, X_train, y_train, X_val, y_val),
                       n_trials=n_trials, timeout=timeout)
        # Refit on best params
        best = study.best_params
        self.model = XGBClassifier(**best, ...)
        self.model.fit(X_train, y_train, ...)
        self.val_auc = roc_auc_score(y_val, self.model.predict_proba(X_val)[:,1])
        return self.val_auc

    def save(self, path):
        joblib.dump({"model": self.model, "val_auc": self.val_auc,
                     "feature_names": self.feature_names}, path)
""")]
    story += [H2("4.5 SHAP Feature Importance")]
    story += [P("""
<b>SHAP (SHapley Additive exPlanations)</b> values explain individual predictions by distributing
the prediction output among features according to cooperative game theory (Shapley values from economics).
""")]
    story += [P("""
For a model output <i>f(x)</i>, SHAP assigns each feature a value φ_i such that:
f(x) = E[f(X)] + Σ φ_i, where E[f(X)] is the base rate (mean prediction).
A positive φ_i means feature i pushed the prediction <i>higher</i> (more likely to default).
A negative φ_i means it pushed the prediction <i>lower</i> (less likely to default).
""")]
    story += [Code("""
# In predict.py
explainer = shap.TreeExplainer(clf.model)
shap_vals = explainer.shap_values(X)   # shape: (1, n_features)
# shap_values are already in log-odds space for XGBoost
# Positive = increases P(default), Negative = decreases P(default)
""")]
    story += [info_box(
        "<b>SHAP vs. Feature Importance:</b> Standard XGBoost feature importance (gain, split count) is global "
        "— it tells you which features matter on average across all predictions. SHAP values are local — they "
        "explain this specific prediction for this specific applicant. Two applicants can have opposite SHAP "
        "values for the same feature if it affects them differently.",
        C_GREEN)]
    story += [PageBreak()]

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 5 — CONFORMAL PREDICTION / MAPIE
    # ═══════════════════════════════════════════════════════════════════════
    story += [H1("5. Layer 2 — Conformal Prediction & Grey Zone (MAPIE)")]
    story += [H2("5.1 The Problem: Why Probability Alone Is Not Enough")]
    story += [P("""
XGBoost outputs P(default) = 0.42. But how confident is that estimate? With 8% positive rate and
label noise, probabilities near 0.5 carry enormous uncertainty. Approving everything below 0.5 and
denying everything above leads to a 40% grey zone of cases where the model genuinely does not know.
Conformal prediction formalises this uncertainty with rigorous statistical guarantees.
""")]
    story += [H2("5.2 Conformal Prediction Theory")]
    story += [P("""
<b>Conformal prediction</b> is a framework for constructing prediction sets with guaranteed coverage.
Instead of a single predicted class, it outputs a set {0}, {1}, or {0, 1}, where the set contains
the true label with probability ≥ 1 - α (e.g., 90%).
""")]
    story += [P("The key concept is the <b>nonconformity score (conformal score)</b>. For a calibration example (x_i, y_i):")]
    story += [Code("""
s_i = 1 - f(x_i)[y_i]
# i.e., 1 minus the model's predicted probability of the TRUE class
# High score = model was wrong/uncertain | Low score = model was confident and correct
""")]
    story += [P("""
On a held-out calibration set, we compute all scores {s_1, …, s_n} and find the (1-α) quantile:
""")]
    story += [Code("""
q_hat = quantile({s_1,...,s_n}, level=(1-α)(1 + 1/n))
# e.g., at α=0.10, we find the 90th percentile of calibration scores
""")]
    story += [P("""
At test time, a label y is included in the prediction set if: 1 - f(x)[y] ≤ q_hat.
If <b>both</b> labels satisfy this, the applicant is in the grey zone (uncertain).
If <b>neither</b> does (rare), the model is extremely confident but could be wrong.
""")]
    story += [H2("5.3 LAC (Least Ambiguous Classifier-Based)")]
    story += [P("""
The <b>LAC method</b> (Sadinle et al., 2019) is the correct conformal score for binary classification.
It produces prediction <i>sets</i> (not intervals) and is the method supported by MAPIE for binary targets.
""")]
    story += [P("""
<b>Why not RAPS?</b> RAPS (Regularised Adaptive Prediction Sets) was designed for <i>multi-class</i>
classification where the prediction set is a subset of many labels. For binary classification (0 or 1),
RAPS reduces to a degenerate case and MAPIE raises a ValueError. LAC is the correct choice here.
""")]
    story += [H2("5.4 MAPIE Implementation")]
    story += [Code("""
# In grey_zone.py
from mapie.classification import MapieClassifier

class GreyZonePredictor:
    def calibrate(self, base_model, X_cal, y_cal):
        # cv="prefit" means: the model is already fitted, just use it for scoring
        self.mapie = MapieClassifier(estimator=base_model, cv="prefit", method="lac")
        self.mapie.fit(X_cal, y_cal)   # calibration only — computes q_hat
        self._estimator = base_model   # store separately (cv="prefit" bug: no .estimator_ attr)

        # Compute prediction sets at α=0.10 (90% coverage target)
        _, pred_sets = self.mapie.predict(X_cal, alpha=0.10, include_last_label=True)
        # pred_sets shape: (n, 2, 1) — [includes_class_0, includes_class_1] for each row
""")]
    story += [H2("5.5 Three-Zone Interpretation")]
    story += [Code("""
def predict_single(self, X):
    _, pred_sets = self.mapie.predict(X, alpha=self.alpha, include_last_label=True)
    incl_0 = bool(pred_sets[0, 0, 0])   # prediction set includes class 0 (approve)
    incl_1 = bool(pred_sets[0, 1, 0])   # prediction set includes class 1 (default)

    if incl_0 and not incl_1:   zone = "approve"  # confident approval
    elif incl_1 and not incl_0: zone = "deny"      # confident denial
    else:                        zone = "grey"      # {0,1} — human review needed
""")]
    story += [info_box(
        "<b>Grey Zone Rate in Practice:</b> The trained model produces ~41% grey zone, 46.5% approve, 12.6% deny. "
        "This seems high, but it reflects genuine model uncertainty at α=0.10 (tight coverage guarantee). "
        "Increasing α to 0.20 would shrink the grey zone to ~25% but reduce the statistical guarantee to 80% coverage. "
        "The grey zone rate is a direct product of model calibration and threshold choice — not a bug.",
        C_AMBER)]
    story += [PageBreak()]

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 6 — DICE-ML COUNTERFACTUALS
    # ═══════════════════════════════════════════════════════════════════════
    story += [H1("6. Layer 3 — Counterfactual Generation (DiCE-ML)")]
    story += [H2("6.1 What Is Algorithmic Recourse?")]
    story += [P("""
<b>Algorithmic recourse</b> is the right of an individual to understand and contest an automated decision.
In credit, this means: given a denial, what is the minimal change to my profile that would flip the decision?
These changes are called <b>counterfactuals</b> — they describe a hypothetical "what if" version of the applicant.
""")]
    story += [P("""
A good counterfactual satisfies four properties (Wachter et al., 2017):
<b>(1) Validity</b> — the CF is actually classified as the desired class.
<b>(2) Proximity</b> — the CF is as close as possible to the original (minimal change).
<b>(3) Sparsity</b> — as few features change as possible.
<b>(4) Feasibility</b> — the changes are actionable (you cannot become younger).
""")]
    story += [H2("6.2 DiCE-ML Overview")]
    story += [P("""
<b>DiCE (Diverse Counterfactual Explanations)</b> from Microsoft Research generates multiple diverse
counterfactuals simultaneously. Diversity ensures a user receives a menu of options, not just the
single closest point in feature space. This is critical for recourse: Path A might require income growth
(takes 12 months) while Path B requires loan reduction (immediate) — both should be presented.
""")]
    story += [H2("6.3 The Genetic Algorithm Backend")]
    story += [P("""
DiCE supports three backends: gradient-based (requires differentiable models), KD-tree (fast but less diverse),
and <b>genetic algorithm</b> (used here). The genetic algorithm:
""")]
    for item in [
        "<b>Works with any model</b> — XGBoost is not differentiable end-to-end (tree splits are discrete), so gradient methods fail. The genetic algorithm treats the model as a black box.",
        "<b>Evolves a population</b> of candidate CFs over many generations, selecting the fittest (closest to target class, closest to original, most diverse from each other).",
        "<b>Fitness function</b> balances validity (is it class 0?), proximity (how close to original?), and diversity (how different from other CFs?).",
        "<b>Applies constraints</b> during evolution — features in <code>features_to_vary</code> are the only ones modified; <code>permitted_range</code> sets hard bounds.",
    ]:
        story += [B(item)]
    story += [SP(6)]
    story += [H2("6.4 CounterfactualGenerator Code Walkthrough")]
    story += [H3("6.4.1 Setup Phase (called once during training)")]
    story += [Code("""
def setup(self, model, training_sample, continuous_features, outcome_col="TARGET"):
    # DiCE needs a data distribution estimate — uses training_sample (2k rows)
    self.dice_data = dice_ml.Data(
        dataframe=training_sample,          # must include TARGET column
        continuous_features=continuous_features,  # ALL features (after label encoding)
        outcome_name="TARGET"
    )
    self.dice_model = dice_ml.Model(model=model, backend="sklearn")
    # backend="sklearn" means DiCE will call model.predict_proba(X) — XGBoost supports this
    self.explainer = dice_ml.Dice(self.dice_data, self.dice_model, method="genetic")
""")]
    story += [H3("6.4.2 Generation Phase (called per applicant)")]
    story += [Code("""
def generate(self, instance, num_cfs=6, desired_class=0):
    # Step 1: Determine which features can vary (recourse-eligible only)
    mutable_features = get_recourse_features(self.all_features)  # 15 features

    # Step 2: Build permitted_range with directional constraints
    permitted_range = build_permitted_range(instance_dict, self.feature_min, self.feature_max)
    # e.g., EXT_SOURCE_1: [current_value, 1.0]  (can only increase)
    # e.g., INSTAL_LATE_RATE: [0.0, current_value]  (can only decrease)

    # Step 3: Clean instance data
    instance = instance.fillna(0.0)
    instance = self._snap_categoricals(instance)

    # Step 4: Run DiCE genetic algorithm
    dice_exp = self.explainer.generate_counterfactuals(
        query_instances=instance,
        total_CFs=num_cfs,
        desired_class=0,           # flip to Approved (class 0 = no default)
        features_to_vary=mutable_features,
        permitted_range=permitted_range,
    )
    return self._parse_dice_output(dice_exp, instance, mutable_features, RECOURSE_DIRECTION)
""")]
    story += [H3("6.4.3 Why features_to_vary Matters")]
    story += [P("""
Without restriction, DiCE would vary all 300+ features, producing paths like "change your region code
from 2 to 1" or "change your DAYS_ID_PUBLISH by 50 days" — mathematically valid but practically meaningless.
By restricting to 15 interpretable features (EXT_SOURCE scores, loan parameters, employment duration,
credit utilization, installment history), paths are limited to levers a human can actually pull.
""")]
    story += [H3("6.4.4 The Snap Categoricals Problem")]
    story += [P("""
After label encoding and median imputation, some binary features (e.g., FLAG_OWN_REALTY = 0 or 1)
get a median value of 0.5. DiCE's internal data table only contains {0.0, 1.0} for this feature,
so it rejects 0.5 as "a category that does not exist in the dataset."
""")]
    story += [Code("""
def _snap_categoricals(self, instance):
    for col in instance.columns:
        if col in continuous_features: continue
        allowed_vals = dice_df[col].dropna().unique()
        val = float(instance[col].iloc[0])
        if val not in allowed_vals:
            nearest = min(allowed_vals, key=lambda x: abs(x - val))
            instance.at[instance.index[0], col] = nearest
    return instance
# 0.5 → snapped to nearest allowed value (0.0 or 1.0)
""")]
    story += [H3("6.4.5 Direction Post-Filter")]
    story += [P("""
DiCE's genetic algorithm doesn't strictly enforce <code>permitted_range</code> — the genetic search
occasionally produces solutions slightly outside bounds. Additionally, the algorithm is unconstrained
in direction by default, so it might "improve" an applicant's profile by reducing their credit score
(if the model has a counterintuitive feature correlation).
""")]
    story += [P("""
The direction post-filter in <code>_parse_dice_output</code> discards any change that moves
a constrained feature in the wrong direction:
""")]
    story += [Code("""
RECOURSE_DIRECTION = {
    "EXT_SOURCE_1": "increase",   # credit scores should only go up
    "EXT_SOURCE_2": "increase",
    "EXT_SOURCE_3": "increase",
    "AMT_CREDIT":   "decrease",   # smaller loan = easier to approve
    "INSTAL_INST_DPD_MEAN": "decrease",  # fewer late payments = better
    "INSTAL_LATE_RATE":     "decrease",
    "CC_CC_LIMIT_USE_RATIO_MEAN": "decrease",
    "AMT_INCOME_TOTAL":     "increase",
    ...
}

# In _parse_dice_output:
if feat in directions:
    if directions[feat] == "increase" and delta < 0: continue  # skip
    if directions[feat] == "decrease" and delta > 0: continue  # skip
""")]
    story += [PageBreak()]

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 7 — EFFORT RANKER
    # ═══════════════════════════════════════════════════════════════════════
    story += [H1("7. Layer 4 — Effort Ranker & Recourse Paths")]
    story += [H2("7.1 The Ranking Problem")]
    story += [P("""
DiCE may return 4-6 valid counterfactual paths. All of them are valid (they all flip the prediction
to Approved). But they are not equally good for the applicant. Path A might require a 12-month wait to
build credit history. Path B might require only reducing the loan amount — achievable immediately.
The effort ranker converts raw paths into a ranked menu ordered by overall desirability.
""")]
    story += [H2("7.2 Effort Score Computation")]
    story += [P("The effort score for a single feature change is:")]
    story += [Code("""
effort_i = (|Δfeature_i| / σ_i) × time_weight_i

Where:
  |Δfeature_i| = absolute change required in feature i
  σ_i          = training-set standard deviation of feature i
                 (normalises to comparable "std-dev units")
  time_weight_i = domain estimate of months to shift feature i by 1σ

Total effort = Σ effort_i  (summed across all changed features in the path)
""")]
    story += [P("The time weights (from CONSTRAINTS.PY) encode domain knowledge:")]
    story += [Code("""
TIME_WEIGHTS = {
    "AMT_CREDIT":       0.5,   # can request a smaller loan immediately
    "AMT_ANNUITY":      0.5,   # immediately adjustable
    "AMT_INCOME_TOTAL": 9.0,   # income growth takes 6-12 months
    "DAYS_EMPLOYED":    4.0,   # tenure grows at 1 month/month — 4 std devs = 4 months wait
    "EXT_SOURCE_1":     5.0,   # credit score improvement: ~4-6 months of good behaviour
    "EXT_SOURCE_2":     4.0,
    "EXT_SOURCE_3":     4.0,
    "CC_CC_LIMIT_USE_RATIO_MEAN": 3.0,  # reduce utilization: 2-3 months
    "INSTAL_INST_DPD_MEAN":       3.0,  # on-time payments: 2-4 months
    ...
}
""")]
    story += [H2("7.3 Composite Score")]
    story += [Code("""
flip_probability  = 1 - P(default | counterfactual_row)
feasibility_score = 1 / (1 + total_effort)

composite_score   = flip_weight * flip_probability
                  + cost_weight * feasibility_score
                  # default weights: 0.5 + 0.5

# flip_probability ∈ [0,1]: higher = more likely to be approved after change
# feasibility_score ∈ (0,1]: higher = less effort required (1.0 = zero effort)
# composite ∈ (0,1]: higher is better overall
""")]
    story += [P("""
<b>Why not just rank by flip probability?</b> A path that gives 95% flip chance by requiring 18 months
of income growth is worse than a path giving 70% flip chance immediately. The composite score trades off
these two dimensions. The 0.5/0.5 weights are equal but configurable.
""")]
    story += [H2("7.4 Human-Readable Step Generation")]
    story += [P("""
The <b>_generate_steps()</b> method converts raw change dictionaries into natural language.
Key formatting decisions:
""")]
    for item in [
        "<b>Large values (>1000):</b> formatted with commas as monetary amounts, shown with percentage change.",
        "<b>Ratio/score values (≤1):</b> formatted with 3 decimal places, shown as absolute delta (Δ) rather than percentage — because '0.25 → 0.50 (+100%)' is misleading (doubling a credit score isn't twice as hard as it sounds).",
        "<b>DAYS_EMPLOYED special case:</b> DAYS_EMPLOYED is stored as a negative number (days before application date). Displayed as 'Increase employment tenure from 20 mo to 26 mo (Δ 6 mo)' for human clarity.",
        "<b>Near-zero origin guard:</b> When original value ≈ 0, percentage change would be astronomical (+100 billion%). Skip percentage display; show absolute values only.",
    ]:
        story += [B(item)]
    story += [PageBreak()]

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 8 — CONSTRAINTS SYSTEM
    # ═══════════════════════════════════════════════════════════════════════
    story += [H1("8. Feature Constraints System")]
    story += [P("""
The constraints module (<code>backend/ml/constraints.py</code>) is the most domain-knowledge-heavy
component. It encodes the rules that make counterfactuals <i>feasible</i> rather than just mathematically valid.
""")]
    story += [H2("8.1 Three Constraint Levels")]
    story += [section_table([
        [P("IMMUTABLE_FEATURES", "h3"),   P("Features the applicant cannot change: age (DAYS_BIRTH), gender, family size, anomaly flags, regional credit rating, historical bureau counts. DiCE never touches these.", "body")],
        [P("FEATURE_BOUNDS", "h3"),        P("Relative bounds: (min_factor, max_factor) applied to the current value. E.g., AMT_CREDIT: (0.70, 1.30) means the loan can change ±30% from current. AMT_INCOME_TOTAL: (1.00, 2.00) means income can only increase (up to 2×).", "body")],
        [P("FEATURE_RANGE_BOUNDS", "h3"),  P("Absolute bounds regardless of current value. EXT_SOURCE_*: (0.0, 1.0). These features are always in [0,1] by definition. Used for DiCE's permitted_range.", "body")],
        [P("RECOURSE_DIRECTION", "h3"),    P("Directional constraints applied AFTER DiCE generation (post-filter). Ensures EXT_SOURCE only increases, INSTAL_LATE_RATE only decreases, etc. Compensates for DiCE genetic algorithm's loose constraint enforcement.", "body")],
        [P("RECOURSE_ELIGIBLE_FEATURES","h3"), P("The 15-feature whitelist passed to DiCE's features_to_vary. Restricts the genetic algorithm's search space to interpretable, actionable levers only.", "body")],
    ], [4.5*cm, 11.5*cm])]

    story += [H2("8.2 How build_permitted_range() Works")]
    story += [Code("""
def build_permitted_range(instance, feature_min, feature_max):
    permitted = {}

    # 1. Relative bounds (from FEATURE_BOUNDS)
    for feat, (lo_factor, hi_factor) in FEATURE_BOUNDS.items():
        current = instance[feat]
        lo = max(feature_min[feat], current * lo_factor)
        hi = min(feature_max[feat], current * hi_factor)
        permitted[feat] = [lo, hi]

    # 2. Absolute range bounds (from FEATURE_RANGE_BOUNDS)
    for feat, (abs_lo, abs_hi) in FEATURE_RANGE_BOUNDS.items():
        permitted[feat] = [abs_lo, abs_hi]

    # 3. Directional constraints (lock one end to current value)
    for feat, direction in RECOURSE_DIRECTION.items():
        lo, hi = permitted.get(feat, [feature_min[feat], feature_max[feat]])
        if direction == "increase":
            lo = max(lo, instance[feat])   # cannot go below current
        elif direction == "decrease":
            hi = min(hi, instance[feat])   # cannot go above current
        if lo <= hi:
            permitted[feat] = [lo, hi]

    return permitted
""")]
    story += [PageBreak()]

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 9 — FASTAPI BACKEND
    # ═══════════════════════════════════════════════════════════════════════
    story += [H1("9. Backend API (FastAPI)")]
    story += [H2("9.1 Application Lifespan & State")]
    story += [P("""
FastAPI's <b>lifespan</b> context manager loads all ML artifacts once at startup and stores them
on <code>app.state</code>. This is critical for performance — loading a 50MB XGBoost model on every
request would make the API unusable.
""")]
    story += [Code("""
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup
    app.state.classifier   = CreditClassifier.load(MODEL_PATH)
    app.state.grey_zone    = GreyZonePredictor.load(MAPIE_PATH)
    app.state.cf_generator = CounterfactualGenerator.load(DICE_DATA_PATH)
    app.state.feature_names  = joblib.load(FEATURE_NAMES_PATH)
    app.state.feature_stats  = joblib.load(FEATURE_STATS_PATH)
    app.state.label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    yield
    # Cleanup on shutdown (nothing needed here)
""")]
    story += [H2("9.2 The _build_feature_row() Function")]
    story += [P("""
This is the most complex inference function. It takes a sparse API payload (only the fields the user
filled in) and builds a complete 300+ feature row suitable for the model:
""")]
    story += [Code("""
def _build_feature_row(features_dict, feature_names, feature_stats, label_encoders):
    # Step 1: Start from training medians (handles missing fields gracefully)
    row = {f: feature_stats["median"].get(f, 0.0) for f in feature_names}

    # Step 2: Override with provided values
    for k, v in features_dict.items():
        if k in row and v is not None:
            row[k] = v

    # Step 3: Apply feature engineering (recompute derived features)
    if row.get("AMT_INCOME_TOTAL") and row.get("AMT_CREDIT"):
        row["CREDIT_INCOME_RATIO"] = row["AMT_CREDIT"] / (row["AMT_INCOME_TOTAL"] + 1)
    if all(row.get(f) for f in ["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]):
        row["EXT_SOURCE_MEAN"] = np.mean([row[f] for f in ["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]])
    ...

    # Step 4: Label-encode categoricals using saved encoders
    for col, le in label_encoders.items():
        if col in row:
            try:
                row[col] = le.transform([str(row[col])])[0]
            except ValueError:
                row[col] = le.transform([str(le.classes_[0])])[0]  # fallback

    # Step 5: Build single-row DataFrame in exact feature order
    return pd.DataFrame([row])[feature_names]
""")]
    story += [H2("9.3 API Endpoints")]
    story += [section_table([
        [P("GET /health","h3"),       P("Returns model_loaded, grey_zone_loaded, cf_generator_loaded booleans. Used by frontend status badge.", "body")],
        [P("POST /api/predict","h3"), P("Takes ApplicantFeatures JSON. Returns zone (approve/grey/deny), p_default, zone_color, shap_top_features. SHAP calculated on-the-fly with TreeExplainer.", "body")],
        [P("POST /api/recourse","h3"),P("Takes {applicant: ApplicantFeatures, num_paths: int}. Runs full Layer 3+4 pipeline. Returns ranked paths with steps, flip_probability, effort_score, time_estimate.", "body")],
        [P("GET /","h3"),             P("Serves index.html from frontend/templates/. FastAPI mounts the frontend directory as static files.", "body")],
    ], [4*cm, 12*cm])]
    story += [PageBreak()]

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 10 — FRONTEND
    # ═══════════════════════════════════════════════════════════════════════
    story += [H1("10. Frontend (Alpine.js + Tailwind CSS)")]
    story += [H2("10.1 Technology Choices")]
    story += [P("""
The frontend uses <b>Alpine.js</b> (lightweight reactive JavaScript, 15KB) and <b>Tailwind CSS</b>
(utility-first CSS framework), both loaded from CDN. This means no build step, no npm, no webpack —
the entire frontend is a single HTML file that any browser can render.
""")]
    for item in [
        "<b>Alpine.js vs React/Vue:</b> React would require a build pipeline, Node.js, npm, and separate deployment. Alpine.js adds reactivity with simple x-data, x-model, x-show directives directly in HTML attributes. For a single-page form with results display, Alpine is optimal.",
        "<b>Tailwind CDN vs build:</b> The CDN version includes all Tailwind utilities but cannot tree-shake unused ones (~3MB vs ~10KB in production). For a demo/portfolio project, the simplicity outweighs the size penalty.",
        "<b>Chart.js:</b> Used for the SHAP bar chart rendering (progress bars in the current implementation, could be upgraded to Chart.js bar chart).",
    ]:
        story += [B(item)]
    story += [SP(6)]
    story += [H2("10.2 Alpine.js Reactivity Pattern")]
    story += [Code("""
<body x-data="creditApp()" x-init="init()">
  <!-- x-data creates a reactive scope for the entire page -->
  <!-- x-init runs the init() function once on page load -->

  <!-- Two-way binding: form.AMT_INCOME_TOTAL updates when input changes -->
  <input x-model.number="form.AMT_INCOME_TOTAL" />

  <!-- Conditional display: only show loading spinner when loading=true -->
  <div x-show="loading">Loading...</div>

  <!-- Loop over result data -->
  <template x-for="path in result.paths" :key="path.rank">
    <div x-text="path.flip_probability"></div>
  </template>
""")]
    story += [H2("10.3 Async API Call Pattern")]
    story += [Code("""
async runPrediction() {
  this.loading = true;
  // Step 1: Hit predict endpoint
  const pred = await fetch("/api/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  }).then(r => r.json());

  // Step 2: Only call recourse for grey/deny (approve needs no paths)
  let recourse = {recourse_available: false, paths: []};
  if (pred.zone !== "approve") {
    this.loadingStage = "Generating recourse paths…";
    recourse = await fetch("/api/recourse", {...}).then(r => r.json());
  }

  // Step 3: Merge predict + recourse into single result object
  this.result = {...pred, ...recourse};
  this.loading = false;
}
""")]
    story += [PageBreak()]

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 11 — TRAINING PIPELINE
    # ═══════════════════════════════════════════════════════════════════════
    story += [H1("11. Training Pipeline: End-to-End Walkthrough")]
    story += [H2("11.1 Running the Pipeline")]
    story += [Code("""
# Full training (90-180 minutes on Apple M-chip):
python -m backend.scripts.train

# Resume from saved XGBoost model (runs MAPIE + DiCE only, ~10 minutes):
python -m backend.scripts.resume_training
""")]
    story += [H2("11.2 Stage-by-Stage Breakdown")]
    stages = [
        ("Stage 1: Feature Engineering", """
build_features(DATA_DIR, split="train", use_supplementary=True) reads all 6 CSVs,
processes each supplementary table, and left-joins onto application_train.csv.
Output: 307,511 rows × ~324 columns. Runtime: ~90 seconds.
        """),
        ("Stage 2: Label Encoding", """
encode_categoricals(X_raw) fits LabelEncoders on all object/category columns.
Saves encoders to label_encoders.pkl. Also saves feature_names.pkl (ordered list)
and feature_stats.pkl (mean/std/median/min/max per column).
        """),
        ("Stage 3: Train/Val/Calibration Split", """
Three-way stratified split:
  - Calibration (15%): for MAPIE conformal calibration
  - Train (72%): for XGBoost fitting
  - Validation (13%): for Optuna trial evaluation and early stopping
Stratified by TARGET to preserve 8% default rate in all splits.
        """),
        ("Stage 4: XGBoost + Optuna", """
CreditClassifier.train() runs 60 Optuna trials (or 1-hour timeout).
Each trial: fit XGBoost on train, evaluate AUC on val, report to Optuna.
Best hyperparameters are used to refit the final model.
Saves xgb_model.pkl.
        """),
        ("Stage 5: MAPIE Calibration", """
GreyZonePredictor.calibrate() wraps the XGBoost model with MapieClassifier(cv="prefit").
Fits the conformal calibration using ONLY the calibration split (15% of data).
Computes q_hat at α=0.10. Saves mapie_clf.pkl.
        """),
        ("Stage 6: DiCE Setup", """
Samples 2,000 rows from training data (with TARGET column).
CounterfactualGenerator.setup() creates DiCE data/model/explainer objects.
Saves dice_data.pkl (the full generator including the fitted DiCE explainer).
Also saves training_sample.pkl for debugging.
        """),
    ]
    for title, desc in stages:
        story += [KeepTogether([
            H3(title),
            P(desc.strip()),
            SP(4),
        ])]
    story += [PageBreak()]

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 12 — TESTING
    # ═══════════════════════════════════════════════════════════════════════
    story += [H1("12. Testing Strategy")]
    story += [P("""
The project has 48 automated tests split across three test files. Tests use pytest and do not require
trained models — they test logic in isolation using mock data.
""")]
    story += [H2("12.1 test_constraints.py (22 tests)")]
    story += [P("Covers: immutable feature list membership, FEATURE_BOUNDS values, RECOURSE_ELIGIBLE_FEATURES coverage, build_permitted_range() logic, directional constraint enforcement, get_time_weight() fallback, feature label formatting.")]
    story += [H2("12.2 test_effort_ranker.py (26 tests)")]
    story += [P("Covers: effort score formula (|Δ|/σ × time_weight), composite score with equal weights, path ranking by score, empty path handling, zero-std-dev guard (1e-6 floor), DAYS_EMPLOYED tenure formatting, percentage-overflow guard for near-zero origins.")]
    story += [H2("12.3 test_api.py (10 tests)")]
    story += [P("Covers: /health endpoint structure, /predict returns 503 without model, /recourse returns 503 without model, Pydantic validation on malformed input, zone classification boundaries.")]
    story += [Code("""
# Run all tests:
make test          # full suite with coverage
make test-fast     # no coverage overhead (faster)
pytest backend/tests/ -v   # verbose output
""")]
    story += [PageBreak()]

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 13 — DEPLOYMENT
    # ═══════════════════════════════════════════════════════════════════════
    story += [H1("13. Deployment (HuggingFace Spaces + Docker)")]
    story += [H2("13.1 HuggingFace Spaces")]
    story += [P("""
HuggingFace Spaces provides free GPU/CPU inference hosting for ML demos. Docker Spaces let you
deploy any containerised application. The key constraint: port 7860 must be exposed (Spaces proxies
traffic to this port regardless of your internal application port).
""")]
    story += [H2("13.2 Dockerfile")]
    story += [Code("""
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# HuggingFace Spaces requires port 7860
EXPOSE 7860
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
""")]
    story += [P("""
<b>Important:</b> Trained model artifacts (*.pkl) are excluded from git via .gitignore.
For HF Spaces deployment, either: (a) include pre-trained artifacts in the repo (adds ~200MB),
or (b) add a startup script that downloads them from HF Hub / S3 on first boot.
""")]
    story += [H2("13.3 requirements.txt Pinning Rationale")]
    story += [section_table([
        [P("pandas==1.5.3","code"),    P("dice-ml 0.11 requires pandas < 2.0. pandas 2.0 changed the copy-on-write semantics that DiCE depends on.", "body")],
        [P("numpy==1.24.4","code"),    P("Pinned for pandas 1.5.3 compatibility (numpy 2.0 has breaking changes for pandas 1.x).", "body")],
        [P("xgboost==2.0.3","code"),   P("Latest stable with sklearn API compatibility and hist tree method on Apple Silicon.", "body")],
        [P("scikit-learn==1.3.2","code"),P("MAPIE 0.8.3 requires scikit-learn 1.3.x for the MapieClassifier API.", "body")],
        [P("mapie==0.8.3","code"),     P("First version with stable LAC method for binary classification.", "body")],
        [P("dice-ml==0.11","code"),    P("Last version with genetic algorithm backend stable on non-differentiable models.", "body")],
    ], [4*cm, 12*cm])]
    story += [PageBreak()]

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 14 — COMPONENT INTERACTIONS
    # ═══════════════════════════════════════════════════════════════════════
    story += [H1("14. How All Components Talk to Each Other")]
    story += [H2("14.1 Training Time Interactions")]
    story += [Code("""
train.py
  ├── feature_engineering.py  →  307k × 324 DataFrame
  ├── classifier.py           →  xgb_model.pkl + feature_names.pkl + feature_stats.pkl
  ├── grey_zone.py            →  mapie_clf.pkl (wraps xgb_model)
  └── counterfactuals.py      →  dice_data.pkl (wraps xgb_model + training sample)
       └── constraints.py     →  RECOURSE_ELIGIBLE_FEATURES, permitted_range rules
""")]
    story += [H2("14.2 Inference Time Interactions (POST /api/predict)")]
    story += [Code("""
routes/predict.py
  ├── _build_feature_row()
  │     ├── feature_stats.pkl   (median imputation)
  │     ├── label_encoders.pkl  (categorical encoding)
  │     └── feature_names.pkl   (column ordering)
  │
  ├── classifier.model.predict_proba(X)    → p_default
  ├── grey_zone.predict_single(X)          → zone, zone_label, confidence
  └── shap.TreeExplainer(model).shap_values(X) → feature attributions
""")]
    story += [H2("14.3 Inference Time Interactions (POST /api/recourse)")]
    story += [Code("""
routes/recourse.py
  ├── _build_feature_row()     (same as predict)
  ├── grey_zone.predict_single(X)   → zone (skip if approve)
  │
  ├── cf_generator.generate(X)
  │     ├── constraints.get_recourse_features()  → 15-feature whitelist
  │     ├── constraints.build_permitted_range()  → per-feature [lo, hi] bounds
  │     ├── DiCE genetic algorithm               → 4-6 raw counterfactual rows
  │     └── _parse_dice_output()
  │           └── RECOURSE_DIRECTION post-filter → clean change dicts
  │
  └── EffortRanker.rank(paths, model, X)
        ├── model.predict_proba(cf_row)  → flip_probability per path
        ├── feature_stats["std"]         → normalise feature deltas
        ├── constraints.TIME_WEIGHTS     → domain effort per feature
        └── _generate_steps()            → human-readable action strings
""")]
    story += [PageBreak()]

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 15 — INTERVIEW Q&A
    # ═══════════════════════════════════════════════════════════════════════
    story += [H1("15. Interview Panel Q&A")]
    story += [P("""
The following 50 questions cover theory, implementation decisions, debugging history, and trade-offs
across all components of the project. They are organised by topic and range from entry-level to senior.
""")]
    story += [HR()]

    qa_pairs = [
        # ── XGBoost / ML Theory ──────────────────────────────────────────
        ("XGBOOST & GRADIENT BOOSTING", None),
        ("Q1: What is gradient boosting and how does XGBoost differ from sklearn's GradientBoostingClassifier?",
         "Gradient boosting builds an additive ensemble by fitting each new tree to the negative gradient (pseudo-residuals) "
         "of the loss. XGBoost differs in three key ways: (1) it uses a second-order Taylor approximation of the loss function "
         "giving more accurate split gains, (2) it adds L1/L2 regularisation directly into the tree scoring formula, and "
         "(3) its hist tree_method bins continuous values into histograms, making split finding O(n_bins × features) instead of "
         "O(n × features) — critical for 307k rows."),
        ("Q2: Why did you use scale_pos_weight and what value did you use?",
         "The dataset is 8% positive (default), so positives are underrepresented 11.5×. scale_pos_weight = "
         "count_negative / count_positive ≈ 11.5 tells XGBoost to treat each positive example as 11.5 negatives "
         "during gradient computation. Without it, the model optimises accuracy by predicting 0 for nearly everything, "
         "achieving 92% accuracy but near-zero recall on defaults — useless for risk management."),
        ("Q3: What is early stopping and why is it used with Optuna?",
         "Early stopping stops adding trees when the validation metric (AUC) hasn't improved for N consecutive rounds "
         "(N=50 here). In Optuna, this is critical because the search explores high n_estimators values (up to 800). "
         "Without early stopping, every trial would train all 800 trees even if the optimal was 200. Early stopping "
         "makes most trials terminate at 100-300 trees, fitting 3-4× more trials in the same time budget."),
        ("Q4: What is AUC-ROC and why is it the right metric here?",
         "AUC-ROC (Area Under the Receiver Operating Characteristic curve) measures a model's ability to rank positives "
         "above negatives across all possible thresholds. It's threshold-independent, making it ideal during hyperparameter "
         "search where we haven't chosen an operating threshold yet. For imbalanced datasets, AUC is more informative than "
         "accuracy because a model predicting all zeros would get 92% accuracy but 0.5 AUC."),
        ("Q5: What does Optuna's TPE algorithm do differently from random search or grid search?",
         "TPE (Tree-structured Parzen Estimator) builds a probabilistic model of which hyperparameter configurations "
         "yield good results. It models P(config | good_trial) and P(config | bad_trial) using kernel density estimation, "
         "then samples from configurations where the good/bad ratio is highest. This focuses trials on promising regions, "
         "typically finding better results than random search in the same number of trials — especially when some parameters "
         "strongly interact (e.g., learning_rate and n_estimators)."),

        # ── SHAP ─────────────────────────────────────────────────────────
        ("SHAP EXPLAINABILITY", None),
        ("Q6: What are SHAP values and how are they computed for tree models?",
         "SHAP values measure each feature's contribution to a prediction relative to the expected (base) prediction. "
         "For tree models, TreeSHAP computes exact Shapley values in O(TLD²) time (T trees, L leaves, D max depth) "
         "by traversing each tree and computing the expected prediction change when a feature is included vs. excluded "
         "across all possible feature subsets. The result is additive: f(x) = E[f] + Σ φ_i, where φ_i can be positive "
         "(increases risk) or negative (decreases risk)."),
        ("Q7: How do you interpret a negative SHAP value?",
         "A negative SHAP value for feature i means that feature i's value in this specific instance pushes the prediction "
         "BELOW the baseline (mean prediction). E.g., EXT_SOURCE_2=0.8 with SHAP=-0.15 means: compared to the average "
         "applicant, having EXT_SOURCE_2=0.8 reduces P(default) by roughly 0.15 in log-odds space. The feature makes "
         "this person look better than average."),
        ("Q8: Why is SHAP used here instead of showing feature importances?",
         "Feature importances (gain, split count) are global — they describe which features matter across ALL predictions. "
         "SHAP values are local — they explain THIS specific prediction for THIS applicant. Two people with the same "
         "DAYS_EMPLOYED value can have opposite SHAP values for it if the rest of their profiles differ. For a loan "
         "officer reviewing a specific application, local explanation is far more actionable than global importance."),

        # ── Conformal Prediction ──────────────────────────────────────────
        ("CONFORMAL PREDICTION & MAPIE", None),
        ("Q9: What is conformal prediction and what guarantee does it provide?",
         "Conformal prediction wraps any base classifier to output prediction SETS (instead of single predictions) "
         "with a formal coverage guarantee: the true label is contained in the prediction set with probability ≥ 1−α, "
         "where α is user-specified. The guarantee is distribution-free (no assumptions about data distribution) and "
         "holds for any exchangeable dataset. At α=0.10, 90% of test predictions contain the true label."),
        ("Q10: What is a nonconformity score?",
         "A nonconformity score measures how 'unusual' a calibration example is under the current model. For classification, "
         "the LAC score is s_i = 1 − f(x_i)[y_i]: one minus the model's confidence in the true class. A score near 0 "
         "means the model was very confident and correct (typical). A score near 1 means the model was confident it was "
         "the wrong class (atypical). The quantile of calibration scores becomes the threshold q_hat."),
        ("Q11: Why did you use LAC instead of RAPS?",
         "RAPS (Regularised Adaptive Prediction Sets) was designed for multi-class classification with many labels "
         "(e.g., ImageNet with 1000 classes). It regularises the prediction set size to avoid trivially including "
         "all classes. For binary classification with only 2 classes, RAPS is degenerate — MAPIE raises a ValueError "
         "('invalid method for binary target'). LAC is the correct conformal method for binary targets."),
        ("Q12: How does the grey zone arise from conformal prediction?",
         "At α=0.10, the threshold q_hat is the 90th percentile of calibration nonconformity scores. A test point "
         "enters the prediction set for label y if its nonconformity score for y is ≤ q_hat. If BOTH labels pass "
         "this test (score_0 ≤ q_hat AND score_1 ≤ q_hat), both are included: prediction set = {0,1}. This means "
         "the model cannot confidently exclude either outcome — the grey zone. It occurs when the model's probability "
         "is close to 0.5, where conformal scores for both classes are low."),
        ("Q13: Why does cv='prefit' require storing the estimator separately?",
         "When MapieClassifier is initialised with cv='prefit', it expects an already-fitted model — it only calibrates "
         "on the provided data without refitting. However, MapieClassifier's .estimator_ attribute is only populated when "
         "cv is an integer (cross-validation splits). In cv='prefit' mode, the attribute doesn't exist. The fix is to "
         "store self._estimator = base_model separately and use it directly for predict_proba() calls."),

        # ── DiCE / Counterfactuals ─────────────────────────────────────────
        ("DiCE-ML COUNTERFACTUALS", None),
        ("Q14: What is a counterfactual explanation and why is it useful for credit decisions?",
         "A counterfactual explanation answers: 'What is the closest possible world in which the decision would have been different?' "
         "For credit: 'If your income were £150k instead of £90k, you would be approved.' This is actionable — it tells "
         "the applicant exactly what to change. Compare to a feature importance explanation: 'Income is important for your "
         "decision' — informative but not actionable."),
        ("Q15: Why the genetic algorithm backend over gradient-based or KD-tree?",
         "Gradient-based DiCE requires a differentiable model (neural network). XGBoost uses discrete tree splits — "
         "not differentiable end-to-end. KD-tree search finds the nearest training example that flips the prediction, "
         "which is fast but produces less diverse counterfactuals (often very similar paths). The genetic algorithm "
         "treats the model as a black box, produces diverse paths, and handles feasibility constraints natively."),
        ("Q16: What is the NaN problem with DiCE and how did you solve it?",
         "DiCE's genetic algorithm rejects inputs containing NaN (it can't evolve from an undefined starting point). "
         "The training sample had 172k NaN values (from supplementary tables not available for some applicants). Fix: "
         "fill all NaN with 0.0 both in the training sample (setup time) and in the instance (generate time). 0.0 is a "
         "safe fallback because all continuous features are treated as DiCE-continuous — NaN becomes a valid 0 without "
         "DiCE treating it as a missing category."),
        ("Q17: Explain the 'category does not occur' DiCE error and your fix.",
         "DiCE stores feature value distributions internally. After label encoding, binary features like FLAG_OWN_REALTY "
         "have values {0, 1}. Median imputation produces FLAG_OWN_REALTY = 0.5 for applicants with missing data. DiCE "
         "rejects 0.5 as 'a category value that never occurred in training data.' Fix: _snap_categoricals() rounds "
         "each non-continuous feature's value to the nearest allowed value in the DiCE training data before passing "
         "the instance to generate_counterfactuals()."),
        ("Q18: Why does DiCE sometimes violate permitted_range constraints?",
         "DiCE's genetic algorithm uses permitted_range as soft constraints in the fitness function — it penalises "
         "out-of-range values but doesn't guarantee they never appear. The genetic crossover and mutation operators "
         "can occasionally produce offspring outside bounds if the penalty weight isn't high enough. The solution is "
         "a post-filter in _parse_dice_output() that drops any change violating the directional constraint, regardless "
         "of what DiCE computed."),
        ("Q19: Why do you restrict features_to_vary to 15 features?",
         "With 300+ mutable features, DiCE generates paths with 30-50 feature changes — most of which are label-encoded "
         "categoricals or obscure engineered features that an applicant cannot meaningfully act on. By restricting to "
         "15 interpretable levers (EXT_SOURCE scores, loan amount, employment, credit utilization, installment history), "
         "paths have 4-8 changes, all of which a loan officer can explain to the applicant in plain English."),
        ("Q20: What is the purpose of RECOURSE_DIRECTION and why can't you just rely on permitted_range?",
         "RECOURSE_DIRECTION prevents counterintuitive suggestions: 'Reduce your credit score from 0.25 to 0.20 to get "
         "approved.' While mathematically possible (the model has learned some non-monotone correlations), advising an "
         "applicant to worsen their credit score is nonsensical. permitted_range in DiCE is soft (genetic algorithm can "
         "violate it slightly), so the direction filter is a hard post-processing rule that discards any change moving "
         "a feature in the wrong direction."),

        # ── Effort Ranker ─────────────────────────────────────────────────
        ("EFFORT RANKER", None),
        ("Q21: Why normalise feature deltas by standard deviation?",
         "Features have wildly different scales: AMT_INCOME_TOTAL changes by 50,000 while EXT_SOURCE_2 changes by 0.1. "
         "Without normalisation, income changes would dominate the effort score purely due to scale. Dividing by σ converts "
         "all changes to 'standard deviation units' — a 1σ change in any feature has the same base effort of 1.0, then "
         "multiplied by time_weight to reflect actual difficulty."),
        ("Q22: How does the time_weight system work and who validated the estimates?",
         "Each feature in TIME_WEIGHTS has a domain-knowledge estimate of months to shift that feature by 1 standard "
         "deviation. AMT_CREDIT = 0.5 (can request a different loan immediately). EXT_SOURCE_* = 4-5 (credit score "
         "improvement takes months of good behaviour). AMT_INCOME_TOTAL = 9.0 (meaningful income growth takes 6-12 months). "
         "The estimates were validated by loan officer domain knowledge. They are intentionally approximate — the goal is "
         "relative ranking, not precise time prediction."),
        ("Q23: Why does the composite score use 0.5/0.5 weights?",
         "Equal weights represent a neutral position: a path that is 100% likely to work but takes 2 years scores "
         "the same as a path that is 50% likely to work but is achievable in 3 months. In practice, weights could "
         "be adjusted per loan product or applicant preference. For a portfolio project, equal weights are the simplest "
         "defensible default. The weights are configurable parameters in EffortRanker.__init__."),
        ("Q24: How do you compute flip_probability and why isn't it the same as 1 - p_default_original?",
         "flip_probability = 1 - P(default | counterfactual_row). This is computed by running the XGBoost model on the "
         "counterfactual feature row (not the original). It tells us: if the applicant actually achieved all the changes "
         "in this path, what would their approval probability be? It's not 1 - p_default because the CF row is a "
         "different point in feature space — the model sees a materially different applicant profile."),

        # ── System Design ─────────────────────────────────────────────────
        ("SYSTEM DESIGN & ARCHITECTURE", None),
        ("Q25: Why use joblib for serialisation instead of pickle?",
         "joblib.dump/load is specifically optimised for numpy arrays and large scientific Python objects. It uses "
         "memory-mapped file I/O and optional compression, making it 2-5× faster than pickle for objects containing "
         "large numpy arrays (like XGBoost's internal boosters). The API is identical to pickle but with better "
         "performance for ML artifacts."),
        ("Q26: Why is the training sample (2k rows) stored separately from the DiCE generator?",
         "DiCE internally stores the training sample for distribution estimation. The separate training_sample.pkl "
         "allows debugging — you can inspect the distribution DiCE is using, check for NaN contamination, verify "
         "feature ranges, and re-run DiCE setup experiments without re-running the full pipeline."),
        ("Q27: What would happen if feature_names.pkl ordering mismatched the model?",
         "XGBoost models are sensitive to column ordering when loaded via sklearn API. If feature_names had columns "
         "in a different order than during training, XGBoost would silently use the wrong values for the wrong features, "
         "producing garbage predictions. The fix: always rebuild the DataFrame with [feature_names] column ordering "
         "at the end of _build_feature_row(), and save feature_names.pkl immediately after fitting."),
        ("Q28: Why use FastAPI's lifespan instead of global variables?",
         "Global variables in Python modules are initialised once at import time. FastAPI's lifespan (async context "
         "manager) loads artifacts at server startup and makes them accessible via app.state, which is the recommended "
         "pattern for shared state in ASGI applications. It also allows clean shutdown logic (releasing GPU memory, "
         "closing database connections) that global variables cannot provide."),
        ("Q29: How would you scale this to handle 1000 concurrent users?",
         "The bottleneck is the DiCE genetic algorithm (~10-30 seconds per request). Solutions: (1) Queue-based architecture "
         "(Celery + Redis) — accept requests immediately, return a job_id, poll for results. (2) Cache counterfactuals "
         "for common applicant profiles using a vector similarity cache (FAISS or Redis). (3) Pre-compute counterfactuals "
         "for all applicants in batch after each training run, store in a database. (4) Deploy multiple worker processes "
         "with uvicorn workers=4 and a load balancer."),
        ("Q30: What is the purpose of the resume_training.py script?",
         "Training XGBoost with 60 Optuna trials takes 90-180 minutes. If the pipeline fails at Stage 5 (MAPIE) or "
         "Stage 6 (DiCE), you don't want to retrain from scratch. resume_training.py loads the saved XGBoost model "
         "and runs only Stages 5-6, taking ~5-10 minutes. It also exactly reproduces the same train/val/cal splits "
         "using the same random seed, ensuring the calibration data is always held-out from training."),

        # ── Data / Feature Engineering ────────────────────────────────────
        ("DATA & FEATURE ENGINEERING", None),
        ("Q31: Why name the DataFrame index SK_ID_CURR when building supplementary tables?",
         "When you do main_df.join(bureau_df, on='SK_ID_CURR'), pandas looks for a column named SK_ID_CURR "
         "in main_df and matches it against bureau_df's INDEX (not a column). If the index has no name, "
         "pandas cannot match — raises KeyError: 'SK_ID_CURR'. Setting pd.Index(..., name='SK_ID_CURR') gives "
         "the index a name, enabling the join. This was a critical bug that caused all 5 supplementary tables "
         "to fail joining."),
        ("Q32: Why replace groupby lambda functions with vectorized operations?",
         "df.groupby().apply(lambda x: ...) is Python-level iteration — for a 1.7M row bureau.csv, it spawns "
         "a Python loop with 307,511 iterations. Vectorized operations like (bureau['CREDIT_ACTIVE']=='Active')"
         ".groupby(bureau['SK_ID_CURR']).sum() stay in C-level pandas/numpy code throughout, running 10-50× faster. "
         "This reduced feature engineering from ~20 minutes to ~90 seconds."),
        ("Q33: Why do derived features use +1 in denominators?",
         "CREDIT_INCOME_RATIO = AMT_CREDIT / (AMT_INCOME_TOTAL + 1). Without the +1, any applicant with "
         "AMT_INCOME_TOTAL=0 produces inf, which propagates through all dependent features. The +1 causes a "
         "<0.001% error for high incomes and prevents the feature matrix from containing inf values that "
         "XGBoost cannot handle (it would produce NaN predictions for those rows)."),
        ("Q34: What is DAYS_BIRTH and why is it negative?",
         "DAYS_BIRTH is the number of days between the applicant's birth date and the application date, "
         "expressed as a negative number. A value of -12,000 means the applicant was born 12,000 days "
         "(≈32.8 years) before the application. Negative encoding is a convention in the Home Credit dataset — "
         "all date-relative features use this format. AGE_YEARS = -DAYS_BIRTH / 365.25 converts it to intuitive years."),

        # ── Debugging & Pitfalls ──────────────────────────────────────────
        ("DEBUGGING & COMMON PITFALLS", None),
        ("Q35: You mentioned pandas==1.5.3 is pinned. What breaks in pandas 2.0?",
         "pandas 2.0 introduced Copy-on-Write (CoW) semantics — operations that previously modified DataFrames "
         "in-place now return copies. dice-ml 0.11 was written assuming the old in-place modification behaviour: "
         "it calls df[col] = ... expecting to modify the DiCE internal dataframe, but under CoW this silently "
         "creates a copy that gets discarded. The result is silent data corruption in the counterfactual generation. "
         "pandas 1.5.3 is the last release before CoW was introduced."),
        ("Q36: What debugging steps would you take if p_default seemed wrong at inference time?",
         "Checklist: (1) Print X.shape and X.dtypes — verify feature count and no unexpected object columns. "
         "(2) Check for NaN: X.isna().sum().sum() — any NaN would corrupt XGBoost predictions. "
         "(3) Verify feature_names ordering matches training: clf.model.feature_names_in_ vs X.columns.tolist(). "
         "(4) Check label encoder mappings: le.classes_ for each categorical. "
         "(5) Recompute derived features manually and compare to training stats. "
         "(6) Compare the raw predict_proba output to the SHAP explanation for consistency."),
        ("Q37: How would you handle a new categorical value at inference time (unseen label)?",
         "LabelEncoder raises ValueError on unseen labels. The current fix: wrap in try/except and fall back to "
         "le.classes_[0] (the first seen class). A better production solution: use sklearn's OrdinalEncoder with "
         "handle_unknown='use_encoded_value' and unknown_value=-1, then let XGBoost treat -1 as a separate category. "
         "Or use target encoding / frequency encoding which generalise to unseen categories naturally."),
        ("Q38: The grey zone rate is 41% — isn't that too high?",
         "It depends on the use case. At α=0.10 (tight 90% coverage guarantee), the conformal prediction set is "
         "conservative — it would rather say 'uncertain' than risk a wrong confident prediction. For a loan product "
         "where false positives (approving a defaulter) are costly, 41% human review is acceptable: it catches all "
         "genuinely borderline cases. Tuning α to 0.20 reduces grey zone to ~25% but the guarantee weakens to 80%. "
         "The trade-off is deliberate and configurable."),

        # ── Fairness / Ethics ─────────────────────────────────────────────
        ("FAIRNESS, ETHICS & REGULATION", None),
        ("Q39: How would you test this model for demographic bias?",
         "Run demographic disparity analysis: compute approval rates and false positive rates separately for groups "
         "(gender, age bracket, region). Check equalized odds — false positive rate and false negative rate should "
         "be similar across groups. Measure disparate impact ratio: approval_rate_minority / approval_rate_majority "
         "should be ≥ 0.8 (the 80% rule from US EEOC). Note: CODE_GENDER is in IMMUTABLE_FEATURES — DiCE will "
         "never suggest changing gender as a recourse action, which is ethically correct."),
        ("Q40: What are the regulatory implications of this system?",
         "EU AI Act classifies credit scoring as 'high-risk AI.' Requirements include: (1) human oversight — the "
         "grey zone addresses this by flagging uncertain decisions for review. (2) Transparency — SHAP explanations "
         "and recourse paths fulfil the 'right to explanation.' (3) Non-discrimination — immutable feature constraints "
         "prevent the model from suggesting protected-attribute changes. (4) Auditability — all artifacts are saved "
         "with version metadata for regulatory inspection."),
        ("Q41: A recourse path suggests 'reduce loan amount by 30%' — isn't that obvious advice?",
         "Yes, and that's actually correct behaviour. The most achievable recourse paths often involve immediate "
         "adjustments (requesting a smaller loan, extending the term). These are structurally obvious but the model "
         "quantifies them: 'reducing by 30% gives you a 67% flip chance.' The value-add is the quantified flip "
         "probability and the effort ranking — a loan officer can now say 'this applicant would almost certainly "
         "qualify for a 25% smaller loan' rather than just 'try applying for less.'"),

        # ── Advanced ──────────────────────────────────────────────────────
        ("ADVANCED / SENIOR QUESTIONS", None),
        ("Q42: How would you improve the model AUC beyond 0.79?",
         "Current state: 13 Optuna trials, n_estimators capped at 800. Improvements: (1) Run all 60 trials with "
         "longer timeout. (2) Add more supplementary table features (bureau_balance.csv currently not used). "
         "(3) Add interaction features (EXT_SOURCE_MEAN × DAYS_EMPLOYED_RATIO). (4) Try LightGBM or CatBoost "
         "as alternatives. (5) Stack XGBoost with a logistic regression on SHAP values (stacking). "
         "Expected: 0.82-0.84 with full feature set and proper tuning."),
        ("Q43: How would you handle concept drift in production?",
         "Monitor: (1) P(default) distribution shift over time (if average p_default drifts, recalibrate MAPIE). "
         "(2) Feature distribution shift (log mean/std of incoming features, alert if Z-score > 3). "
         "(3) Grey zone rate changes (increases may indicate distribution shift). "
         "Response: periodic retraining on rolling window, or continual learning with XGBoost's update() method. "
         "MAPIE calibration is cheap (~5 minutes) and can be redone without full retraining."),
        ("Q44: Why not use a neural network for this task?",
         "XGBoost outperforms neural networks on tabular data with limited preprocessing — this is well-established "
         "(Grinsztajn et al., 2022, 'Why tree-based models still outperform deep learning on tabular data'). "
         "More importantly for this project: gradient-based DiCE requires a differentiable model, but the genetic "
         "DiCE backend works with any black-box. If a neural network were used, gradient-based DiCE would be available "
         "but would be slower and less diverse than genetic search for this feature space."),
        ("Q45: What is the difference between proximity and sparsity in counterfactuals?",
         "Proximity measures the distance between the original and CF in feature space (e.g., L1 or L2 norm). "
         "Sparsity measures how many features changed (L0 norm). A CF could be sparse but not proximate (change "
         "one feature by a huge amount) or proximate but not sparse (change 50 features by tiny amounts). DiCE "
         "optimises a weighted combination. For user-facing explanations, sparsity is more important — people "
         "prefer '3 things to change' over '30 things to change slightly.'"),
        ("Q46: How does DiCE ensure diversity across counterfactuals?",
         "DiCE adds a diversity term to the genetic algorithm's fitness function: CFs are penalised if they are "
         "too similar to each other (measured by mean pairwise L1 distance across CFs). The total fitness = "
         "validity_loss + proximity_loss - diversity_term. The diversity weight is a hyperparameter in DiCE; "
         "higher weight produces more varied paths but may sacrifice proximity. This is why Path 1 might change "
         "the loan amount while Path 2 changes employment and Path 3 changes credit scores."),
        ("Q47: What would a production monitoring dashboard track for this system?",
         "(1) Request latency by endpoint (predict vs recourse — expect 100ms vs 10-30s). "
         "(2) Zone distribution over time (approve%/grey%/deny%). "
         "(3) Recourse path length distribution (paths with too many steps suggest constraint relaxation needed). "
         "(4) Flip probability distribution (paths consistently below 60% suggest model has drifted). "
         "(5) Feature importance stability (SHAP values should be consistent across time). "
         "(6) Cache hit rate for repeat applicants."),
        ("Q48: Why does the effort ranker use MAX time across changed features, not total?",
         "Steps in a recourse path can be worked on in parallel. An applicant can simultaneously: reduce their "
         "loan amount (immediate), stop late payments (ongoing), and work on credit score improvement (ongoing). "
         "The bottleneck (longest step) determines when all conditions are met. Using MAX rather than SUM reflects "
         "that the total calendar time is determined by the slowest action, not the sum of all durations."),
        ("Q49: How would you validate that the counterfactual paths are actually achievable?",
         "Three validation approaches: (1) Simulation — apply the CF changes to real applicants in the test set, "
         "re-score with the model, and check that flip_probability matches actual approval rate within confidence "
         "intervals. (2) Human expert review — have loan officers score a sample of paths for plausibility. "
         "(3) Follow-up study — if deployed, track applicants who received recourse paths and compare 12-month "
         "re-application outcomes against predicted flip probabilities. This is the gold standard for recourse validation."),
        ("Q50: What would you do differently if you were building this for production?",
         "(1) Replace label encoding with target/frequency encoding for high-cardinality categoricals (more robust). "
         "(2) Use conformal prediction with conditional coverage (separate calibration per demographic group) for fairer guarantees. "
         "(3) Add a recourse feasibility filter: compare CF feature values against realistic percentiles in the training set "
         "to avoid 'impossible' suggestions. "
         "(4) Implement a database layer (PostgreSQL) to store application history and enable trend analysis. "
         "(5) Add authenticated API (OAuth2/JWT) with audit logging per the EU AI Act's traceability requirements. "
         "(6) Move DiCE generation to an async background task with WebSocket notifications to the frontend."),
    ]

    for item in qa_pairs:
        q, a = item
        if a is None:
            # Section header
            story += [SP(10), HR()]
            story += [P(f"<b>── {q} ──</b>", "h2")]
            story += [HR(), SP(4)]
        else:
            story += [KeepTogether([
                P(q, "qa_q"),
                P(a, "qa_a"),
                SP(2),
            ])]

    story += [PageBreak()]
    story += [HR()]
    story += [SP(20)]
    story += [P("End of Report", "caption")]
    story += [P("Credit Recourse Engine — XGBoost · MAPIE · DiCE-ML · FastAPI · Alpine.js", "caption")]

    return story

# ── Build PDF ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Building report → {OUTPUT_PATH}")
    doc = make_doc(OUTPUT_PATH)
    story = build_content()
    doc.build(story)
    size_mb = OUTPUT_PATH.stat().st_size / 1_048_576
    print(f"Done — {OUTPUT_PATH.name}  ({size_mb:.1f} MB)")
