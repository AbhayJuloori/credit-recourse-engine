"""
Feature engineering pipeline for Home Credit Default Risk.

Aggregates 6 supplementary tables into per-applicant features.
Adds domain-knowledge derived features to the application table.
Target: push XGBoost AUC from ~0.76 (application only) → 0.80+.

Table hierarchy:
  application_train/test.csv          ← base (1 row per applicant)
  bureau.csv + bureau_balance.csv     ← credit bureau history
  previous_application.csv            ← prior Home Credit applications
  POS_CASH_balance.csv                ← POS and cash loan snapshots
  credit_card_balance.csv             ← credit card snapshots
  installments_payments.csv           ← installment payment history
"""

import gc
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────────────────────────────────────

def _agg_numeric(df: pd.DataFrame, group_var: str, prefix: str) -> pd.DataFrame:
    """Aggregate all numeric columns with mean / max / min / sum / std."""
    num_cols = [
        c for c in df.select_dtypes("number").columns
        if c != group_var
    ]
    agg = df.groupby(group_var)[num_cols].agg(["mean", "max", "min", "sum", "std"])
    agg.columns = [f"{prefix}_{col}_{stat}".upper() for col, stat in agg.columns]
    return agg.reset_index()


def _agg_categorical(df: pd.DataFrame, group_var: str, prefix: str) -> pd.DataFrame:
    """One-hot encode categoricals then sum counts per group."""
    cat_cols = [
        c for c in df.select_dtypes("object").columns
        if c != group_var
    ]
    if not cat_cols:
        return df[[group_var]].drop_duplicates()

    dummies = pd.get_dummies(df[cat_cols], prefix=cat_cols)
    dummies[group_var] = df[group_var].values
    agg = dummies.groupby(group_var).sum()
    agg.columns = [f"{prefix}_{c}".upper() for c in agg.columns]
    return agg.reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# Bureau + Bureau Balance
# ─────────────────────────────────────────────────────────────────────────────

def process_bureau(data_dir: Path) -> pd.DataFrame:
    """
    Aggregate bureau.csv + bureau_balance.csv → per SK_ID_CURR features.

    bureau_balance is first aggregated to bureau level (per SK_ID_BUREAU),
    then that result is merged with bureau and re-aggregated to applicant level.
    """
    logger.info("Processing bureau tables…")

    bureau = pd.read_csv(data_dir / "bureau.csv")
    bb = pd.read_csv(data_dir / "bureau_balance.csv")

    # ── Aggregate bureau_balance to bureau level ──────────────────────────────
    bb_counts = bb.groupby("SK_ID_BUREAU")["STATUS"].value_counts().unstack(fill_value=0)
    bb_counts.columns = [f"BB_STATUS_{s}_COUNT" for s in bb_counts.columns]
    bb_counts["BB_MONTHS_BALANCE_COUNT"] = bb.groupby("SK_ID_BUREAU").size()
    bb_counts["BB_MONTHS_BALANCE_MAX"] = bb.groupby("SK_ID_BUREAU")["MONTHS_BALANCE"].max()

    bureau = bureau.merge(bb_counts.reset_index(), on="SK_ID_BUREAU", how="left")
    del bb, bb_counts
    gc.collect()

    # ── Manual key aggregations (faster than generic for large table) ─────────
    g = bureau.groupby("SK_ID_CURR")
    agg = pd.DataFrame(index=pd.Index(bureau["SK_ID_CURR"].unique(), name="SK_ID_CURR"))

    agg["BUREAU_LOAN_COUNT"] = g.size()
    agg["BUREAU_ACTIVE_COUNT"] = (bureau["CREDIT_ACTIVE"] == "Active").groupby(bureau["SK_ID_CURR"]).sum()
    agg["BUREAU_CLOSED_COUNT"] = (bureau["CREDIT_ACTIVE"] == "Closed").groupby(bureau["SK_ID_CURR"]).sum()

    for col in ["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT",
                "AMT_CREDIT_SUM_OVERDUE", "AMT_ANNUITY"]:
        if col in bureau.columns:
            agg[f"BUREAU_{col}_MEAN"] = g[col].mean()
            agg[f"BUREAU_{col}_SUM"] = g[col].sum()
            agg[f"BUREAU_{col}_MAX"] = g[col].max()

    for col in ["DAYS_CREDIT", "DAYS_CREDIT_ENDDATE", "DAYS_CREDIT_UPDATE"]:
        if col in bureau.columns:
            agg[f"BUREAU_{col}_MEAN"] = g[col].mean()
            agg[f"BUREAU_{col}_MIN"] = g[col].min()
            agg[f"BUREAU_{col}_MAX"] = g[col].max()

    agg["BUREAU_CNT_CREDIT_PROLONG_SUM"] = g["CNT_CREDIT_PROLONG"].sum()
    agg["BUREAU_CREDIT_TYPE_NUNIQUE"] = g["CREDIT_TYPE"].nunique()

    # bureau_balance rollups
    for s_col in [c for c in bureau.columns if c.startswith("BB_")]:
        agg[f"BUREAU_{s_col}_MEAN"] = g[s_col].mean()
        agg[f"BUREAU_{s_col}_SUM"] = g[s_col].sum()

    result = agg.reset_index()
    del bureau, agg, g
    gc.collect()

    logger.info(f"Bureau features: {result.shape[1] - 1} columns")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Previous Applications
# ─────────────────────────────────────────────────────────────────────────────

def process_previous_applications(data_dir: Path) -> pd.DataFrame:
    """Aggregate previous_application.csv → per SK_ID_CURR features."""
    logger.info("Processing previous applications…")

    prev = pd.read_csv(data_dir / "previous_application.csv")

    # Replace anomalous values
    prev["DAYS_FIRST_DRAWING"].replace(365243, np.nan, inplace=True)
    prev["DAYS_FIRST_DUE"].replace(365243, np.nan, inplace=True)
    prev["DAYS_LAST_DUE_1ST_VERSION"].replace(365243, np.nan, inplace=True)
    prev["DAYS_LAST_DUE"].replace(365243, np.nan, inplace=True)
    prev["DAYS_TERMINATION"].replace(365243, np.nan, inplace=True)

    # Derived features
    prev["PREV_CREDIT_TO_APP_RATIO"] = prev["AMT_CREDIT"] / (prev["AMT_APPLICATION"] + 1e-9)
    prev["PREV_DOWN_TO_CREDIT_RATIO"] = prev["AMT_DOWN_PAYMENT"] / (prev["AMT_CREDIT"] + 1e-9)

    g = prev.groupby("SK_ID_CURR")
    agg = pd.DataFrame(index=pd.Index(prev["SK_ID_CURR"].unique(), name="SK_ID_CURR"))

    agg["PREV_APP_COUNT"] = g.size()
    agg["PREV_APPROVED_COUNT"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").groupby(prev["SK_ID_CURR"]).sum()
    agg["PREV_REFUSED_COUNT"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").groupby(prev["SK_ID_CURR"]).sum()
    agg["PREV_CANCELED_COUNT"] = (prev["NAME_CONTRACT_STATUS"] == "Canceled").groupby(prev["SK_ID_CURR"]).sum()
    agg["PREV_APPROVAL_RATE"] = agg["PREV_APPROVED_COUNT"] / (agg["PREV_APP_COUNT"] + 1e-9)

    for col in ["AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT", "AMT_DOWN_PAYMENT",
                "AMT_GOODS_PRICE", "CNT_PAYMENT", "PREV_CREDIT_TO_APP_RATIO",
                "PREV_DOWN_TO_CREDIT_RATIO"]:
        if col in prev.columns:
            agg[f"PREV_{col}_MEAN"] = g[col].mean()
            agg[f"PREV_{col}_MAX"] = g[col].max()
            agg[f"PREV_{col}_MIN"] = g[col].min()

    for col in ["DAYS_DECISION", "DAYS_FIRST_DUE", "DAYS_LAST_DUE"]:
        if col in prev.columns:
            agg[f"PREV_{col}_MEAN"] = g[col].mean()
            agg[f"PREV_{col}_MAX"] = g[col].max()

    agg["PREV_NAME_YIELD_GROUP_HIGH_COUNT"] = (
        (prev["NAME_YIELD_GROUP"] == "high").groupby(prev["SK_ID_CURR"]).sum()
    )
    agg["PREV_NAME_YIELD_GROUP_LOW_NORMAL_COUNT"] = (
        prev["NAME_YIELD_GROUP"].isin(["low_normal", "low_action"]).groupby(prev["SK_ID_CURR"]).sum()
    )

    # Approved-only stats
    approved = prev[prev["NAME_CONTRACT_STATUS"] == "Approved"]
    if len(approved) > 0:
        ga = approved.groupby("SK_ID_CURR")
        agg["PREV_APPROVED_AMT_CREDIT_MEAN"] = ga["AMT_CREDIT"].mean()
        agg["PREV_APPROVED_CNT_PAYMENT_MEAN"] = ga["CNT_PAYMENT"].mean()

    result = agg.reset_index()
    del prev, agg, g
    gc.collect()

    logger.info(f"Previous application features: {result.shape[1] - 1} columns")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# POS CASH Balance
# ─────────────────────────────────────────────────────────────────────────────

def process_pos_cash(data_dir: Path) -> pd.DataFrame:
    """Aggregate POS_CASH_balance.csv → per SK_ID_CURR features."""
    logger.info("Processing POS CASH balance…")

    pos = pd.read_csv(data_dir / "POS_CASH_balance.csv")

    g = pos.groupby("SK_ID_CURR")
    agg = pd.DataFrame(index=pd.Index(pos["SK_ID_CURR"].unique(), name="SK_ID_CURR"))

    agg["POS_COUNT"] = g.size()
    agg["POS_MONTHS_BALANCE_MEAN"] = g["MONTHS_BALANCE"].mean()
    agg["POS_MONTHS_BALANCE_SIZE"] = g["MONTHS_BALANCE"].nunique()

    for col in ["SK_DPD", "SK_DPD_DEF", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE"]:
        if col in pos.columns:
            agg[f"POS_{col}_MEAN"] = g[col].mean()
            agg[f"POS_{col}_MAX"] = g[col].max()
            agg[f"POS_{col}_SUM"] = g[col].sum()

    agg["POS_COMPLETED_COUNT"] = (
        (pos["NAME_CONTRACT_STATUS"] == "Completed").groupby(pos["SK_ID_CURR"]).sum()
    )
    agg["POS_ACTIVE_COUNT"] = (
        (pos["NAME_CONTRACT_STATUS"] == "Active").groupby(pos["SK_ID_CURR"]).sum()
    )

    result = agg.reset_index()
    del pos, agg, g
    gc.collect()

    logger.info(f"POS CASH features: {result.shape[1] - 1} columns")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Credit Card Balance
# ─────────────────────────────────────────────────────────────────────────────

def process_credit_card(data_dir: Path) -> pd.DataFrame:
    """Aggregate credit_card_balance.csv → per SK_ID_CURR features."""
    logger.info("Processing credit card balance…")

    cc = pd.read_csv(data_dir / "credit_card_balance.csv")

    # Derived ratios
    cc["CC_LIMIT_USE_RATIO"] = cc["AMT_BALANCE"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1e-9)
    cc["CC_PAYMENT_RATIO"] = cc["AMT_PAYMENT_CURRENT"] / (cc["AMT_BALANCE"] + 1e-9)
    cc["CC_DRAWING_RATIO"] = cc["AMT_DRAWINGS_CURRENT"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1e-9)

    g = cc.groupby("SK_ID_CURR")
    agg = pd.DataFrame(index=pd.Index(cc["SK_ID_CURR"].unique(), name="SK_ID_CURR"))

    agg["CC_COUNT"] = g.size()
    agg["CC_MONTHS_BALANCE_MEAN"] = g["MONTHS_BALANCE"].mean()

    for col in ["AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_ATM_CURRENT",
                "AMT_DRAWINGS_CURRENT", "AMT_DRAWINGS_OTHER_CURRENT",
                "AMT_DRAWINGS_POS_CURRENT", "AMT_PAYMENT_CURRENT",
                "AMT_PAYMENT_TOTAL_CURRENT", "AMT_RECEIVABLE_PRINCIPAL",
                "CC_LIMIT_USE_RATIO", "CC_PAYMENT_RATIO", "CC_DRAWING_RATIO",
                "CNT_DRAWINGS_ATM_CURRENT", "CNT_DRAWINGS_CURRENT",
                "SK_DPD", "SK_DPD_DEF"]:
        if col in cc.columns:
            agg[f"CC_{col}_MEAN"] = g[col].mean()
            agg[f"CC_{col}_MAX"] = g[col].max()
            agg[f"CC_{col}_MIN"] = g[col].min()

    result = agg.reset_index()
    del cc, agg, g
    gc.collect()

    logger.info(f"Credit card features: {result.shape[1] - 1} columns")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Installments Payments
# ─────────────────────────────────────────────────────────────────────────────

def process_installments(data_dir: Path) -> pd.DataFrame:
    """Aggregate installments_payments.csv → per SK_ID_CURR features."""
    logger.info("Processing installments payments…")

    # Load in chunks to handle 13.6M rows — reduce peak memory
    chunks = []
    for chunk in pd.read_csv(
        data_dir / "installments_payments.csv", chunksize=2_000_000
    ):
        # Derived columns before aggregation
        chunk["INST_DPD"] = chunk["DAYS_ENTRY_PAYMENT"] - chunk["DAYS_INSTALMENT"]
        chunk["INST_DBD"] = chunk["DAYS_INSTALMENT"] - chunk["DAYS_ENTRY_PAYMENT"]
        chunk["INST_PAYMENT_RATIO"] = chunk["AMT_PAYMENT"] / (chunk["AMT_INSTALMENT"] + 1e-9)
        chunk["INST_PAYMENT_DIFF"] = chunk["AMT_PAYMENT"] - chunk["AMT_INSTALMENT"]
        chunks.append(chunk)

    inst = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    inst["INST_DPD"] = inst["INST_DPD"].clip(lower=0)
    inst["INST_DBD"] = inst["INST_DBD"].clip(lower=0)

    g = inst.groupby("SK_ID_CURR")
    agg = pd.DataFrame(index=pd.Index(inst["SK_ID_CURR"].unique(), name="SK_ID_CURR"))

    agg["INSTAL_COUNT"] = g.size()
    agg["INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE"] = g["NUM_INSTALMENT_VERSION"].nunique()

    for col in ["INST_DPD", "INST_DBD", "INST_PAYMENT_RATIO", "INST_PAYMENT_DIFF",
                "AMT_INSTALMENT", "AMT_PAYMENT"]:
        agg[f"INSTAL_{col}_MEAN"] = g[col].mean()
        agg[f"INSTAL_{col}_MAX"] = g[col].max()
        agg[f"INSTAL_{col}_SUM"] = g[col].sum()
        agg[f"INSTAL_{col}_STD"] = g[col].std()

    # Late payment count and rate
    agg["INSTAL_LATE_COUNT"] = g.apply(lambda x: (x["INST_DPD"] > 0).sum())
    agg["INSTAL_LATE_RATE"] = agg["INSTAL_LATE_COUNT"] / (agg["INSTAL_COUNT"] + 1e-9)

    result = agg.reset_index()
    del inst, agg, g
    gc.collect()

    logger.info(f"Installment features: {result.shape[1] - 1} columns")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Application-level engineered features
# ─────────────────────────────────────────────────────────────────────────────

def engineer_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-knowledge derived features to the application table.
    All operations on existing columns only — no external data required.
    """
    df = df.copy()

    # ── Financial ratios ─────────────────────────────────────────────────────
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1e-9)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1e-9)
    df["CREDIT_TERM"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1e-9)
    df["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1e-9)
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"].replace(0, 1))
    if "AMT_GOODS_PRICE" in df.columns:
        df["GOODS_TO_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + 1e-9)

    # ── Employment ───────────────────────────────────────────────────────────
    # DAYS_EMPLOYED = 365243 means unemployed in this dataset
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df["DAYS_EMPLOYED_RATIO"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"].abs() + 1e-9)

    # ── Age ──────────────────────────────────────────────────────────────────
    df["AGE_YEARS"] = df["DAYS_BIRTH"].abs() / 365.25

    # ── Time-since ratios (guard optional columns) ───────────────────────────
    if "DAYS_ID_PUBLISH" in df.columns:
        df["ID_TO_BIRTH_RATIO"] = df["DAYS_ID_PUBLISH"].abs() / (df["DAYS_BIRTH"].abs() + 1e-9)
    if "DAYS_LAST_PHONE_CHANGE" in df.columns:
        df["PHONE_TO_BIRTH_RATIO"] = df["DAYS_LAST_PHONE_CHANGE"].abs() / (df["DAYS_BIRTH"].abs() + 1e-9)
        df["PHONE_TO_EMPLOY_RATIO"] = df["DAYS_LAST_PHONE_CHANGE"].abs() / (df["DAYS_EMPLOYED"].abs() + 1e-9)

    # ── External source combinations ─────────────────────────────────────────
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    available_ext = [c for c in ext_cols if c in df.columns]
    if available_ext:
        df["EXT_SOURCE_MEAN"] = df[available_ext].mean(axis=1)
        df["EXT_SOURCE_STD"] = df[available_ext].std(axis=1)
        df["EXT_SOURCE_WEIGHTED"] = (
            df.get("EXT_SOURCE_1", 0) * 2
            + df.get("EXT_SOURCE_2", 3)
            + df.get("EXT_SOURCE_3", 2)
        ) / (7 + 1e-9)
        df["EXT_SOURCE_PRODUCT"] = df[available_ext].prod(axis=1)

    # ── OWN_CAR_AGE ratios ───────────────────────────────────────────────────
    if "OWN_CAR_AGE" in df.columns:
        df["CAR_TO_BIRTH_RATIO"] = df["OWN_CAR_AGE"] / (df["AGE_YEARS"] + 1e-9)
        df["CAR_TO_EMPLOY_RATIO"] = df["OWN_CAR_AGE"] / (df["DAYS_EMPLOYED"].abs() / 365.25 + 1e-9)

    # ── Document flags sum ───────────────────────────────────────────────────
    flag_doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if flag_doc_cols:
        df["FLAG_DOCUMENTS_SUM"] = df[flag_doc_cols].sum(axis=1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Master builder
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    data_dir: Path,
    split: str = "train",
    use_supplementary: bool = True,
) -> pd.DataFrame:
    """
    Load all tables, engineer features, and return a merged dataframe.

    Args:
        data_dir: directory containing Home Credit CSV files
        split: 'train' or 'test'
        use_supplementary: if False, use only application table features

    Returns:
        DataFrame with all engineered features. 'TARGET' column present
        only for split='train'.
    """
    logger.info(f"Building features for split='{split}'…")

    csv_name = "application_train.csv" if split == "train" else "application_test.csv"
    app = pd.read_csv(data_dir / csv_name)

    # ── Application-level engineering ────────────────────────────────────────
    app = engineer_application_features(app)
    logger.info(f"Application rows: {len(app)}, cols after engineering: {app.shape[1]}")

    if not use_supplementary:
        return app

    # ── Merge supplementary tables ────────────────────────────────────────────
    bureau_feats = process_bureau(data_dir)
    prev_feats = process_previous_applications(data_dir)
    pos_feats = process_pos_cash(data_dir)
    cc_feats = process_credit_card(data_dir)
    inst_feats = process_installments(data_dir)

    for feats, name in [
        (bureau_feats, "bureau"),
        (prev_feats, "previous_application"),
        (pos_feats, "POS_CASH"),
        (cc_feats, "credit_card"),
        (inst_feats, "installments"),
    ]:
        before = app.shape[1]
        app = app.merge(feats, on="SK_ID_CURR", how="left")
        logger.info(f"After {name} merge: +{app.shape[1] - before} cols")
        del feats
        gc.collect()

    logger.info(f"Final feature matrix: {app.shape[0]} rows × {app.shape[1]} cols")
    return app


# ─────────────────────────────────────────────────────────────────────────────
# Label encoding helper
# ─────────────────────────────────────────────────────────────────────────────

def encode_categoricals(
    df: pd.DataFrame,
    encoders: Optional[dict] = None,
) -> tuple:
    """
    Label-encode all object columns.

    If encoders is None, fit new encoders on df.
    Returns (encoded_df, encoders_dict).
    """
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()
    fitted = encoders or {}

    for col in df.select_dtypes("object").columns:
        le = fitted.get(col, LabelEncoder())
        if col not in fitted:
            le.fit(df[col].astype(str).fillna("NaN"))
            fitted[col] = le
        df[col] = le.transform(df[col].astype(str).fillna("NaN"))

    return df, fitted
