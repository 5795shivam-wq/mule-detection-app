"""
Money Mule Risk Detection - Self-Service Dashboard
Upload an Excel file (with 'Accounts' and 'Transactions' sheets) and get
instant risk scoring, using the same 6-rule engine we built and verified.

IMPORTANT: This is an ALERT system, not a determination system.
Flagged accounts require human investigation - this tool never declares
"Mule Account = YES."

To run:
    pip install streamlit pandas numpy --break-system-packages
    streamlit run mule_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

st.set_page_config(page_title="Money Mule Risk Detection", layout="wide")

st.title("🔍 Money Mule Risk Detection Dashboard")
st.caption("Upload an Excel file with 'Accounts' and 'Transactions' sheets to run the rule engine.")

with st.expander("ℹ️ Expected file format (click to see required columns)"):
    st.markdown("""
    **Accounts sheet:** `account_id`, `account_type`, `account_open_date`, `six_month_amb`,
    `address`, `city`, `pin_code`, `phone`, `pan_ref`

    **Transactions sheet:** `account_id`, `timestamp`, `amount`, `type` (credit/debit), `counterparty_id`
    """)

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])


def score_accounts(accounts, txns):
    """Runs Rules 1, 2a, 2b, 3, 7, 11, 12 and the composite score on the given data."""
    phone_counts = accounts.groupby("phone")["account_id"].apply(list).to_dict()
    address_counts = accounts.groupby("address")["account_id"].apply(list).to_dict()

    def get_shared_identifier_flag(acc_id, phone, address):
        matches = []
        if phone in phone_counts and len(phone_counts[phone]) > 1:
            others = [a for a in phone_counts[phone] if a != acc_id]
            matches.append(f"phone shared with {', '.join(others)}")
        if address in address_counts and len(address_counts[address]) > 1 and str(address).strip() != "":
            others = [a for a in address_counts[address] if a != acc_id]
            matches.append(f"address shared with {', '.join(others)}")
        return matches

    results = []
    for _, acc in accounts.iterrows():
        aid = acc["account_id"]
        amb = acc["six_month_amb"]
        acc_txns = txns[txns["account_id"] == aid].sort_values("timestamp").reset_index(drop=True)
        evidence = []

        # Rule 1
        max_txn = acc_txns["amount"].max() if len(acc_txns) else 0
        ratio_r1 = max_txn / amb if amb > 0 else 0
        rule1_flag = ratio_r1 > 5
        if rule1_flag:
            evidence.append(f"Transaction/AMB anomaly: {ratio_r1:.1f}x (max txn Rs{max_txn:,.0f} vs AMB Rs{amb:,.0f})")

        # Rule 2a/2b
        acc_txns["date"] = acc_txns["timestamp"].dt.date
        symmetric_count_days, symmetric_value_days = 0, 0
        if len(acc_txns) > 0:
            daily = acc_txns.groupby(["date", "type"]).agg(count=("amount", "count"), value=("amount", "sum")).unstack(fill_value=0)
            if not daily.empty and "count" in daily.columns.get_level_values(0):
                credit_counts = daily["count"].get("credit", pd.Series(dtype=float))
                debit_counts = daily["count"].get("debit", pd.Series(dtype=float))
                credit_values = daily["value"].get("credit", pd.Series(dtype=float))
                debit_values = daily["value"].get("debit", pd.Series(dtype=float))
                for d in daily.index:
                    cc, dc = credit_counts.get(d, 0), debit_counts.get(d, 0)
                    cv, dv = credit_values.get(d, 0), debit_values.get(d, 0)
                    if cc + dc == 0:
                        continue
                    if abs(cc - dc) / (cc + dc) <= 0.05:
                        symmetric_count_days += 1
                    if cv + dv > 0 and abs(cv - dv) / (cv + dv) <= 0.05:
                        symmetric_value_days += 1
        rule2a_flag = symmetric_count_days >= 3
        rule2b_flag = symmetric_value_days >= 3
        if rule2a_flag:
            evidence.append(f"{symmetric_count_days} days with symmetric credit/debit COUNT")
        if rule2b_flag:
            evidence.append(f"{symmetric_value_days} days with symmetric credit/debit VALUE")

        # Rule 3
        credits = acc_txns[acc_txns["type"] == "credit"]
        debits = acc_txns[acc_txns["type"] == "debit"].copy()
        total_credit_value = credits["amount"].sum()
        passthrough_value, passthrough_times = 0, []
        for _, c in credits.iterrows():
            window_end = c["timestamp"] + timedelta(hours=24)
            matching_debits = debits[(debits["timestamp"] > c["timestamp"]) & (debits["timestamp"] <= window_end)]
            if not matching_debits.empty:
                linked = matching_debits.loc[matching_debits["amount"].idxmax()]
                matched_amt = min(linked["amount"], c["amount"])
                passthrough_value += matched_amt
                passthrough_times.append((linked["timestamp"] - c["timestamp"]).total_seconds() / 3600)
        passthrough_pct = (passthrough_value / total_credit_value * 100) if total_credit_value > 0 else 0
        median_hours = np.median(passthrough_times) if passthrough_times else None
        rule3_flag = passthrough_pct >= 80
        if rule3_flag:
            med_str = f", median {median_hours:.1f}h to debit" if median_hours is not None else ""
            evidence.append(f"{passthrough_pct:.0f}% of credited funds moved out within 24h{med_str}")

        # Rule 7
        rule7_flag, dormant_days, burst_window = False, None, pd.DataFrame()
        if len(acc_txns) >= 2:
            gaps = acc_txns["timestamp"].diff().dt.days.dropna()
            if len(gaps) > 0 and gaps.max() >= 30:
                gap_end_idx = gaps.idxmax()
                gap_end_time = acc_txns.loc[gap_end_idx, "timestamp"]
                burst_window = acc_txns[(acc_txns["timestamp"] >= gap_end_time) &
                                          (acc_txns["timestamp"] <= gap_end_time + timedelta(days=10))]
                if len(burst_window) >= 4:
                    rule7_flag = True
                    dormant_days = int(gaps.max())
        if rule7_flag:
            evidence.append(f"Dormant for {dormant_days} days, then {len(burst_window)} transactions within 10 days")

        # Rule 11
        kyc_fields = [acc.get("address", ""), acc.get("pin_code", ""), acc.get("pan_ref", "")]
        missing_count = sum(1 for f in kyc_fields if pd.isna(f) or str(f).strip() == "")
        invalid_phone = "X" in str(acc.get("phone", ""))
        rule11_flag = (missing_count > 0) or invalid_phone
        if rule11_flag:
            issues = []
            if pd.isna(acc.get("address")) or str(acc.get("address", "")).strip() == "":
                issues.append("missing address")
            if pd.isna(acc.get("pin_code")) or str(acc.get("pin_code", "")).strip() == "":
                issues.append("missing PIN code")
            if pd.isna(acc.get("pan_ref")) or str(acc.get("pan_ref", "")).strip() == "":
                issues.append("missing PAN reference")
            if invalid_phone:
                issues.append("invalid phone format")
            evidence.append(f"KYC incomplete: {', '.join(issues)}")

        # Rule 12
        shared_matches = get_shared_identifier_flag(aid, acc.get("phone"), acc.get("address"))
        rule12_flag = len(shared_matches) > 0
        if rule12_flag:
            evidence.append(f"Shared identifier risk: {'; '.join(shared_matches)}")

        # Composite score
        w_r1, w_r3, w_r2, w_r7, w_r11, w_r12 = 25, 20, 17, 13, 13, 12
        base_score = 0
        if rule1_flag:
            base_score += w_r1 * min(ratio_r1 / 10, 1.0)
        if rule3_flag:
            base_score += w_r3 * min(passthrough_pct / 100, 1.0)
        if rule2a_flag or rule2b_flag:
            sym_days = max(symmetric_count_days, symmetric_value_days)
            base_score += w_r2 * min(sym_days / 5, 1.0)
        if rule7_flag:
            base_score += w_r7
        if rule11_flag:
            severity = min(missing_count + (1 if invalid_phone else 0), 4) / 4
            base_score += w_r11 * severity
        if rule12_flag:
            base_score += w_r12

        behavioral_rules_triggered = sum([rule1_flag, rule2a_flag or rule2b_flag, rule3_flag, rule7_flag])
        agreement_bonus = min(max(0, behavioral_rules_triggered - 1) * 8, 24)
        score = round(min(base_score + agreement_bonus, 100), 1)

        if score >= 85:
            band = "CRITICAL"
        elif score >= 70:
            band = "HIGH"
        elif score >= 40:
            band = "MEDIUM"
        else:
            band = "LOW"

        results.append({
            "account_id": aid, "risk_score": score, "risk_band": band,
            "evidence": " | ".join(evidence) if evidence else "No rules triggered",
            "recommended_action": "Enhanced Review / Fraud & AML Investigation" if band in ["HIGH", "CRITICAL"]
                                    else ("Monitor" if band == "MEDIUM" else "No action"),
            "rule1_flag": rule1_flag, "rule2a_flag": rule2a_flag, "rule2b_flag": rule2b_flag,
            "rule3_flag": rule3_flag, "rule7_flag": rule7_flag, "rule11_flag": rule11_flag, "rule12_flag": rule12_flag,
        })

    return pd.DataFrame(results).sort_values("risk_score", ascending=False)


if uploaded_file is not None:
    try:
        accounts = pd.read_excel(uploaded_file, sheet_name="Accounts")
        txns = pd.read_excel(uploaded_file, sheet_name="Transactions", parse_dates=["timestamp"])
    except Exception as e:
        st.error(f"Could not read the file. Make sure it has 'Accounts' and 'Transactions' sheets with the expected columns. Error: {e}")
        st.stop()

    with st.spinner("Running rule engine..."):
        results = score_accounts(accounts, txns)

    st.success(f"Scored {len(results)} accounts.")

    # ---- Funnel metrics ----
    st.subheader("Detection Funnel")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Accounts", len(results))
    col2.metric("Rule 1 Triggered", int((results["rule1_flag"]).sum()))
    col3.metric("Medium or Above", int((results["risk_band"].isin(["MEDIUM", "HIGH", "CRITICAL"])).sum()))
    col4.metric("High or Above", int((results["risk_band"].isin(["HIGH", "CRITICAL"])).sum()))
    col5.metric("Investigation Queue (Critical)", int((results["risk_band"] == "CRITICAL").sum()))

    # ---- Risk band distribution ----
    st.subheader("Risk Band Distribution")
    band_counts = results["risk_band"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0)
    st.bar_chart(band_counts)

    # ---- Investigation queue table ----
    st.subheader("🚨 Investigation Queue (High & Critical)")
    queue = results[results["risk_band"].isin(["HIGH", "CRITICAL"])][
        ["account_id", "risk_score", "risk_band", "evidence", "recommended_action"]
    ]
    if queue.empty:
        st.info("No accounts currently require investigation.")
    else:
        st.dataframe(queue, use_container_width=True, hide_index=True)

    # ---- Full results, downloadable ----
    with st.expander("View full results (all accounts)"):
        st.dataframe(results, use_container_width=True, hide_index=True)

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download full results as CSV", csv, "mule_risk_scores.csv", "text/csv")

else:
    st.info("👆 Upload an Excel file to get started. Use the format shown above.")
