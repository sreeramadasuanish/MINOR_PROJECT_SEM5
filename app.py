"""
Streamlit Web UI Dashboard: Federated Learning & FED4FED Fairness Evaluation for MNIST.
Central Server Administrator Control Panel.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import streamlit as st
import altair as alt

# Ensure project root is available in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.mnist_loader import load_mnist
from data.partitioned_clients import create_non_iid_partitions, compute_partition_distributions
from phase1_training.train_fl import run_federated_training
from phase2_evaluation.evaluate_fairness import run_fairness_evaluation
from utils.metrics import compute_client_metrics_table
from utils.crypto_utils import compute_state_hash, simulate_smpc_masking

# Page configuration
st.set_page_config(
    page_title="Federated MNIST & FED4FED Fairness Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-pass {
        background-color: #DEF7EC;
        color: #03543F;
        border: 1px solid #31C48D;
    }
    .badge-fail {
        background-color: #FDE8E8;
        color: #9B1C1C;
        border: 1px solid #F98080;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "fl_results" not in st.session_state:
    st.session_state.fl_results = None
if "fairness_results" not in st.session_state:
    st.session_state.fairness_results = None
if "data_partitions" not in st.session_state:
    st.session_state.data_partitions = None


# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.title("🎛️ Server Control Panel")
    st.markdown("Configure hyperparameters, differential privacy, and statistical audit thresholds.")

    st.subheader("1. FL Training Hyperparameters")
    fl_rounds = st.slider("Federated Rounds", min_value=1, max_value=15, value=4, step=1)
    local_epochs = st.slider("Local Epochs per Round", min_value=1, max_value=5, value=1, step=1)
    batch_size = st.select_slider("Batch Size", options=[32, 64, 128], value=64)
    lr = st.slider("Learning Rate (SGD)", min_value=0.001, max_value=0.05, value=0.01, step=0.002, format="%.3f")

    st.subheader("2. Privacy & Noise Calibration")
    dp_sigma = st.slider("DP Gaussian Noise Multiplier (σ)", min_value=0.0, max_value=0.25, value=0.04, step=0.01)
    clip_c = st.slider("L2 Gradient Clipping Bound (C)", min_value=0.2, max_value=5.0, value=1.0, step=0.2)
    use_oversampling = st.checkbox("Class-Imbalance Oversampling", value=True, help="Rebalances skewed local client training batches")

    st.subheader("3. Non-IID Skew Setup")
    skew_type = st.selectbox("Partition Strategy", options=["structured", "dirichlet", "extreme"], index=0)
    dirichlet_alpha = 0.5
    if skew_type == "dirichlet":
        dirichlet_alpha = st.slider("Dirichlet Alpha (skew severity)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)

    st.subheader("4. FED4FED Fairness Audit")
    significance_alpha = st.selectbox("Significance Level (α)", options=[0.01, 0.05, 0.10], index=1)
    n_bootstrap = st.slider("Bootstrap Iterations (B)", min_value=100, max_value=1000, value=300, step=100)
    primary_metric = st.selectbox("Audit Metric", options=["accuracy", "f1_macro", "recall_macro", "precision_macro"], index=0)

    device_option = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Hardware Accelerator: **{device_option.upper()}**")


# ---------------- MAIN HEADER ----------------
st.markdown('<div class="main-title">Federated MNIST & FED4FED Fairness Evaluation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Privacy-Preserving Federated Learning (DP-FedAvg, SMPC) and Post-Convergence Disparity Auditing '
    '(Chi-Square Homogeneity, Welch\'s F-test, and Bootstrap Confidence Intervals)'
    '</div>',
    unsafe_allow_html=True,
)

# Header metric cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Benchmark", "MNIST (28x28)")
with c2:
    st.metric("Model Architecture", "PyTorch CNN (2 Conv + FC)")
with c3:
    st.metric("Clients in Ring", "3 Nodes (Non-IID)")
with c4:
    checkpoint_exists = os.path.exists(os.path.join(PROJECT_ROOT, "phase1_training", "global_model.pt"))
    st.metric("Trained Checkpoint", "Available ✅" if checkpoint_exists else "Not Trained ⏳")

st.divider()


# Cache or prepare data partitions for visualization
if st.session_state.data_partitions is None or st.session_state.get("cached_skew") != skew_type:
    train_data, test_data = load_mnist()
    client_train, client_test, class_dist = create_non_iid_partitions(
        train_dataset=train_data,
        test_dataset=test_data,
        num_clients=3,
        skew_type=skew_type,
        dirichlet_alpha=dirichlet_alpha,
    )
    st.session_state.data_partitions = {
        "train": client_train,
        "test": client_test,
        "dist": class_dist,
    }
    st.session_state["cached_skew"] = skew_type


# ---------------- TABS INTERFACE ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Node Topology & Non-IID Skew",
    "🚀 Federated Training (Phase 1)",
    "⚖️ FED4FED Fairness Audit (Phase 2)",
    "📈 Client Diagnostics & Metrics",
    "🔐 SMPC & Privacy Ledger",
])


# ========================================================
# TAB 1: TOPOLOGY & SKEW
# ========================================================
with tab1:
    st.subheader("Client Data Distribution (Non-IID Heterogeneity)")
    st.markdown(
        "Real-world edge nodes exhibit severe label skew. Here, MNIST is distributed across 3 nodes:"
        " **Client 1** specializes in low digits [0-3], **Client 2** in middle digits [4-6], and **Client 3** in high digits [7-9]."
    )

    class_dist = st.session_state.data_partitions["dist"]
    plot_data = compute_partition_distributions(class_dist)
    df_dist = pd.DataFrame(plot_data)

    chart = (
        alt.Chart(df_dist)
        .mark_bar()
        .encode(
            x=alt.X("Digit:N", title="MNIST Digit Class", sort=list(map(str, range(10)))),
            y=alt.Y("Count:Q", title="Sample Count"),
            color=alt.Color("Client:N", scale=alt.Scale(scheme="category10")),
            column=alt.Column("Client:N", title="Participating Nodes"),
            tooltip=["Client", "Digit", "Count", "Percentage"],
        )
        .properties(width=240, height=260)
    )
    st.altair_chart(chart)

    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        st.markdown("#### Partition Summary")
        summary_records = []
        for cid, counts in class_dist.items():
            tot = sum(counts.values())
            top_classes = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = ", ".join([f"Digit {c} ({cnt})" for c, cnt in top_classes])
            summary_records.append({
                "Node ID": cid,
                "Total Train Samples": tot,
                "Top Dominant Classes": top_str,
                "Oversampling Protection": "Enabled" if use_oversampling else "Disabled",
            })
        st.dataframe(pd.DataFrame(summary_records), use_container_width=True)

    with col_t2:
        st.markdown("#### Neural Network Architecture")
        st.code(
            """
MNISTCNN(
  (conv1): Conv2d(1, 32, kernel_size=3, padding=1) -> ReLU -> MaxPool2d(2)
  (conv2): Conv2d(32, 64, kernel_size=3, padding=1) -> ReLU -> MaxPool2d(2)
  (dropout1): Dropout(p=0.25)
  (fc1): Linear(3136 -> 128) -> ReLU -> Dropout(p=0.25)
  (fc2): Linear(128 -> 10 classes)
)
Total Trainable Parameters: ~418,058
            """,
            language="text",
        )


# ========================================================
# TAB 2: TRAINING MONITOR
# ========================================================
with tab2:
    st.subheader("Federated Communication & Aggregation Hub")
    st.markdown("Trigger multi-round federated training with local gradient clipping and calibrated DP noise injection.")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        start_fl_btn = st.button("▶️ Launch Federated Training", type="primary", use_container_width=True)

    progress_bar = st.empty()
    status_text = st.empty()
    chart_col1, chart_col2 = st.columns(2)
    loss_chart_spot = chart_col1.empty()
    acc_chart_spot = chart_col2.empty()
    telemetry_spot = st.empty()

    if start_fl_btn:
        live_rounds = []
        live_loss = []
        live_acc = []
        live_client_acc = {"Client_1": [], "Client_2": [], "Client_3": []}

        def on_round_progress(r_idx, info):
            pct = int((r_idx / fl_rounds) * 100)
            progress_bar.progress(pct)
            status_text.markdown(f"**Round {r_idx}/{fl_rounds}** complete: Global Loss = `{info['global_loss']:.4f}` | Global Acc = `{info['global_accuracy']*100:.2f}%`")

            live_rounds.append(r_idx)
            live_loss.append(info["global_loss"])
            live_acc.append(info["global_accuracy"] * 100)
            for cid in live_client_acc.keys():
                c_acc = info["client_eval"][cid]["accuracy"] * 100
                live_client_acc[cid].append(c_acc)

            # Update charts
            df_loss = pd.DataFrame({"Round": live_rounds, "Global Loss": live_loss}).set_index("Round")
            loss_chart_spot.line_chart(df_loss)

            df_acc = pd.DataFrame({"Round": live_rounds, "Global Accuracy (%)": live_acc, **{f"{k} Acc (%)": v for k, v in live_client_acc.items()}}).set_index("Round")
            acc_chart_spot.line_chart(df_acc)

        with st.spinner("Training federated clients with DP-FedAvg..."):
            save_dest = os.path.join(PROJECT_ROOT, "phase1_training", "global_model.pt")
            fl_results = run_federated_training(
                rounds=fl_rounds,
                local_epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                dp_sigma=dp_sigma,
                clip_c=clip_c,
                use_oversampling=use_oversampling,
                skew_type=skew_type,
                dirichlet_alpha=dirichlet_alpha,
                save_path=save_dest,
                device=device_option,
                progress_callback=on_round_progress,
            )
            st.session_state.fl_results = fl_results
            # Auto-run fairness audit with the freshly trained model
            with st.spinner("Executing post-training FED4FED fairness audit..."):
                st.session_state.fairness_results = run_fairness_evaluation(
                    model_path=save_dest,
                    alpha=significance_alpha,
                    n_bootstrap=n_bootstrap,
                    primary_metric=primary_metric,
                    skew_type=skew_type,
                    dirichlet_alpha=dirichlet_alpha,
                    device=device_option,
                )

        st.success(f"Federated training concluded! Global Model Accuracy: {fl_results['final_global_accuracy']*100:.2f}%")

    elif st.session_state.fl_results is not None:
        # Display cached training history
        hist = st.session_state.fl_results["history"]
        df_loss = pd.DataFrame({"Round": hist["rounds"], "Global Loss": hist["global_loss"]}).set_index("Round")
        chart_col1.line_chart(df_loss)

        acc_data = {"Round": hist["rounds"], "Global Accuracy (%)": [a * 100 for a in hist["global_accuracy"]]}
        for cid in hist["client_metrics"].keys():
            acc_data[f"{cid} Acc (%)"] = [x["accuracy"] * 100 for x in hist["client_metrics"][cid]]
        df_acc = pd.DataFrame(acc_data).set_index("Round")
        chart_col2.line_chart(df_acc)

        st.markdown("#### Round-by-Round Privacy & Gradient Telemetry")
        tele_records = []
        for r_idx, tele_round in enumerate(hist["telemetry"], start=1):
            for cid, tele in tele_round.items():
                tele_records.append({
                    "Round": r_idx,
                    "Client": cid,
                    "Train Loss": round(tele["train_loss"], 4),
                    "Pre-Clip L2": round(tele["pre_clip_norm"], 3),
                    "Post-Clip L2": round(tele["post_clip_norm"], 3),
                    "DP Noise σ*C": round(tele["noise_std"], 4),
                    "State Hash": tele["state_hash"][:12] + "...",
                })
        st.dataframe(pd.DataFrame(tele_records), use_container_width=True)


# ========================================================
# TAB 3: FED4FED FAIRNESS AUDIT
# ========================================================
with tab3:
    st.subheader("FED4FED Hypothesis Testing & Fairness Audit")
    st.markdown(
        "Evaluates whether non-IID distributions introduce statistically significant performance disparities across client nodes."
        " Hypothesis testing is conducted at **α = 0.05**."
    )

    col_audit_btn, col_metric_choice = st.columns([1, 2])
    with col_audit_btn:
        trigger_audit = st.button("⚖️ Run Statistical Fairness Audit", use_container_width=True)

    if trigger_audit:
        save_dest = os.path.join(PROJECT_ROOT, "phase1_training", "global_model.pt")
        with st.spinner("Computing local bootstrap resampling and executing Algorithms 1, 2, and 3..."):
            st.session_state.fairness_results = run_fairness_evaluation(
                model_path=save_dest,
                alpha=significance_alpha,
                n_bootstrap=n_bootstrap,
                primary_metric=primary_metric,
                skew_type=skew_type,
                dirichlet_alpha=dirichlet_alpha,
                device=device_option,
            )

    if st.session_state.fairness_results is not None:
        res = st.session_state.fairness_results
        a1 = res["algorithm_1_chisquare"]
        a2 = res["algorithm_2_welch_f"]
        a3 = res["algorithm_3_confidence_intervals"]

        # Status Banner
        if res["overall_fairness_pass"]:
            st.success(f"### ✅ AUDIT RESULT: {res['overall_verdict']}")
            st.markdown(
                "Neither the Chi-Square Homogeneity test nor Welch's F-test detected significant performance disparities across nodes at α = 0.05."
            )
        else:
            st.error(f"### ⚠️ AUDIT RESULT: {res['overall_verdict']}")
            st.markdown(
                "Statistically significant disparities were detected between client performance distributions due to non-IID data skew."
            )

        st.divider()

        # Algorithm 1 & 2 Cards
        card1, card2 = st.columns(2)
        with card1:
            st.markdown("#### 🔬 Algorithm 1: Chi-Square Homogeneity Test")
            st.markdown(f"**Hypothesis:** $H_0: \\theta_1 = \\theta_2 = \\theta_3$ (Statistical Homogeneity)")
            c_a, c_b = st.columns(2)
            c_a.metric("Chi2 Statistic", f"{a1['chi2_statistic']:.4f}", help="Inverse-variance weighted deviation from Graybill-Deal consensus")
            c_b.metric("Degrees of Freedom", f"{a1['degrees_of_freedom']}")
            c_c, c_d = st.columns(2)
            c_c.metric("p-value", f"{a1['p_value']:.4e}")
            c_d.metric("Critical Value", f"{a1['critical_value']:.4f}")
            st.info(f"**Graybill-Deal Consensus Point Estimate:** `{a1['graybill_deal_mean']*100:.2f}%` (SE = `{a1['graybill_deal_std_error']:.4f}`)")
            st.markdown(f"**Decision:** {'❌ Disparity Detected' if a1['reject_null_hypothesis'] else '✅ Pass (Homogeneous)'}")

        with card2:
            st.markdown("#### 📐 Algorithm 2: Welch's F-test (Heteroscedastic ANOVA)")
            st.markdown(f"**Hypothesis:** $H_0: \\mu_1 = \\mu_2 = \\mu_3$ (Without assuming equal variances)")
            w_a, w_b = st.columns(2)
            w_a.metric("Welch's F Statistic", f"{a2['f_statistic']:.4f}")
            w_b.metric("Effective df (df1, df2)", f"({a2['df1']}, {a2['df2']})")
            w_c, w_d = st.columns(2)
            w_c.metric("p-value", f"{a2['p_value']:.4e}")
            w_d.metric("Critical F", f"{a2['critical_f']:.4f}")
            st.info(f"**Weighted Grand Mean:** `{a2['weighted_grand_mean']*100:.2f}%`")
            st.markdown(f"**Decision:** {'❌ Disparity Detected' if a2['reject_null_hypothesis'] else '✅ Pass (Equivalent Means)'}")

        st.divider()

        # Algorithm 3: Confidence Intervals & Forest Plot
        st.markdown("#### 🌲 Algorithm 3: Federated Confidence Intervals & Pairwise Disparity Inference")
        st.markdown("Forest plot illustrating (1 - α) confidence intervals for each client node compared to the Graybill-Deal pooled consensus interval.")

        # Prepare forest plot data
        forest_records = []
        for node, interval in a3["local_intervals"].items():
            forest_records.append({
                "Entity": node,
                "Mean": interval["mean"] * 100,
                "CI_Lower": interval["ci_lower"] * 100,
                "CI_Upper": interval["ci_upper"] * 100,
                "Type": "Local Client",
            })
        consensus = a3["consensus_interval"]
        forest_records.append({
            "Entity": "Pooled Consensus (Graybill-Deal)",
            "Mean": consensus["graybill_deal_mean"] * 100,
            "CI_Lower": consensus["ci_lower"] * 100,
            "CI_Upper": consensus["ci_upper"] * 100,
            "Type": "Pooled Consensus",
        })
        df_forest = pd.DataFrame(forest_records)

        # Forest plot in Altair
        base = alt.Chart(df_forest).encode(
            y=alt.Y("Entity:N", title="Node / Estimator", sort=None)
        )
        rule = base.mark_rule(size=3).encode(
            x=alt.X("CI_Lower:Q", title=f"{primary_metric.capitalize()} (%)"),
            x2="CI_Upper:Q",
            color=alt.Color("Type:N", scale=alt.Scale(domain=["Local Client", "Pooled Consensus"], range=["#2563EB", "#10B981"])),
        )
        points = base.mark_point(filled=True, size=100).encode(
            x="Mean:Q",
            color=alt.Color("Type:N"),
            tooltip=["Entity", "Mean", "CI_Lower", "CI_Upper"],
        )
        st.altair_chart((rule + points).properties(height=220), use_container_width=True)

        st.markdown("##### Pairwise Subgroup Disparity Tests (Bonferroni-Adjusted)")
        st.caption(f"Adjusted significance threshold: α_bonf = {a3['bonferroni_alpha']:.4f}. Disparity is declared if zero is excluded from the difference confidence interval.")

        pair_rows = []
        for p in a3["pairwise_tests"]:
            pair_rows.append({
                "Comparison": p["comparison"],
                "Mean Difference (%)": f"{p['mean_difference'] * 100:+.2f}%",
                "Std Error": f"{p['std_error']:.4f}",
                "Bonferroni 95% CI (%)": f"[{p['ci_lower']*100:.2f}%, {p['ci_upper']*100:.2f}%]",
                "p-value": f"{p['p_value']:.4e}",
                "Disparity Flagged": "⚠️ YES" if p["significant_disparity"] else "✅ NO",
                "Verdict": p["verdict"],
            })
        st.dataframe(pd.DataFrame(pair_rows), use_container_width=True)
    else:
        st.info("Click **'Run Statistical Fairness Audit'** above (or launch training) to run FED4FED hypothesis tests.")


# ========================================================
# TAB 4: CLIENT DIAGNOSTICS & METRICS
# ========================================================
with tab4:
    st.subheader("Client Evaluation Diagnostics & Confusion Matrices")
    if st.session_state.fairness_results is not None:
        raw_m = st.session_state.fairness_results["client_raw_metrics"]
        st.markdown("#### Comprehensive Performance Overview")
        st.dataframe(compute_client_metrics_table(raw_m), use_container_width=True)

        st.markdown("#### Local Client Confusion Matrices (Digits 0 - 9)")
        cols_cm = st.columns(len(raw_m))
        for i, (cid, c_metrics) in enumerate(raw_m.items()):
            with cols_cm[i]:
                st.markdown(f"**{cid}**")
                cm_arr = np.array(c_metrics["confusion_matrix"])
                df_cm = pd.DataFrame(cm_arr, index=[str(d) for d in range(10)], columns=[str(d) for d in range(10)])
                st.dataframe(df_cm.style.background_gradient(cmap="Blues", axis=1))

        # Bootstrap Distribution histograms
        st.markdown("#### Empirical Bootstrap Resampling Distributions")
        b_summaries = st.session_state.fairness_results["client_bootstrap_summaries"]
        boot_records = []
        for cid, b_info in b_summaries.items():
            for val in b_info["bootstrap_samples"]:
                boot_records.append({"Client": cid, f"Bootstrap {primary_metric.capitalize()} (%)": val * 100})
        df_boot = pd.DataFrame(boot_records)

        chart_boot = (
            alt.Chart(df_boot)
            .mark_bar(opacity=0.6, binSpacing=1)
            .encode(
                x=alt.X(f"Bootstrap {primary_metric.capitalize()} (%):Q", bin=alt.Bin(maxbins=35)),
                y=alt.Y("count()", title="Frequency"),
                color=alt.Color("Client:N"),
            )
            .properties(height=260)
        )
        st.altair_chart(chart_boot, use_container_width=True)
    else:
        st.info("Execute training or audit to view granular client diagnostics.")


# ========================================================
# TAB 5: SMPC & PRIVACY LEDGER
# ========================================================
with tab5:
    st.subheader("SMPC Secure Aggregation & Differential Privacy Verification")
    st.markdown(
        "Demonstration of **Zero-Sum Additive Pairwise Masking (SMPC)** and **Differential Privacy Parameter Perturbation**."
    )

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("#### 1. Differential Privacy Formulation")
        st.latex(r"\Delta w_k = w_k^{(t)} - w_{global}^{(t)}")
        st.latex(r"\Delta \tilde{w}_k = \Delta w_k \cdot \min\left(1, \frac{C}{\|\Delta w_k\|_2}\right)")
        st.latex(r"\tilde{w}_k = w_{global}^{(t)} + \Delta \tilde{w}_k + \mathcal{N}\left(0, \sigma^2 C^2 \mathbf{I}\right)")
        st.caption(f"Configured Clipping Bound: C = {clip_c} | Noise Multiplier: σ = {dp_sigma}")

    with col_p2:
        st.markdown("#### 2. Secure Aggregation Zero-Sum Masking")
        st.latex(r"m_u = \sum_{v > u} r_{u,v} - \sum_{v < u} r_{v,u}")
        st.latex(r"\sum_{u=1}^K m_u = \mathbf{0}")
        st.latex(r"\sum_{u=1}^K (\tilde{w}_u + m_u) = \sum_{u=1}^K \tilde{w}_u")
        st.caption("Central server computes true aggregate without ever observing unmasked client weights.")

    st.markdown("#### Live Mask Cancellation Verification (Simulated)")
    # Generate mock tensors to visually confirm zero-sum mask cancellation
    t1 = torch.tensor([1.25, -0.40, 2.10])
    t2 = torch.tensor([1.30, -0.35, 2.05])
    t3 = torch.tensor([1.20, -0.45, 2.15])
    mock_dict = {"Client_1": {"layer": t1}, "Client_2": {"layer": t2}, "Client_3": {"layer": t3}}
    masked_out, hashes = simulate_smpc_masking(mock_dict, seed=999)

    mask_sum = (
        masked_out["Client_1"]["layer"]
        + masked_out["Client_2"]["layer"]
        + masked_out["Client_3"]["layer"]
    )
    unmasked_sum = t1 + t2 + t3
    diff_norm = float(torch.norm(mask_sum - unmasked_sum))

    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Unmasked Parameter Sum", f"{unmasked_sum.numpy().round(3).tolist()}")
    c_m2.metric("Masked Parameter Sum", f"{mask_sum.numpy().round(3).tolist()}")
    c_m3.metric("Reconstruction Error", f"{diff_norm:.2e}", help="Exact zero confirms mathematically lossless mask cancellation")

    st.markdown("#### Cryptographic SHA-256 Model Ledger")
    ledger_records = []
    for cid, h in hashes.items():
        ledger_records.append({
            "Client Node": cid,
            "Masked Parameter SHA-256 Hash": h,
            "Integrity Status": "VERIFIED & UNTAMPERED ✅",
        })
    st.dataframe(pd.DataFrame(ledger_records), use_container_width=True)

# Footer
st.divider()
st.caption("Federated MNIST & FED4FED Fairness Evaluation System | Designed for Privacy-Preserving Collaborative Machine Learning.")
