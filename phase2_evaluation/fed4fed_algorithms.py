"""
FED4FED Statistical Fairness Algorithms:
  - Algorithm 1: Chi-Square Homogeneity Test (Graybill-Deal & Contingency Table)
  - Algorithm 2: Welch's F-test (Heteroscedastic ANOVA for Unequal Variances)
  - Algorithm 3: Federated Confidence Intervals & Pairwise Disparity Inference
"""

from typing import Dict, List, Any, Union
import numpy as np
from scipy import stats


def compute_algorithm1_chisquare(
    client_stats: Dict[str, Dict[str, Any]],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Algorithm 1: Chi-Square Homogeneity Test.
    Combines independent local statistics from K client nodes to test:
        H0: Performance is statistically homogeneous across all clients (theta_1 = ... = theta_K).
        H1: At least two client performance levels differ significantly.

    Computes:
        1. Graybill-Deal Inverse-Variance Weighted Consensus Chi-Square Statistic.
        2. Contingency Table Pearson Chi-Square on (Correct, Incorrect) outcomes (if counts available).

    Args:
        client_stats: Dict mapping client_id to summary dict with keys:
                      'mean', 'variance', 'n_samples', and optionally 'correct_count', 'incorrect_count'.
        alpha: Significance level (default 0.05).

    Returns:
        Dictionary of consensus estimates, Chi-square statistics, p-values, and verdicts.
    """
    nodes = sorted(list(client_stats.keys()))
    K = len(nodes)
    if K < 2:
        raise ValueError(f"At least 2 clients required for Chi-Square homogeneity test (found {K}).")

    means = np.array([client_stats[k]["mean"] for k in nodes], dtype=np.float64)
    variances = np.array([max(client_stats[k].get("variance", 1e-6), 1e-8) for k in nodes], dtype=np.float64)
    n_samples = np.array([max(client_stats[k].get("n_samples", 100), 2) for k in nodes], dtype=np.float64)

    # 1. Graybill-Deal Weights: w_k = (1 / s_k^2) / sum_j (1 / s_j^2)
    inv_vars = 1.0 / variances
    sum_inv_vars = np.sum(inv_vars)
    weights = inv_vars / sum_inv_vars

    # Consensus estimate
    theta_gd = float(np.sum(weights * means))
    var_theta_gd = float(1.0 / sum_inv_vars)
    se_theta_gd = float(np.sqrt(var_theta_gd))

    # Inverse-variance Chi-Square test statistic:
    # chi2 = sum_{k=1}^K ((theta_k - theta_gd)^2 / s_k^2)
    chi2_stat = float(np.sum(((means - theta_gd) ** 2) / variances))
    df = K - 1

    p_val_gd = float(stats.chi2.sf(chi2_stat, df=df))
    crit_val = float(stats.chi2.ppf(1.0 - alpha, df=df))
    reject_gd = bool(p_val_gd < alpha)

    # 2. Contingency Table Chi-Square Test (Correct vs Incorrect)
    contingency_table = []
    has_counts = True
    for k in nodes:
        if "correct_count" in client_stats[k] and "incorrect_count" in client_stats[k]:
            contingency_table.append([client_stats[k]["correct_count"], client_stats[k]["incorrect_count"]])
        else:
            has_counts = False
            break

    contingency_res = {}
    if has_counts:
        table_arr = np.array(contingency_table)
        chi2_c, p_val_c, df_c, _ = stats.chi2_contingency(table_arr)
        contingency_res = {
            "chi2_contingency": float(chi2_c),
            "p_value_contingency": float(p_val_c),
            "df_contingency": int(df_c),
            "reject_contingency": bool(p_val_c < alpha),
        }

    # Primary verdict: Disparity detected if either test rejects H0
    disparity_detected = reject_gd or contingency_res.get("reject_contingency", False)
    verdict = "DISPARITY DETECTED (Heterogeneous)" if disparity_detected else "FAIR (Homogeneous)"

    per_node_breakdown = {}
    for i, node in enumerate(nodes):
        per_node_breakdown[node] = {
            "mean": float(means[i]),
            "variance": float(variances[i]),
            "weight": float(weights[i]),
            "chi2_contribution": float(((means[i] - theta_gd) ** 2) / variances[i]),
        }

    return {
        "algorithm": "Algorithm 1: Chi-Square Homogeneity Test",
        "K_nodes": K,
        "graybill_deal_mean": theta_gd,
        "graybill_deal_variance": var_theta_gd,
        "graybill_deal_std_error": se_theta_gd,
        "chi2_statistic": chi2_stat,
        "degrees_of_freedom": df,
        "p_value": p_val_gd,
        "critical_value": crit_val,
        "significance_level": alpha,
        "reject_null_hypothesis": disparity_detected,
        "verdict": verdict,
        "node_weights": {nodes[i]: float(weights[i]) for i in range(K)},
        "contingency_test": contingency_res,
        "per_node_breakdown": per_node_breakdown,
    }


def compute_algorithm2_welch_f(
    client_stats: Dict[str, Dict[str, Any]],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Algorithm 2: Welch's F-test (Welch's ANOVA for Unequal Variances).
    Tests equality of means across K client nodes without assuming equal variance (heteroscedasticity).
        H0: mu_1 = mu_2 = ... = mu_K
        H1: At least one group mean differs significantly.

    Formulas:
        w_k = n_k / s_k^2,  W = sum(w_k)
        x'_bar = sum(w_k * mean_k) / W
        MS_between = (1 / (K - 1)) * sum(w_k * (mean_k - x'_bar)^2)
        Lambda = sum( (1 / (n_k - 1)) * (1 - w_k / W)^2 )
        Denominator = 1 + (2 * (K - 2) / (K^2 - 1)) * Lambda
        F = MS_between / Denominator
        df1 = K - 1, df2 = (K^2 - 1) / (3 * Lambda)

    Args:
        client_stats: Dict mapping client_id to summary dict with keys 'mean', 'variance', 'n_samples'.
        alpha: Significance level (default 0.05).

    Returns:
        Dictionary containing Welch's F-statistic, df1, df2, p-value, and fairness decision.
    """
    nodes = sorted(list(client_stats.keys()))
    K = len(nodes)
    if K < 2:
        raise ValueError(f"At least 2 groups required for Welch's F-test (found {K}).")

    means = np.array([client_stats[k]["mean"] for k in nodes], dtype=np.float64)
    variances = np.array([max(client_stats[k].get("variance", 1e-6), 1e-8) for k in nodes], dtype=np.float64)
    n_samples = np.array([max(client_stats[k].get("n_samples", 100), 2) for k in nodes], dtype=np.float64)

    # Weights: w_k = n_k / s_k^2
    w = n_samples / variances
    W = np.sum(w)

    # Weighted grand mean
    x_prime = np.sum(w * means) / W

    # Numerator Mean Square
    ms_between = (1.0 / (K - 1.0)) * np.sum(w * ((means - x_prime) ** 2))

    # Lambda adjustment term
    lambda_term = np.sum((1.0 / (n_samples - 1.0)) * ((1.0 - (w / W)) ** 2))

    # Welch's F denominator
    denom = 1.0 + (2.0 * (K - 2.0) / (K ** 2 - 1.0)) * lambda_term
    f_stat = float(ms_between / max(denom, 1e-12))

    # Degrees of freedom
    df1 = K - 1
    df2 = float((K ** 2 - 1.0) / (3.0 * max(lambda_term, 1e-12)))

    # P-value from F-distribution survival function
    p_value = float(stats.f.sf(f_stat, dfn=df1, dfd=df2))
    crit_f = float(stats.f.ppf(1.0 - alpha, dfn=df1, dfd=df2))

    reject_h0 = bool(p_value < alpha)
    verdict = "DISPARITY DETECTED (Unequal Performance)" if reject_h0 else "FAIR (Equivalent Means)"

    return {
        "algorithm": "Algorithm 2: Welch's F-test (Heteroscedastic ANOVA)",
        "K_groups": K,
        "f_statistic": f_stat,
        "df1": df1,
        "df2": round(df2, 2),
        "p_value": p_value,
        "critical_f": crit_f,
        "significance_level": alpha,
        "weighted_grand_mean": float(x_prime),
        "reject_null_hypothesis": reject_h0,
        "verdict": verdict,
    }


def compute_algorithm3_confidence_intervals(
    client_stats: Dict[str, Dict[str, Any]],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Algorithm 3: Federated Confidence Intervals & Pairwise Disparity Inference.
    Computes:
        1. Local (1 - alpha) Student's t / empirical confidence intervals per client.
        2. Graybill-Deal pooled consensus confidence interval with Welch-Satterthwaite effective df.
        3. Pairwise subgroup disparity comparisons with Bonferroni multi-testing correction.
           If 0 is NOT included in [CI_diff_lower, CI_diff_upper], significant disparity exists.

    Args:
        client_stats: Dict mapping client_id to summary dict with keys 'mean', 'variance', 'n_samples'.
        alpha: Significance level (default 0.05).

    Returns:
        Dictionary of local intervals, consensus interval, and pairwise disparity tests.
    """
    nodes = sorted(list(client_stats.keys()))
    K = len(nodes)
    means = np.array([client_stats[k]["mean"] for k in nodes], dtype=np.float64)
    variances = np.array([max(client_stats[k].get("variance", 1e-6), 1e-8) for k in nodes], dtype=np.float64)
    n_samples = np.array([max(client_stats[k].get("n_samples", 100), 2) for k in nodes], dtype=np.float64)

    # 1. Local Node Confidence Intervals
    local_intervals = {}
    for i, node in enumerate(nodes):
        m = means[i]
        v = variances[i]
        n = n_samples[i]
        se = np.sqrt(v / n)

        # Prefer bootstrap percentile bounds if present
        if "ci_lower" in client_stats[node] and "ci_upper" in client_stats[node]:
            lower = client_stats[node]["ci_lower"]
            upper = client_stats[node]["ci_upper"]
            margin = (upper - lower) / 2.0
        else:
            t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
            lower = float(m - t_crit * se)
            upper = float(m + t_crit * se)
            margin = float(t_crit * se)

        local_intervals[node] = {
            "mean": float(m),
            "std_error": float(se),
            "df": int(n - 1),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "margin_of_error": float(margin),
        }

    # 2. Graybill-Deal Consensus Interval
    inv_vars = 1.0 / variances
    sum_inv_vars = np.sum(inv_vars)
    theta_gd = float(np.sum((inv_vars / sum_inv_vars) * means))
    se_gd = float(np.sqrt(1.0 / sum_inv_vars))

    denom_df = np.sum(1.0 / ((n_samples - 1.0) * (variances ** 2)))
    df_eff = float((sum_inv_vars ** 2) / max(denom_df, 1e-12))
    df_eff = max(1.0, df_eff)

    t_crit_gd = float(stats.t.ppf(1.0 - alpha / 2.0, df=df_eff))
    ci_lower_gd = float(theta_gd - t_crit_gd * se_gd)
    ci_upper_gd = float(theta_gd + t_crit_gd * se_gd)

    consensus_interval = {
        "graybill_deal_mean": theta_gd,
        "std_error": se_gd,
        "effective_df": round(df_eff, 2),
        "ci_lower": ci_lower_gd,
        "ci_upper": ci_upper_gd,
        "margin_of_error": float(t_crit_gd * se_gd),
    }

    # 3. Pairwise Subgroup Disparity Tests with Bonferroni correction
    pairwise_tests = []
    num_comparisons = max(1, (K * (K - 1)) // 2)
    alpha_bonf = alpha / num_comparisons

    any_disparity = False
    for i in range(K):
        for j in range(i + 1, K):
            diff = float(means[i] - means[j])
            se_diff = float(np.sqrt((variances[i] / n_samples[i]) + (variances[j] / n_samples[j])))
            
            # Welch-Satterthwaite degrees of freedom for the pair
            v_i_n = variances[i] / n_samples[i]
            v_j_n = variances[j] / n_samples[j]
            pair_df = float(
                ((v_i_n + v_j_n) ** 2)
                / max(1e-12, (((v_i_n ** 2) / (n_samples[i] - 1.0)) + ((v_j_n ** 2) / (n_samples[j] - 1.0))))
            )
            t_crit_pair = float(stats.t.ppf(1.0 - alpha_bonf / 2.0, df=max(1.0, pair_df)))
            ci_diff_lower = float(diff - t_crit_pair * se_diff)
            ci_diff_upper = float(diff + t_crit_pair * se_diff)

            # Significant disparity if 0 is NOT included in the confidence interval
            is_disparate = bool(ci_diff_lower > 0 or ci_diff_upper < 0)
            if is_disparate:
                any_disparity = True

            t_stat = float(diff / max(se_diff, 1e-12))
            p_val = float(2.0 * stats.t.sf(abs(t_stat), df=max(1.0, pair_df)))

            pairwise_tests.append({
                "comparison": f"{nodes[i]} vs {nodes[j]}",
                "client_a": nodes[i],
                "client_b": nodes[j],
                "mean_difference": diff,
                "std_error": se_diff,
                "adjusted_alpha": alpha_bonf,
                "ci_lower": ci_diff_lower,
                "ci_upper": ci_diff_upper,
                "p_value": p_val,
                "significant_disparity": is_disparate,
                "verdict": "SIGNIFICANT DISPARITY" if is_disparate else "FAIR (Within CI)",
            })

    return {
        "algorithm": "Algorithm 3: Federated Confidence Intervals & Pairwise Disparity",
        "alpha": alpha,
        "bonferroni_alpha": alpha_bonf,
        "local_intervals": local_intervals,
        "consensus_interval": consensus_interval,
        "pairwise_tests": pairwise_tests,
        "any_disparity_detected": any_disparity,
        "verdict": "DISPARITIES DETECTED" if any_disparity else "FAIR / HOMOGENEOUS",
    }
