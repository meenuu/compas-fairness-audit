"""
AlgorithmicAuditor
==================
A reusable fairness audit framework for any sklearn binary classifier.

Usage:
    auditor = AlgorithmicAuditor(model, X_test, y_test, sensitive_feature=race_test)
    report  = auditor.audit()
    auditor.print_report()
    auditor.generate_pdf("audit_report.pdf")

Author : You
Version: 0.1.0
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix
)


# ── Optional dependencies (graceful fallback) ──────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from fairlearn.postprocessing import ThresholdOptimizer
    from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
class AlgorithmicAuditor:
    """
    Drop-in fairness auditor for any sklearn binary classifier.

    Parameters
    ----------
    model            : fitted sklearn estimator
    X_test           : pd.DataFrame  — test features
    y_test           : array-like    — true binary labels (0/1)
    sensitive_feature: array-like    — group labels (e.g. race strings)
    feature_names    : list[str]     — optional, defaults to X_test.columns
    model_name       : str           — label for reports
    X_train          : pd.DataFrame  — required only for SHAP + fairness remediation
    y_train          : array-like    — required only for fairness remediation
    sensitive_train  : array-like    — required only for fairness remediation
    annual_n         : int           — estimated annual defendants (for human cost)
    """

    def __init__(
        self,
        model,
        X_test,
        y_test,
        sensitive_feature,
        feature_names=None,
        model_name="Model",
        X_train=None,
        y_train=None,
        sensitive_train=None,
        annual_n=50_000,
    ):
        self.model             = model
        self.X_test            = X_test
        self.y_test            = np.array(y_test)
        self.sensitive         = np.array(sensitive_feature)
        self.feature_names     = feature_names or list(X_test.columns)
        self.model_name        = model_name
        self.X_train           = X_train
        self.y_train           = y_train
        self.sensitive_train   = sensitive_train
        self.annual_n          = annual_n

        self.y_pred            = model.predict(X_test)
        self.y_proba           = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba") else None
        )
        self._report           = None
        self._shap_values      = None
        self._fair_pred        = None

    # ── Core metric helpers ────────────────────────────────────────────────────
    def _group_metrics(self, y_true, y_pred, groups):
        results = {}
        for g in sorted(set(groups)):
            mask = groups == g
            yt, yp = y_true[mask], y_pred[mask]
            n  = mask.sum()
            tp = ((yp == 1) & (yt == 1)).sum()
            tn = ((yp == 0) & (yt == 0)).sum()
            fp = ((yp == 1) & (yt == 0)).sum()
            fn = ((yp == 0) & (yt == 1)).sum()
            results[g] = {
                "n"        : int(n),
                "fpr"      : fp / (fp + tn) if (fp + tn) > 0 else np.nan,
                "fnr"      : fn / (fn + tp) if (fn + tp) > 0 else np.nan,
                "accuracy" : (tp + tn) / n  if n > 0 else np.nan,
                "base_rate": float(yt.mean()),
            }
        return results

    def _human_cost(self, group_metrics):
        """
        Estimate extra false positives per year due to FPR disparity.
        Uses the group with highest FPR as reference vs group with lowest.
        """
        fprs = {g: v["fpr"] for g, v in group_metrics.items() if not np.isnan(v["fpr"])}
        if len(fprs) < 2:
            return {}
        max_g = max(fprs, key=fprs.get)
        min_g = min(fprs, key=fprs.get)
        gap   = fprs[max_g] - fprs[min_g]
        proportion = group_metrics[max_g]["n"] / sum(v["n"] for v in group_metrics.values())
        extra = int(self.annual_n * proportion * gap)
        return {
            "highest_fpr_group" : max_g,
            "lowest_fpr_group"  : min_g,
            "fpr_gap"           : round(gap, 4),
            "extra_flagged_year": extra,
            "extra_flagged_day" : max(1, extra // 365),
        }

    # ── SHAP ──────────────────────────────────────────────────────────────────
    def compute_shap(self, max_samples=500):
        if not SHAP_AVAILABLE:
            print("shap not installed. Run: pip install shap")
            return None
        sample = min(max_samples, len(self.X_test))
        X_sample = self.X_test.iloc[:sample]
        try:
            explainer = shap.TreeExplainer(self.model)
            sv = explainer.shap_values(X_sample)
            # Handle 3D output (n_samples, n_features, n_classes)
            if isinstance(sv, list):
                sv = sv[1]
            elif hasattr(sv, "ndim") and sv.ndim == 3:
                sv = sv[:, :, 1]
            self._shap_values = sv
            self._shap_X      = X_sample
            self._shap_ev     = explainer.expected_value
        except Exception:
            try:
                masker   = shap.maskers.Independent(self.X_test, max_samples=100)
                explainer = shap.LinearExplainer(self.model, masker)
                self._shap_values = explainer.shap_values(X_sample)
                self._shap_X      = X_sample
                self._shap_ev     = explainer.expected_value
            except Exception as e:
                print(f"SHAP failed: {e}")
                return None
        return self._shap_values

    # ── Fairness remediation ───────────────────────────────────────────────────
    def remediate(self, constraints="equalized_odds"):
        if not FAIRLEARN_AVAILABLE:
            print("fairlearn not installed. Run: pip install fairlearn")
            return None
        if self.X_train is None or self.y_train is None or self.sensitive_train is None:
            print("Pass X_train, y_train, sensitive_train to AlgorithmicAuditor to enable remediation.")
            return None
        optimizer = ThresholdOptimizer(
            estimator      = self.model,
            constraints    = constraints,
            predict_method = "predict_proba",
            objective      = "balanced_accuracy_score",
        )
        optimizer.fit(self.X_train, self.y_train, sensitive_features=self.sensitive_train)
        self._fair_pred = optimizer.predict(self.X_test, sensitive_features=self.sensitive)
        self._optimizer = optimizer
        return self._fair_pred

    # ── Main audit method ──────────────────────────────────────────────────────
    def audit(self, compute_shap=True, remediate=True):
        """
        Run the full audit pipeline.
        Returns a dict with all findings.
        """
        # Overall performance
        overall = {
            "accuracy" : round(accuracy_score(self.y_test, self.y_pred), 4),
            "roc_auc"  : round(roc_auc_score(self.y_test, self.y_proba), 4)
                         if self.y_proba is not None else None,
            "n_test"   : len(self.y_test),
        }

        # Fairness by group
        group_metrics = self._group_metrics(self.y_test, self.y_pred, self.sensitive)

        # Equalized odds gap
        eod_gap = None
        if FAIRLEARN_AVAILABLE:
            try:
                eod_gap = round(equalized_odds_difference(
                    self.y_test, self.y_pred, sensitive_features=self.sensitive), 4)
            except Exception:
                pass

        # Human cost
        human_cost = self._human_cost(group_metrics)

        # SHAP
        shap_importance = None
        if compute_shap and SHAP_AVAILABLE:
            sv = self.compute_shap()
            if sv is not None:
                shap_importance = dict(zip(
                    self.feature_names,
                    np.abs(sv).mean(axis=0).round(4)
                ))

        # Remediation
        fair_metrics = None
        fair_eod     = None
        if remediate and FAIRLEARN_AVAILABLE and self.X_train is not None:
            fp = self.remediate()
            if fp is not None:
                fair_metrics = self._group_metrics(self.y_test, fp, self.sensitive)
                try:
                    fair_eod = round(equalized_odds_difference(
                        self.y_test, fp, sensitive_features=self.sensitive), 4)
                except Exception:
                    pass

        self._report = {
            "model_name"     : self.model_name,
            "audit_date"     : datetime.now().strftime("%Y-%m-%d %H:%M"),
            "overall"        : overall,
            "group_metrics"  : group_metrics,
            "eod_gap"        : eod_gap,
            "human_cost"     : human_cost,
            "shap_importance": shap_importance,
            "fair_metrics"   : fair_metrics,
            "fair_eod"       : fair_eod,
        }
        return self._report

    # ── Console report ─────────────────────────────────────────────────────────
    def print_report(self):
        if self._report is None:
            self.audit()
        r = self._report
        w = 60

        print("=" * w)
        print(f"  ALGORITHMIC AUDIT REPORT")
        print(f"  Model : {r['model_name']}")
        print(f"  Date  : {r['audit_date']}")
        print("=" * w)

        print("\n── OVERALL PERFORMANCE ──")
        print(f"  Accuracy : {r['overall']['accuracy']:.4f}")
        if r['overall']['roc_auc']:
            print(f"  ROC-AUC  : {r['overall']['roc_auc']:.4f}")
        print(f"  Test set : {r['overall']['n_test']:,} samples")

        print("\n── FAIRNESS BY GROUP ──")
        df = pd.DataFrame(r['group_metrics']).T
        df.index.name = "Group"
        print(df[["n", "fpr", "fnr", "accuracy", "base_rate"]].round(4).to_string())

        if r['eod_gap'] is not None:
            verdict = "FAIL" if r['eod_gap'] > 0.1 else "PASS"
            print(f"\n  Equalized Odds Gap : {r['eod_gap']:.4f}  [{verdict}]")
            print(f"  (> 0.10 = significant disparity)")

        if r['human_cost']:
            hc = r['human_cost']
            print(f"\n── HUMAN COST ──")
            print(f"  {hc['highest_fpr_group']} defendants have {hc['fpr_gap']:.1%} higher FPR")
            print(f"  than {hc['lowest_fpr_group']} defendants.")
            print(f"  Estimated extra wrongly flagged per year : {hc['extra_flagged_year']:,}")
            print(f"  That is approximately {hc['extra_flagged_day']} people per day.")

        if r['shap_importance']:
            print(f"\n── SHAP FEATURE IMPORTANCE ──")
            for feat, val in sorted(r['shap_importance'].items(),
                                    key=lambda x: -x[1]):
                bar = "█" * int(val * 200)
                print(f"  {feat:<20} {val:.4f}  {bar}")

        if r['fair_metrics'] and r['fair_eod'] is not None:
            print(f"\n── AFTER FAIRNESS REMEDIATION ──")
            df2 = pd.DataFrame(r['fair_metrics']).T
            print(df2[["fpr", "fnr", "accuracy"]].round(4).to_string())
            reduction = (r['eod_gap'] - r['fair_eod']) / r['eod_gap'] * 100
            print(f"\n  EOD gap reduced: {r['eod_gap']:.4f} → {r['fair_eod']:.4f}")
            print(f"  ({reduction:.1f}% improvement)")

        print("\n" + "=" * w)

    # ── PDF report ─────────────────────────────────────────────────────────────
    def generate_pdf(self, path="audit_report.pdf"):
        if self._report is None:
            self.audit()
        r = self._report

        # ── Style ─────────────────────────────────────────────────────────────
        NAVY   = "#1B3A6B"
        BLUE   = "#2563EB"
        RED    = "#DC2626"
        GREEN  = "#16A34A"
        ORANGE = "#EA580C"
        GRAY   = "#64748B"
        LGRAY  = "#F8FAFC"

        plt.rcParams.update({
            "font.family"      : "DejaVu Sans",
            "axes.spines.top"  : False,
            "axes.spines.right": False,
            "figure.facecolor" : "white",
        })

        groups  = list(r["group_metrics"].keys())
        fprs    = [r["group_metrics"][g]["fpr"] for g in groups]
        fnrs    = [r["group_metrics"][g]["fnr"] for g in groups]
        accs    = [r["group_metrics"][g]["accuracy"] for g in groups]

        with PdfPages(path) as pdf:

            # ── PAGE 1: Cover + Summary ────────────────────────────────────────
            fig = plt.figure(figsize=(8.5, 11))
            fig.patch.set_facecolor("white")

            # Header band
            ax_hdr = fig.add_axes([0, 0.88, 1, 0.12])
            ax_hdr.set_facecolor(NAVY)
            ax_hdr.axis("off")
            ax_hdr.text(0.05, 0.65, "ALGORITHMIC AUDIT REPORT",
                        color="white", fontsize=18, fontweight="bold",
                        transform=ax_hdr.transAxes)
            ax_hdr.text(0.05, 0.2, f"{r['model_name']}  ·  {r['audit_date']}",
                        color="#94A3B8", fontsize=10,
                        transform=ax_hdr.transAxes)

            # Summary cards
            cards = [
                ("Accuracy",     f"{r['overall']['accuracy']:.1%}", BLUE),
                ("ROC-AUC",      f"{r['overall']['roc_auc']:.3f}" if r['overall']['roc_auc'] else "N/A", BLUE),
                ("EOD Gap",      f"{r['eod_gap']:.3f}" if r['eod_gap'] else "N/A",
                 RED if (r['eod_gap'] or 0) > 0.1 else GREEN),
                ("Test samples", f"{r['overall']['n_test']:,}", GRAY),
            ]
            for i, (label, val, color) in enumerate(cards):
                x = 0.05 + i * 0.235
                ax = fig.add_axes([x, 0.75, 0.21, 0.10])
                ax.set_facecolor(LGRAY)
                ax.axis("off")
                ax.text(0.5, 0.75, label, ha="center", va="center",
                        fontsize=9, color=GRAY, transform=ax.transAxes)
                ax.text(0.5, 0.25, val, ha="center", va="center",
                        fontsize=16, fontweight="bold", color=color,
                        transform=ax.transAxes)

            # Key findings text
            ax_txt = fig.add_axes([0.05, 0.45, 0.9, 0.27])
            ax_txt.axis("off")
            ax_txt.text(0, 1.0, "Key Findings", fontsize=13,
                        fontweight="bold", color=NAVY, va="top")

            findings = []
            if r["eod_gap"] and r["eod_gap"] > 0.1:
                findings.append(
                    f"FINDING 1 — SIGNIFICANT RACIAL DISPARITY DETECTED\n"
                    f"   Equalized Odds Gap = {r['eod_gap']:.3f} (threshold: 0.10 = fail)\n"
                    f"   The model treats different racial groups unequally."
                )
            if r["human_cost"]:
                hc = r["human_cost"]
                findings.append(
                    f"FINDING 2 — HUMAN COST OF BIAS\n"
                    f"   {hc['highest_fpr_group']} defendants face {hc['fpr_gap']:.1%} higher false positive rate\n"
                    f"   than {hc['lowest_fpr_group']} defendants.\n"
                    f"   Estimated {hc['extra_flagged_year']:,} people wrongly flagged per year\n"
                    f"   ({hc['extra_flagged_day']} per day) due to this disparity."
                )
            if r["fair_eod"] is not None:
                reduction = (r["eod_gap"] - r["fair_eod"]) / r["eod_gap"] * 100
                fair_acc  = accuracy_score(
                    self.y_test, self._fair_pred) if self._fair_pred is not None else None
                acc_cost  = (r["overall"]["accuracy"] - fair_acc) if fair_acc else None
                findings.append(
                    f"FINDING 3 — REMEDIATION TESTED\n"
                    f"   Fairness-constrained retraining reduces EOD gap by {reduction:.1f}%\n"
                    f"   ({r['eod_gap']:.3f} → {r['fair_eod']:.3f}).\n"
                    + (f"   Accuracy cost: {acc_cost:.1%}. Recommended for deployment." if acc_cost else "")
                )
            if r["shap_importance"]:
                top = sorted(r["shap_importance"].items(), key=lambda x: -x[1])[:2]
                findings.append(
                    f"FINDING 4 — TOP PREDICTIVE FEATURES (SHAP)\n"
                    f"   {top[0][0]} (|SHAP|={top[0][1]:.4f}) and "
                    f"{top[1][0]} (|SHAP|={top[1][1]:.4f})\n"
                    f"   are the strongest drivers of predictions."
                )

            y_pos = 0.88
            for f in findings:
                lines = f.split("\n")
                ax_txt.text(0, y_pos, lines[0], fontsize=9, fontweight="bold",
                            color=NAVY, va="top")
                for line in lines[1:]:
                    y_pos -= 0.10
                    ax_txt.text(0.02, y_pos, line, fontsize=9,
                                color="#334155", va="top")
                y_pos -= 0.14

            # Recommendation box
            ax_rec = fig.add_axes([0.05, 0.30, 0.9, 0.12])
            ax_rec.set_facecolor("#FFF7ED")
            ax_rec.axis("off")
            rec_text = (
                "RECOMMENDATION\n"
                "Based on this audit, the model as deployed fails the equalized odds fairness standard.\n"
                "We recommend: (1) deploying the fairness-constrained variant, (2) requiring human review\n"
                "for borderline scores, and (3) replacing re-arrest labels with re-conviction labels."
            )
            ax_rec.text(0.02, 0.85, rec_text, fontsize=8.5, color="#7C2D12",
                        va="top", transform=ax_rec.transAxes, linespacing=1.6)
            ax_rec.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False,
                             edgecolor=ORANGE, linewidth=1.5,
                             transform=ax_rec.transAxes))

            # Footer
            ax_ft = fig.add_axes([0, 0, 1, 0.04])
            ax_ft.set_facecolor(LGRAY)
            ax_ft.axis("off")
            ax_ft.text(0.5, 0.5,
                       f"Generated by AlgorithmicAuditor v0.1.0  ·  Page 1 of 3",
                       ha="center", va="center", fontsize=8, color=GRAY,
                       transform=ax_ft.transAxes)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ── PAGE 2: Fairness Charts ────────────────────────────────────────
            fig2 = plt.figure(figsize=(8.5, 11))
            fig2.patch.set_facecolor("white")

            fig2.text(0.05, 0.96, "Fairness Analysis by Group",
                      fontsize=14, fontweight="bold", color=NAVY)
            fig2.text(0.05, 0.93,
                      "False Positive Rate (FPR): wrongly labelled high-risk  ·  "
                      "False Negative Rate (FNR): missed actual recidivists",
                      fontsize=9, color=GRAY)

            ax1 = fig2.add_axes([0.08, 0.65, 0.85, 0.24])
            x   = np.arange(len(groups))
            w   = 0.35
            colors_fpr = [RED if f == max(fprs) else "#93C5FD" for f in fprs]
            bars = ax1.bar(x, fprs, width=0.6, color=colors_fpr, alpha=0.9)
            ax1.set_xticks(x)
            ax1.set_xticklabels(groups, rotation=25, ha="right", fontsize=9)
            ax1.set_ylabel("False Positive Rate", fontsize=9)
            ax1.set_title("FPR by Race — Who Gets Wrongly Flagged as High-Risk",
                          fontsize=10, color=NAVY)
            ax1.set_ylim(0, 1)
            for bar, val in zip(bars, fprs):
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         val + 0.01, f"{val:.2f}",
                         ha="center", va="bottom", fontsize=8)
            ax1.axhline(np.mean(fprs), color=GRAY, linestyle="--",
                        linewidth=0.8, label=f"Mean = {np.mean(fprs):.2f}")
            ax1.legend(fontsize=8)

            ax2 = fig2.add_axes([0.08, 0.34, 0.85, 0.24])
            ax2.bar(x, fnrs, width=0.6, color="#86EFAC", alpha=0.9)
            ax2.set_xticks(x)
            ax2.set_xticklabels(groups, rotation=25, ha="right", fontsize=9)
            ax2.set_ylabel("False Negative Rate", fontsize=9)
            ax2.set_title("FNR by Race — Who Gets Wrongly Cleared",
                          fontsize=10, color=NAVY)
            ax2.set_ylim(0, 1)
            for i, val in enumerate(fnrs):
                ax2.text(i, val + 0.01, f"{val:.2f}",
                         ha="center", va="bottom", fontsize=8)

            # Before vs after (if remediation available)
            if r["fair_metrics"]:
                fair_fprs = [r["fair_metrics"][g]["fpr"] for g in groups]
                ax3 = fig2.add_axes([0.08, 0.06, 0.85, 0.22])
                ax3.bar(x - w/2, fprs,      width=w, color="coral",
                        alpha=0.85, label="Original model")
                ax3.bar(x + w/2, fair_fprs, width=w, color="steelblue",
                        alpha=0.85, label="Fair model")
                ax3.set_xticks(x)
                ax3.set_xticklabels(groups, rotation=25, ha="right", fontsize=9)
                ax3.set_ylabel("FPR", fontsize=9)
                ax3.set_title("Before vs After Fairness Remediation (FPR)",
                              fontsize=10, color=NAVY)
                ax3.set_ylim(0, 1)
                ax3.legend(fontsize=8)

            fig2.text(0.5, 0.01, "Page 2 of 3", ha="center",
                      fontsize=8, color=GRAY)
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

            # ── PAGE 3: SHAP + Pareto + Human Cost ────────────────────────────
            fig3 = plt.figure(figsize=(8.5, 11))
            fig3.patch.set_facecolor("white")
            fig3.text(0.05, 0.96, "Explainability & Human Cost",
                      fontsize=14, fontweight="bold", color=NAVY)

            # SHAP bar chart
            if r["shap_importance"]:
                ax4 = fig3.add_axes([0.30, 0.72, 0.65, 0.20])
                feats  = list(r["shap_importance"].keys())
                vals   = list(r["shap_importance"].values())
                sorted_pairs = sorted(zip(vals, feats), reverse=True)
                sv, sf = zip(*sorted_pairs)
                ax4.barh(sf, sv, color=BLUE, alpha=0.8)
                ax4.set_xlabel("Mean |SHAP Value|", fontsize=9)
                ax4.set_title("Feature Importance (SHAP)", fontsize=10, color=NAVY)
                ax4.tick_params(labelsize=8)

            # Pareto curve (accuracy vs EOD gap across thresholds)
            if self.y_proba is not None and FAIRLEARN_AVAILABLE:
                try:
                    thresholds   = np.linspace(0.1, 0.9, 25)
                    pareto_acc   = []
                    pareto_gap   = []
                    for t in thresholds:
                        yp = (self.y_proba >= t).astype(int)
                        pareto_acc.append(accuracy_score(self.y_test, yp))
                        try:
                            pareto_gap.append(equalized_odds_difference(
                                self.y_test, yp, sensitive_features=self.sensitive))
                        except Exception:
                            pareto_gap.append(np.nan)

                    valid = [(g, a) for g, a in zip(pareto_gap, pareto_acc)
                             if not np.isnan(g)]
                    if valid:
                        vg, va = zip(*valid)
                        ax5 = fig3.add_axes([0.08, 0.44, 0.85, 0.22])
                        ax5.plot(vg, va, color=BLUE, linewidth=2,
                                 label="Threshold sweep")
                        ax5.scatter([r["eod_gap"]], [r["overall"]["accuracy"]],
                                    color="coral", s=80, zorder=5,
                                    label=f"Original model")
                        if r["fair_eod"] and self._fair_pred is not None:
                            fair_acc = accuracy_score(self.y_test, self._fair_pred)
                            ax5.scatter([r["fair_eod"]], [fair_acc],
                                        color=GREEN, s=80, zorder=5,
                                        label="Fair model")
                        ax5.set_xlabel("Equalized Odds Gap (lower = fairer)", fontsize=9)
                        ax5.set_ylabel("Accuracy", fontsize=9)
                        ax5.set_title("Accuracy vs Fairness — The Pareto Tradeoff",
                                      fontsize=10, color=NAVY)
                        ax5.legend(fontsize=8)
                        ax5.grid(True, alpha=0.2)
                except Exception:
                    pass

            # Human cost panel
            if r["human_cost"]:
                hc  = r["human_cost"]
                ax6 = fig3.add_axes([0.08, 0.22, 0.85, 0.18])
                ax6.axis("off")
                ax6.set_facecolor("#FEF2F2")
                ax6.add_patch(plt.Rectangle((0, 0), 1, 1, fill=True,
                              facecolor="#FEF2F2", edgecolor=RED, linewidth=1))
                ax6.text(0.02, 0.85,
                         "HUMAN COST OF THE FAIRNESS GAP",
                         fontsize=10, fontweight="bold", color=RED, va="top",
                         transform=ax6.transAxes)
                lines = [
                    f"  Group with highest FPR : {hc['highest_fpr_group']}",
                    f"  Group with lowest FPR  : {hc['lowest_fpr_group']}",
                    f"  FPR gap                : {hc['fpr_gap']:.1%}",
                    f"  Extra wrongly flagged  : {hc['extra_flagged_year']:,} per year",
                    f"                         : {hc['extra_flagged_day']} per day",
                    f"  (Assuming {self.annual_n:,} annual defendants at this distribution)",
                ]
                y = 0.65
                for line in lines:
                    ax6.text(0.02, y, line, fontsize=9, color="#7F1D1D",
                             transform=ax6.transAxes)
                    y -= 0.14

            # Methodology note
            ax7 = fig3.add_axes([0.08, 0.06, 0.85, 0.13])
            ax7.axis("off")
            ax7.text(0, 1.0, "Methodology", fontsize=9,
                     fontweight="bold", color=NAVY, va="top")
            method = (
                "This audit uses: (1) group-disaggregated FPR/FNR to measure disparate impact, "
                "(2) Equalized Odds\n"
                "as the primary fairness criterion (Hardt et al., 2016), "
                "(3) SHAP TreeExplainer for feature attribution\n"
                "(Lundberg & Lee, 2017), and (4) ThresholdOptimizer from fairlearn "
                "for post-processing remediation.\n"
                "Human cost estimates assume equal base-rate distribution across "
                f"{self.annual_n:,} annual cases."
            )
            ax7.text(0, 0.78, method, fontsize=8.5, color="#334155",
                     va="top", linespacing=1.5)

            fig3.text(0.5, 0.01, "Page 3 of 3  ·  AlgorithmicAuditor v0.1.0",
                      ha="center", fontsize=8, color=GRAY)

            pdf.savefig(fig3, bbox_inches="tight")
            plt.close(fig3)

        print(f"PDF audit report saved: {path}")
        return path

    # ── Convenience: plot fairness charts inline ───────────────────────────────
    def plot_fairness(self):
        if self._report is None:
            self.audit()
        r      = self._report
        groups = list(r["group_metrics"].keys())
        fprs   = [r["group_metrics"][g]["fpr"] for g in groups]
        fnrs   = [r["group_metrics"][g]["fnr"] for g in groups]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Fairness Audit — FPR & FNR by Group",
                     fontsize=14, fontweight="bold")

        for ax, vals, title in zip(
            axes,
            [fprs, fnrs],
            ["False Positive Rate (wrongly flagged)",
             "False Negative Rate (wrongly cleared)"]
        ):
            colors = ["#DC2626" if v == max(vals) else "#93C5FD" for v in vals]
            bars   = ax.bar(groups, vals, color=colors, alpha=0.9)
            ax.set_xticklabels(groups, rotation=30, ha="right")
            ax.set_ylim(0, 1)
            ax.set_title(title, fontsize=11)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + 0.01, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.show()


# ── Quick-start example ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("AlgorithmicAuditor loaded. Import and use as:")
    print()
    print("  from algorithmic_auditor import AlgorithmicAuditor")
    print()
    print("  auditor = AlgorithmicAuditor(")
    print("      model            = rf,")
    print("      X_test           = X_test,")
    print("      y_test           = y_test,")
    print("      sensitive_feature= race_test,")
    print("      model_name       = 'Random Forest',")
    print("      X_train          = X_train,")
    print("      y_train          = y_train,")
    print("      sensitive_train  = race_train,")
    print("  )")
    print()
    print("  report = auditor.audit()")
    print("  auditor.print_report()")
    print("  auditor.generate_pdf('audit_report.pdf')")
