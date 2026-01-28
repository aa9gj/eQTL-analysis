"""Results handling and visualization for eQTL analysis.

This module provides:
- Loading and parsing of tensorQTL results
- Multiple testing correction
- Result filtering and annotation
- Visualization of QTL results
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from eqtl_analysis.utils.io import ensure_directory
from eqtl_analysis.utils.logging import get_logger
from eqtl_analysis.utils.validators import validate_file_exists

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class EQTLSummary:
    """Summary statistics for eQTL results."""

    total_associations: int
    significant_associations: int
    genes_tested: int
    genes_with_eqtl: int
    variants_tested: int
    lead_variants: int
    fdr_threshold: float


class EQTLResults:
    """Handler for eQTL analysis results."""

    def __init__(
        self,
        results_file: str | Path | None = None,
        fdr_threshold: float = 0.05,
        output_dir: str | Path = "results/summary",
    ) -> None:
        """
        Initialize results handler.

        Args:
            results_file: Path to results file.
            fdr_threshold: FDR threshold for significance.
            output_dir: Output directory for processed results.
        """
        self.fdr_threshold = fdr_threshold
        self.output_dir = ensure_directory(output_dir)
        self._results: pd.DataFrame | None = None
        self._summary: EQTLSummary | None = None

        if results_file:
            self.load(results_file)

    @property
    def results(self) -> pd.DataFrame | None:
        """Get loaded results."""
        return self._results

    @property
    def summary(self) -> EQTLSummary | None:
        """Get results summary."""
        return self._summary

    def load(
        self,
        results_file: str | Path,
        file_format: Literal["auto", "parquet", "tsv", "csv"] = "auto",
    ) -> pd.DataFrame:
        """
        Load eQTL results from file.

        Args:
            results_file: Path to results file.
            file_format: File format.

        Returns:
            Results DataFrame.
        """
        results_file = Path(results_file)
        validate_file_exists(results_file, "Results file")

        # Auto-detect format
        if file_format == "auto":
            suffix = results_file.suffix.lower()
            if suffix == ".parquet":
                file_format = "parquet"
            elif suffix in (".tsv", ".txt") or str(results_file).endswith(".txt.gz"):
                file_format = "tsv"
            elif suffix == ".csv":
                file_format = "csv"
            else:
                file_format = "tsv"

        # Load data
        if file_format == "parquet":
            self._results = pd.read_parquet(results_file)
        elif file_format == "tsv":
            self._results = pd.read_csv(results_file, sep="\t")
        else:
            self._results = pd.read_csv(results_file)

        logger.info(f"Loaded {len(self._results)} results from {results_file}")

        # Calculate summary
        self._calculate_summary()

        return self._results

    def _calculate_summary(self) -> None:
        """Calculate summary statistics."""
        if self._results is None:
            return

        df = self._results

        # Determine significance column
        sig_col = None
        for col in ["qval", "q_value", "fdr", "adj_pval"]:
            if col in df.columns:
                sig_col = col
                break

        n_significant = 0
        n_genes_with_eqtl = 0

        if sig_col:
            sig_mask = df[sig_col] <= self.fdr_threshold
            n_significant = sig_mask.sum()
            if "phenotype_id" in df.columns:
                n_genes_with_eqtl = df.loc[sig_mask, "phenotype_id"].nunique()

        # Count genes and variants
        n_genes = df["phenotype_id"].nunique() if "phenotype_id" in df.columns else 0
        n_variants = df["variant_id"].nunique() if "variant_id" in df.columns else 0

        # Count lead variants (best per gene)
        n_lead = 0
        if "phenotype_id" in df.columns and sig_col:
            lead_df = self.get_lead_variants()
            n_lead = len(lead_df)

        self._summary = EQTLSummary(
            total_associations=len(df),
            significant_associations=n_significant,
            genes_tested=n_genes,
            genes_with_eqtl=n_genes_with_eqtl,
            variants_tested=n_variants,
            lead_variants=n_lead,
            fdr_threshold=self.fdr_threshold,
        )

    def apply_fdr_correction(
        self,
        method: Literal["bh", "bonferroni", "storey"] = "bh",
        pval_column: str = "pval_nominal",
    ) -> pd.DataFrame:
        """
        Apply multiple testing correction.

        Args:
            method: Correction method.
            pval_column: Column containing p-values.

        Returns:
            Results with added q-value column.
        """
        if self._results is None:
            raise ValueError("No results loaded")

        from scipy import stats

        pvals = self._results[pval_column].values

        if method == "bh":
            # Benjamini-Hochberg
            from statsmodels.stats.multitest import multipletests

            _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
        elif method == "bonferroni":
            qvals = np.minimum(pvals * len(pvals), 1.0)
        elif method == "storey":
            qvals = self._storey_qvalue(pvals)
        else:
            raise ValueError(f"Unknown correction method: {method}")

        self._results["qval"] = qvals

        n_sig = (qvals <= self.fdr_threshold).sum()
        logger.info(
            f"Applied {method} correction: {n_sig} significant at FDR {self.fdr_threshold}"
        )

        self._calculate_summary()
        return self._results

    def _storey_qvalue(self, pvals: np.ndarray) -> np.ndarray:
        """Calculate Storey q-values."""
        # Estimate pi0 (proportion of true nulls)
        lambdas = np.arange(0.05, 0.95, 0.05)
        pi0_est = []

        for lam in lambdas:
            pi0_est.append(np.mean(pvals > lam) / (1 - lam))

        # Use smoothed estimate
        pi0 = min(np.median(pi0_est), 1.0)

        # Calculate q-values
        n = len(pvals)
        sorted_idx = np.argsort(pvals)
        sorted_pvals = pvals[sorted_idx]

        qvals = np.zeros(n)
        qvals[-1] = pi0 * sorted_pvals[-1]

        for i in range(n - 2, -1, -1):
            qvals[i] = min(
                pi0 * n * sorted_pvals[i] / (i + 1),
                qvals[i + 1],
            )

        # Reorder to original
        qvals_original = np.zeros(n)
        qvals_original[sorted_idx] = qvals

        return qvals_original

    def get_significant(
        self,
        threshold: float | None = None,
    ) -> pd.DataFrame:
        """
        Get significant associations.

        Args:
            threshold: FDR threshold. Uses default if None.

        Returns:
            Significant associations.
        """
        if self._results is None:
            raise ValueError("No results loaded")

        threshold = threshold or self.fdr_threshold

        # Find q-value column
        for col in ["qval", "q_value", "fdr", "adj_pval"]:
            if col in self._results.columns:
                return self._results[self._results[col] <= threshold].copy()

        logger.warning("No q-value column found, returning empty DataFrame")
        return pd.DataFrame()

    def get_lead_variants(
        self,
        threshold: float | None = None,
    ) -> pd.DataFrame:
        """
        Get lead (best) variant per gene.

        Args:
            threshold: FDR threshold.

        Returns:
            Lead variants DataFrame.
        """
        sig_results = self.get_significant(threshold)

        if len(sig_results) == 0:
            return pd.DataFrame()

        # Find p-value column
        pval_col = None
        for col in ["pval_nominal", "pval", "p_value", "pval_beta"]:
            if col in sig_results.columns:
                pval_col = col
                break

        if pval_col is None:
            logger.warning("No p-value column found")
            return sig_results.drop_duplicates("phenotype_id")

        # Get best variant per gene
        lead = sig_results.loc[
            sig_results.groupby("phenotype_id")[pval_col].idxmin()
        ]

        return lead

    def annotate_results(
        self,
        gene_annotation: pd.DataFrame | None = None,
        variant_annotation: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Annotate results with gene and variant information.

        Args:
            gene_annotation: Gene annotation DataFrame.
            variant_annotation: Variant annotation DataFrame.

        Returns:
            Annotated results.
        """
        if self._results is None:
            raise ValueError("No results loaded")

        annotated = self._results.copy()

        # Annotate genes
        if gene_annotation is not None and "phenotype_id" in annotated.columns:
            gene_cols = [c for c in gene_annotation.columns
                        if c not in annotated.columns]
            if gene_cols:
                annotated = annotated.merge(
                    gene_annotation[gene_cols],
                    left_on="phenotype_id",
                    right_index=True,
                    how="left",
                )
                logger.info(f"Added gene annotations: {gene_cols}")

        # Annotate variants
        if variant_annotation is not None and "variant_id" in annotated.columns:
            var_cols = [c for c in variant_annotation.columns
                       if c not in annotated.columns]
            if var_cols:
                annotated = annotated.merge(
                    variant_annotation[var_cols],
                    left_on="variant_id",
                    right_index=True,
                    how="left",
                )
                logger.info(f"Added variant annotations: {var_cols}")

        self._results = annotated
        return annotated

    def save(
        self,
        output_path: str | Path | None = None,
        file_format: Literal["parquet", "tsv", "csv"] = "tsv",
        significant_only: bool = False,
    ) -> Path:
        """
        Save results to file.

        Args:
            output_path: Output file path.
            file_format: Output format.
            significant_only: Only save significant results.

        Returns:
            Path to saved file.
        """
        if self._results is None:
            raise ValueError("No results loaded")

        if output_path is None:
            suffix = ".parquet" if file_format == "parquet" else f".{file_format}"
            output_path = self.output_dir / f"eqtl_results{suffix}"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get data to save
        df = self.get_significant() if significant_only else self._results

        # Save
        if file_format == "parquet":
            df.to_parquet(output_path)
        elif file_format == "tsv":
            df.to_csv(output_path, sep="\t", index=False)
        else:
            df.to_csv(output_path, index=False)

        logger.info(f"Saved {len(df)} results to {output_path}")
        return output_path

    def plot_manhattan(
        self,
        output_path: str | Path | None = None,
        title: str = "eQTL Manhattan Plot",
        figsize: tuple[int, int] = (14, 6),
    ) -> Path | None:
        """
        Create Manhattan plot of results.

        Args:
            output_path: Output file path.
            title: Plot title.
            figsize: Figure size.

        Returns:
            Path to saved plot.
        """
        if self._results is None:
            raise ValueError("No results loaded")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None

        # Get p-value column
        pval_col = None
        for col in ["pval_nominal", "pval", "p_value"]:
            if col in self._results.columns:
                pval_col = col
                break

        if pval_col is None:
            logger.warning("No p-value column found for Manhattan plot")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        # Prepare data
        df = self._results.copy()
        df["-log10p"] = -np.log10(df[pval_col].clip(lower=1e-300))

        # Color by chromosome if available
        if "variant_chr" in df.columns or "chr" in df.columns:
            chr_col = "variant_chr" if "variant_chr" in df.columns else "chr"
            colors = plt.cm.tab20(np.arange(20))

            for i, chrom in enumerate(df[chr_col].unique()):
                mask = df[chr_col] == chrom
                ax.scatter(
                    range(mask.sum()),
                    df.loc[mask, "-log10p"],
                    c=[colors[i % 20]],
                    s=2,
                    alpha=0.5,
                    label=str(chrom) if i < 10 else None,
                )
        else:
            ax.scatter(
                range(len(df)),
                df["-log10p"],
                s=2,
                alpha=0.5,
            )

        # Significance line
        sig_threshold = -np.log10(0.05 / len(df))  # Bonferroni
        ax.axhline(y=sig_threshold, color="r", linestyle="--", alpha=0.5)

        ax.set_xlabel("Variant")
        ax.set_ylabel("-log10(p-value)")
        ax.set_title(title)

        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / "manhattan_plot.png"

        output_path = Path(output_path)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved Manhattan plot to {output_path}")
        return output_path

    def plot_qq(
        self,
        output_path: str | Path | None = None,
        title: str = "eQTL Q-Q Plot",
        figsize: tuple[int, int] = (6, 6),
    ) -> Path | None:
        """
        Create Q-Q plot of p-values.

        Args:
            output_path: Output file path.
            title: Plot title.
            figsize: Figure size.

        Returns:
            Path to saved plot.
        """
        if self._results is None:
            raise ValueError("No results loaded")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None

        # Get p-value column
        pval_col = None
        for col in ["pval_nominal", "pval", "p_value"]:
            if col in self._results.columns:
                pval_col = col
                break

        if pval_col is None:
            logger.warning("No p-value column found for Q-Q plot")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        # Calculate expected vs observed
        pvals = self._results[pval_col].dropna().values
        n = len(pvals)
        expected = -np.log10((np.arange(1, n + 1)) / (n + 1))
        observed = -np.log10(np.sort(pvals))

        # Plot
        ax.scatter(expected, observed, s=2, alpha=0.5)
        ax.plot([0, max(expected)], [0, max(expected)], "r--", alpha=0.5)

        ax.set_xlabel("Expected -log10(p)")
        ax.set_ylabel("Observed -log10(p)")
        ax.set_title(title)

        # Calculate genomic inflation factor
        from scipy import stats

        chi2_obs = stats.chi2.ppf(1 - pvals, df=1)
        lambda_gc = np.median(chi2_obs) / stats.chi2.ppf(0.5, df=1)
        ax.text(0.05, 0.95, f"$\\lambda_{{GC}}$ = {lambda_gc:.3f}",
               transform=ax.transAxes, fontsize=10)

        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / "qq_plot.png"

        output_path = Path(output_path)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved Q-Q plot to {output_path}")
        return output_path

    def generate_report(
        self,
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Generate summary report of eQTL results.

        Args:
            output_path: Output file path.

        Returns:
            Path to report file.
        """
        if self._summary is None:
            raise ValueError("No results loaded")

        if output_path is None:
            output_path = self.output_dir / "eqtl_report.txt"

        output_path = Path(output_path)

        with open(output_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("eQTL Analysis Report\n")
            f.write("=" * 60 + "\n\n")

            f.write("Summary Statistics\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total associations tested: {self._summary.total_associations:,}\n")
            f.write(f"Significant associations: {self._summary.significant_associations:,}\n")
            f.write(f"Genes tested: {self._summary.genes_tested:,}\n")
            f.write(f"Genes with eQTL: {self._summary.genes_with_eqtl:,}\n")
            f.write(f"Variants tested: {self._summary.variants_tested:,}\n")
            f.write(f"Lead variants: {self._summary.lead_variants:,}\n")
            f.write(f"FDR threshold: {self._summary.fdr_threshold}\n")

            if self._summary.genes_tested > 0:
                pct = (self._summary.genes_with_eqtl / self._summary.genes_tested) * 100
                f.write(f"\nPercentage of genes with eQTL: {pct:.1f}%\n")

        logger.info(f"Generated report: {output_path}")
        return output_path
