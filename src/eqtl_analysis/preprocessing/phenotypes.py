"""Phenotype (gene expression) preprocessing module for eQTL analysis.

This module handles:
- Reading and normalizing gene expression data
- Filtering lowly expressed genes
- Sample QC and outlier detection
- Normalization and transformation
- Preparing expression data for tensorQTL
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy import stats

from eqtl_analysis.utils.io import (
    ensure_directory,
    read_expression_matrix,
    write_expression_matrix,
)
from eqtl_analysis.utils.logging import get_logger
from eqtl_analysis.utils.validators import (
    validate_expression_matrix,
    validate_file_exists,
)

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class PhenotypeQCStats:
    """Statistics from phenotype quality control."""

    total_genes: int
    total_samples: int
    genes_after_expression_filter: int
    genes_after_variance_filter: int
    samples_after_outlier_filter: int
    final_genes: int
    final_samples: int


class PhenotypePreprocessor:
    """Preprocessor for gene expression (phenotype) data in eQTL analysis."""

    def __init__(
        self,
        min_expression: float = 1.0,
        min_samples_expressed: float = 0.2,
        min_variance: float = 0.01,
        normalization_method: Literal["tmm", "quantile", "inverse_normal"] = "inverse_normal",
        log_transform: bool = True,
        output_dir: str | Path = "results/phenotypes",
    ) -> None:
        """
        Initialize phenotype preprocessor.

        Args:
            min_expression: Minimum expression threshold (TPM/FPKM/counts).
            min_samples_expressed: Minimum fraction of samples with expression.
            min_variance: Minimum variance threshold.
            normalization_method: Normalization method to apply.
            log_transform: Whether to log2-transform data.
            output_dir: Output directory for processed files.
        """
        self.min_expression = min_expression
        self.min_samples_expressed = min_samples_expressed
        self.min_variance = min_variance
        self.normalization_method = normalization_method
        self.log_transform = log_transform
        self.output_dir = ensure_directory(output_dir)
        self._qc_stats: PhenotypeQCStats | None = None

    @property
    def qc_stats(self) -> PhenotypeQCStats | None:
        """Get QC statistics from last run."""
        return self._qc_stats

    def load_expression_data(
        self,
        file_path: str | Path,
        gene_column: str | None = None,
        sample_mapping: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """
        Load expression data from file.

        Args:
            file_path: Path to expression file.
            gene_column: Column name for gene IDs.
            sample_mapping: Optional sample ID mapping.

        Returns:
            Expression matrix (genes x samples).
        """
        df = read_expression_matrix(file_path, gene_column=gene_column)

        # Apply sample mapping if provided
        if sample_mapping:
            df.columns = [sample_mapping.get(c, c) for c in df.columns]

        # Validate
        validation = validate_expression_matrix(df)
        if not validation["valid"]:
            logger.warning(f"Expression data issues: {validation['issues']}")

        logger.info(f"Loaded expression data: {df.shape[0]} genes x {df.shape[1]} samples")
        return df

    def filter_lowly_expressed(
        self,
        expression: pd.DataFrame,
        min_expression: float | None = None,
        min_samples: float | None = None,
    ) -> pd.DataFrame:
        """
        Filter genes with low expression.

        Args:
            expression: Expression matrix (genes x samples).
            min_expression: Minimum expression threshold.
            min_samples: Minimum fraction of samples expressed.

        Returns:
            Filtered expression matrix.
        """
        min_expression = min_expression or self.min_expression
        min_samples = min_samples or self.min_samples_expressed

        n_samples_required = int(expression.shape[1] * min_samples)

        # Count samples with expression above threshold
        n_expressed = (expression >= min_expression).sum(axis=1)
        mask = n_expressed >= n_samples_required

        filtered = expression[mask]

        logger.info(
            f"Expression filter: {len(expression)} -> {len(filtered)} genes "
            f"(min expr: {min_expression}, min samples: {n_samples_required})"
        )

        return filtered

    def filter_low_variance(
        self,
        expression: pd.DataFrame,
        min_variance: float | None = None,
    ) -> pd.DataFrame:
        """
        Filter genes with low variance.

        Args:
            expression: Expression matrix.
            min_variance: Minimum variance threshold.

        Returns:
            Filtered expression matrix.
        """
        min_variance = min_variance or self.min_variance

        variances = expression.var(axis=1)
        mask = variances >= min_variance

        filtered = expression[mask]

        logger.info(
            f"Variance filter: {len(expression)} -> {len(filtered)} genes "
            f"(min var: {min_variance})"
        )

        return filtered

    def detect_outlier_samples(
        self,
        expression: pd.DataFrame,
        method: Literal["pca", "correlation", "zscore"] = "correlation",
        threshold: float = 3.0,
    ) -> list[str]:
        """
        Detect outlier samples in expression data.

        Args:
            expression: Expression matrix.
            method: Outlier detection method.
            threshold: Threshold for outlier detection.

        Returns:
            List of outlier sample IDs.
        """
        outliers = []

        if method == "correlation":
            # Compute sample correlations
            corr_matrix = expression.corr()
            mean_corr = corr_matrix.mean()
            median_corr = mean_corr.median()
            mad = np.median(np.abs(mean_corr - median_corr))

            # Z-score based on median absolute deviation
            z_scores = (mean_corr - median_corr) / (mad * 1.4826)
            outliers = z_scores[z_scores.abs() > threshold].index.tolist()

        elif method == "pca":
            # PCA-based outlier detection
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2)
            expr_scaled = (expression - expression.mean()) / expression.std()
            pca_result = pca.fit_transform(expr_scaled.T)

            # Mahalanobis-like distance
            center = pca_result.mean(axis=0)
            distances = np.sqrt(((pca_result - center) ** 2).sum(axis=1))
            z_scores = (distances - distances.mean()) / distances.std()

            outlier_mask = np.abs(z_scores) > threshold
            outliers = [expression.columns[i] for i, is_outlier in enumerate(outlier_mask)
                       if is_outlier]

        elif method == "zscore":
            # Simple z-score on total expression
            total_expr = expression.sum()
            z_scores = (total_expr - total_expr.mean()) / total_expr.std()
            outliers = z_scores[z_scores.abs() > threshold].index.tolist()

        if outliers:
            logger.warning(f"Detected {len(outliers)} outlier samples: {outliers}")

        return outliers

    def normalize(
        self,
        expression: pd.DataFrame,
        method: str | None = None,
    ) -> pd.DataFrame:
        """
        Normalize expression data.

        Args:
            expression: Expression matrix.
            method: Normalization method.

        Returns:
            Normalized expression matrix.
        """
        method = method or self.normalization_method

        if method == "tmm":
            return self._normalize_tmm(expression)
        elif method == "quantile":
            return self._normalize_quantile(expression)
        elif method == "inverse_normal":
            return self._normalize_inverse_normal(expression)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def _normalize_tmm(self, expression: pd.DataFrame) -> pd.DataFrame:
        """TMM (Trimmed Mean of M-values) normalization."""
        # Simple TMM-like normalization
        # Calculate size factors
        log_expr = np.log2(expression + 1)
        geo_mean = log_expr.mean(axis=1)
        log_ratios = log_expr.subtract(geo_mean, axis=0)

        # Trim extreme values
        trimmed = log_ratios.apply(lambda x: x[(x > x.quantile(0.05)) &
                                                (x < x.quantile(0.95))])

        # Size factors
        size_factors = 2 ** trimmed.mean()

        # Normalize
        normalized = expression.divide(size_factors, axis=1)
        normalized = normalized * normalized.median().median()

        logger.info("Applied TMM normalization")
        return normalized

    def _normalize_quantile(self, expression: pd.DataFrame) -> pd.DataFrame:
        """Quantile normalization."""
        # Sort each column
        sorted_df = pd.DataFrame(
            np.sort(expression.values, axis=0),
            index=expression.index,
            columns=expression.columns,
        )

        # Calculate mean across rows
        mean_values = sorted_df.mean(axis=1)

        # Rank original data
        ranks = expression.rank(method="min").astype(int) - 1

        # Map ranks to mean values
        normalized = ranks.apply(lambda col: mean_values.iloc[col.values].values)
        normalized.index = expression.index

        logger.info("Applied quantile normalization")
        return normalized

    def _normalize_inverse_normal(self, expression: pd.DataFrame) -> pd.DataFrame:
        """Inverse normal (rank-based) transformation."""
        def inverse_normal_transform(x: pd.Series) -> pd.Series:
            """Transform a series to follow standard normal distribution."""
            n = len(x)
            ranks = x.rank()
            # Blom's transformation
            transformed = stats.norm.ppf((ranks - 0.375) / (n + 0.25))
            return pd.Series(transformed, index=x.index)

        normalized = expression.apply(inverse_normal_transform, axis=1)

        logger.info("Applied inverse normal transformation")
        return normalized

    def log_transform_data(
        self,
        expression: pd.DataFrame,
        pseudocount: float = 1.0,
    ) -> pd.DataFrame:
        """
        Log2 transform expression data.

        Args:
            expression: Expression matrix.
            pseudocount: Pseudocount to add before log transform.

        Returns:
            Log-transformed expression matrix.
        """
        log_expr = np.log2(expression + pseudocount)
        logger.info(f"Applied log2 transformation (pseudocount: {pseudocount})")
        return log_expr

    def prepare_tensorqtl_bed(
        self,
        expression: pd.DataFrame,
        gene_annotation: pd.DataFrame,
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Prepare expression data in BED format for tensorQTL.

        Args:
            expression: Expression matrix (genes x samples).
            gene_annotation: Gene annotation with chr, start, end columns.
            output_path: Output file path.

        Returns:
            Path to BED file.
        """
        if output_path is None:
            output_path = self.output_dir / "expression.bed.gz"

        # Ensure gene annotation has required columns
        required_cols = ["chr", "start", "end"]
        if not all(col in gene_annotation.columns for col in required_cols):
            raise ValueError(f"Gene annotation must have columns: {required_cols}")

        # Match genes
        common_genes = expression.index.intersection(gene_annotation.index)
        if len(common_genes) == 0:
            raise ValueError("No common genes between expression and annotation")

        logger.info(f"Found {len(common_genes)} genes with annotations")

        # Create BED DataFrame
        bed_df = pd.DataFrame({
            "#chr": gene_annotation.loc[common_genes, "chr"],
            "start": gene_annotation.loc[common_genes, "start"].astype(int),
            "end": gene_annotation.loc[common_genes, "end"].astype(int),
            "gene_id": common_genes,
        })

        # Add expression values
        expr_matched = expression.loc[common_genes]
        for col in expr_matched.columns:
            bed_df[col] = expr_matched[col].values

        # Sort by chromosome and position
        bed_df = bed_df.sort_values(["#chr", "start"]).reset_index(drop=True)

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        compression = "gzip" if str(output_path).endswith(".gz") else None
        bed_df.to_csv(output_path, sep="\t", index=False, compression=compression)

        logger.info(f"Wrote tensorQTL BED file: {output_path}")
        return output_path

    def preprocess(
        self,
        expression_file: str | Path,
        gene_annotation_file: str | Path | None = None,
        sample_mapping: dict[str, str] | None = None,
        remove_outliers: bool = True,
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Run complete phenotype preprocessing pipeline.

        Args:
            expression_file: Path to expression data file.
            gene_annotation_file: Path to gene annotation file.
            sample_mapping: Optional sample ID mapping.
            remove_outliers: Whether to remove outlier samples.
            output_path: Output file path.

        Returns:
            Path to processed expression file.
        """
        logger.info("Starting phenotype preprocessing pipeline")

        # Load data
        expression = self.load_expression_data(
            expression_file,
            sample_mapping=sample_mapping,
        )

        initial_genes = len(expression)
        initial_samples = expression.shape[1]

        # Log transform first if data appears to be raw counts
        if self.log_transform and expression.max().max() > 100:
            expression = self.log_transform_data(expression)

        # Filter lowly expressed genes
        expression = self.filter_lowly_expressed(expression)
        genes_after_expr = len(expression)

        # Filter low variance genes
        expression = self.filter_low_variance(expression)
        genes_after_var = len(expression)

        # Detect and remove outliers
        if remove_outliers:
            outliers = self.detect_outlier_samples(expression)
            if outliers:
                expression = expression.drop(columns=outliers)
        samples_after_outlier = expression.shape[1]

        # Normalize
        expression = self.normalize(expression)

        # Store QC stats
        self._qc_stats = PhenotypeQCStats(
            total_genes=initial_genes,
            total_samples=initial_samples,
            genes_after_expression_filter=genes_after_expr,
            genes_after_variance_filter=genes_after_var,
            samples_after_outlier_filter=samples_after_outlier,
            final_genes=len(expression),
            final_samples=expression.shape[1],
        )

        # Prepare output
        if gene_annotation_file:
            validate_file_exists(gene_annotation_file, "Gene annotation file")
            gene_annotation = pd.read_csv(gene_annotation_file, sep="\t", index_col=0)
            output_path = self.prepare_tensorqtl_bed(
                expression,
                gene_annotation,
                output_path,
            )
        else:
            if output_path is None:
                output_path = self.output_dir / "expression_normalized.tsv.gz"
            output_path = write_expression_matrix(
                expression,
                output_path,
                compress=True,
            )

        logger.info(f"Phenotype preprocessing complete: {output_path}")
        return output_path
