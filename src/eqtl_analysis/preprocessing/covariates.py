"""Covariate preprocessing module for eQTL analysis.

This module handles:
- Loading and formatting covariate data
- Computing PEER factors from expression data
- Computing genetic PCs from genotype data
- Combining known and hidden covariates
- Preparing covariates for tensorQTL
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from eqtl_analysis.utils.io import ensure_directory, read_expression_matrix
from eqtl_analysis.utils.logging import get_logger
from eqtl_analysis.utils.validators import (
    validate_covariate_matrix,
    validate_file_exists,
)

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class CovariateStats:
    """Statistics about computed covariates."""

    n_known_covariates: int
    n_peer_factors: int
    n_genotype_pcs: int
    n_total_covariates: int
    n_samples: int


class CovariatePreprocessor:
    """Preprocessor for covariates in eQTL analysis."""

    def __init__(
        self,
        n_peer_factors: int = 15,
        n_genotype_pcs: int = 5,
        include_known_covariates: bool = True,
        output_dir: str | Path = "results/covariates",
    ) -> None:
        """
        Initialize covariate preprocessor.

        Args:
            n_peer_factors: Number of PEER factors to compute.
            n_genotype_pcs: Number of genotype PCs to compute.
            include_known_covariates: Whether to include known covariates.
            output_dir: Output directory for processed files.
        """
        self.n_peer_factors = n_peer_factors
        self.n_genotype_pcs = n_genotype_pcs
        self.include_known_covariates = include_known_covariates
        self.output_dir = ensure_directory(output_dir)
        self._stats: CovariateStats | None = None

    @property
    def stats(self) -> CovariateStats | None:
        """Get statistics from last run."""
        return self._stats

    def load_known_covariates(
        self,
        file_path: str | Path,
        sample_mapping: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """
        Load known covariates from file.

        Args:
            file_path: Path to covariate file (TSV format).
            sample_mapping: Optional sample ID mapping.

        Returns:
            Covariate matrix (covariates x samples).
        """
        file_path = Path(file_path)
        validate_file_exists(file_path, "Covariate file")

        df = pd.read_csv(file_path, sep="\t", index_col=0)

        # Ensure covariates x samples orientation
        if df.shape[0] > df.shape[1]:
            logger.info("Transposing covariate matrix to covariates x samples")
            df = df.T

        # Apply sample mapping
        if sample_mapping:
            df.columns = [sample_mapping.get(c, c) for c in df.columns]

        logger.info(f"Loaded {df.shape[0]} known covariates for {df.shape[1]} samples")
        return df

    def compute_peer_factors(
        self,
        expression: pd.DataFrame,
        n_factors: int | None = None,
        max_iterations: int = 1000,
    ) -> pd.DataFrame:
        """
        Compute PEER (Probabilistic Estimation of Expression Residuals) factors.

        Uses PCA as a fallback when the PEER package is not available.

        Args:
            expression: Expression matrix (genes x samples).
            n_factors: Number of factors to compute.
            max_iterations: Maximum iterations for optimization.

        Returns:
            PEER factors matrix (factors x samples).
        """
        n_factors = n_factors or self.n_peer_factors

        try:
            return self._compute_peer_native(expression, n_factors, max_iterations)
        except ImportError:
            logger.warning("PEER package not available, using PCA as fallback")
            return self._compute_peer_pca_fallback(expression, n_factors)

    def _compute_peer_native(
        self,
        expression: pd.DataFrame,
        n_factors: int,
        max_iterations: int,
    ) -> pd.DataFrame:
        """Compute PEER factors using the native PEER package."""
        import peer

        # Initialize PEER model
        model = peer.PEER()
        model.setPhenoMean(expression.values.T)  # samples x genes
        model.setNk(n_factors)
        model.setNmax_iterations(max_iterations)

        # Run inference
        model.update()

        # Get factors
        factors = model.getX()  # samples x factors
        factor_df = pd.DataFrame(
            factors.T,
            index=[f"PEER{i+1}" for i in range(n_factors)],
            columns=expression.columns,
        )

        logger.info(f"Computed {n_factors} PEER factors")
        return factor_df

    def _compute_peer_pca_fallback(
        self,
        expression: pd.DataFrame,
        n_factors: int,
    ) -> pd.DataFrame:
        """Compute hidden factors using PCA as PEER fallback."""
        # Center and scale
        scaler = StandardScaler()
        expr_scaled = scaler.fit_transform(expression.T)  # samples x genes

        # PCA
        n_components = min(n_factors, min(expr_scaled.shape) - 1)
        pca = PCA(n_components=n_components)
        factors = pca.fit_transform(expr_scaled)  # samples x components

        # Create DataFrame
        factor_df = pd.DataFrame(
            factors.T,
            index=[f"InferredCov{i+1}" for i in range(n_components)],
            columns=expression.columns,
        )

        variance_explained = pca.explained_variance_ratio_.sum() * 100
        logger.info(
            f"Computed {n_components} hidden factors via PCA "
            f"(variance explained: {variance_explained:.1f}%)"
        )

        return factor_df

    def compute_genotype_pcs(
        self,
        genotype_file: str | Path,
        n_pcs: int | None = None,
    ) -> pd.DataFrame:
        """
        Compute principal components from genotype data.

        Args:
            genotype_file: Path to genotype VCF or dosage file.
            n_pcs: Number of PCs to compute.

        Returns:
            PC matrix (PCs x samples).
        """
        n_pcs = n_pcs or self.n_genotype_pcs

        genotype_file = Path(genotype_file)
        validate_file_exists(genotype_file, "Genotype file")

        # Load genotype data
        from eqtl_analysis.utils.io import read_vcf

        logger.info("Loading genotype data for PC computation")
        geno_df = read_vcf(genotype_file, as_dataframe=True)

        # Get sample columns
        meta_cols = {"chrom", "pos", "id", "ref", "alt"}
        sample_cols = [c for c in geno_df.columns if c not in meta_cols]

        # Convert to numeric dosage
        def parse_genotype(gt: str) -> float:
            if pd.isna(gt) or gt in ("./.", ".|."):
                return np.nan
            parts = gt.replace("|", "/").split("/")
            try:
                return sum(int(a) for a in parts)
            except ValueError:
                return np.nan

        dosage_df = geno_df[sample_cols].apply(lambda col: col.map(parse_genotype))

        # Impute missing values with mean
        dosage_df = dosage_df.fillna(dosage_df.mean())

        # Standardize
        scaler = StandardScaler()
        dosage_scaled = scaler.fit_transform(dosage_df.T)  # samples x variants

        # PCA
        n_components = min(n_pcs, min(dosage_scaled.shape) - 1)
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(dosage_scaled)  # samples x components

        # Create DataFrame
        pc_df = pd.DataFrame(
            pcs.T,
            index=[f"genoPC{i+1}" for i in range(n_components)],
            columns=sample_cols,
        )

        variance_explained = pca.explained_variance_ratio_.sum() * 100
        logger.info(
            f"Computed {n_components} genotype PCs "
            f"(variance explained: {variance_explained:.1f}%)"
        )

        return pc_df

    def encode_categorical_covariates(
        self,
        covariates: pd.DataFrame,
        categorical_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        One-hot encode categorical covariates.

        Args:
            covariates: Covariate matrix.
            categorical_columns: Columns to encode. Auto-detected if None.

        Returns:
            Covariate matrix with encoded categoricals.
        """
        if categorical_columns is None:
            # Auto-detect categorical columns
            categorical_columns = []
            for col in covariates.index:
                if covariates.loc[col].dtype == object:
                    categorical_columns.append(col)
                elif covariates.loc[col].nunique() <= 5:
                    categorical_columns.append(col)

        if not categorical_columns:
            return covariates

        # Encode each categorical column
        encoded_dfs = []
        non_categorical = []

        for col in covariates.index:
            if col in categorical_columns:
                row = covariates.loc[col]
                dummies = pd.get_dummies(row, prefix=col, drop_first=True)
                dummy_df = dummies.T
                dummy_df.columns = covariates.columns
                encoded_dfs.append(dummy_df)
            else:
                non_categorical.append(col)

        # Combine
        result = covariates.loc[non_categorical]
        for df in encoded_dfs:
            result = pd.concat([result, df])

        logger.info(
            f"Encoded {len(categorical_columns)} categorical covariates: "
            f"{categorical_columns}"
        )

        return result

    def combine_covariates(
        self,
        *covariate_dfs: pd.DataFrame,
        sample_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Combine multiple covariate DataFrames.

        Args:
            *covariate_dfs: Covariate DataFrames to combine.
            sample_ids: Sample IDs to include. Uses intersection if None.

        Returns:
            Combined covariate matrix.
        """
        if len(covariate_dfs) == 0:
            raise ValueError("At least one covariate DataFrame required")

        # Find common samples
        if sample_ids is None:
            sample_sets = [set(df.columns) for df in covariate_dfs]
            common_samples = set.intersection(*sample_sets)
            sample_ids = sorted(common_samples)

        if len(sample_ids) == 0:
            raise ValueError("No common samples found across covariate matrices")

        # Combine
        combined = pd.concat(
            [df[sample_ids] for df in covariate_dfs],
            axis=0,
        )

        # Check for duplicate covariate names
        if combined.index.duplicated().any():
            logger.warning("Duplicate covariate names detected, making unique")
            combined.index = pd.Index([
                f"{name}_{i}" if dup else name
                for i, (name, dup) in enumerate(
                    zip(combined.index, combined.index.duplicated())
                )
            ])

        logger.info(f"Combined {len(combined)} covariates for {len(sample_ids)} samples")
        return combined

    def scale_covariates(
        self,
        covariates: pd.DataFrame,
        method: Literal["standard", "robust", "minmax"] = "standard",
    ) -> pd.DataFrame:
        """
        Scale covariate values.

        Args:
            covariates: Covariate matrix.
            method: Scaling method.

        Returns:
            Scaled covariate matrix.
        """
        if method == "standard":
            scaled = (covariates.T - covariates.T.mean()) / covariates.T.std()
            scaled = scaled.T
        elif method == "robust":
            median = covariates.median(axis=1)
            mad = (covariates.T - median).abs().median()
            scaled = ((covariates.T - median) / mad).T
        elif method == "minmax":
            min_val = covariates.min(axis=1)
            max_val = covariates.max(axis=1)
            scaled = ((covariates.T - min_val) / (max_val - min_val)).T
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        logger.info(f"Applied {method} scaling to covariates")
        return scaled

    def save_covariates(
        self,
        covariates: pd.DataFrame,
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Save covariates to file in tensorQTL format.

        Args:
            covariates: Covariate matrix.
            output_path: Output file path.

        Returns:
            Path to saved file.
        """
        if output_path is None:
            output_path = self.output_dir / "covariates.txt"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        covariates.to_csv(output_path, sep="\t")

        logger.info(f"Saved covariates to {output_path}")
        return output_path

    def preprocess(
        self,
        expression_file: str | Path,
        genotype_file: str | Path | None = None,
        known_covariates_file: str | Path | None = None,
        sample_ids: list[str] | None = None,
        sample_mapping: dict[str, str] | None = None,
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Run complete covariate preprocessing pipeline.

        Args:
            expression_file: Path to expression data for PEER factors.
            genotype_file: Path to genotype data for PCs.
            known_covariates_file: Path to known covariates.
            sample_ids: Sample IDs to include.
            sample_mapping: Optional sample ID mapping.
            output_path: Output file path.

        Returns:
            Path to processed covariate file.
        """
        logger.info("Starting covariate preprocessing pipeline")

        covariate_dfs = []
        n_known = 0
        n_peer = 0
        n_geno_pcs = 0

        # Load expression data
        expression = read_expression_matrix(expression_file)
        if sample_mapping:
            expression.columns = [
                sample_mapping.get(c, c) for c in expression.columns
            ]

        if sample_ids is None:
            sample_ids = list(expression.columns)

        # Known covariates
        if known_covariates_file and self.include_known_covariates:
            known_cov = self.load_known_covariates(
                known_covariates_file,
                sample_mapping=sample_mapping,
            )
            known_cov = self.encode_categorical_covariates(known_cov)
            covariate_dfs.append(known_cov)
            n_known = len(known_cov)

        # PEER factors
        if self.n_peer_factors > 0:
            peer_factors = self.compute_peer_factors(expression)
            covariate_dfs.append(peer_factors)
            n_peer = len(peer_factors)

        # Genotype PCs
        if genotype_file and self.n_genotype_pcs > 0:
            geno_pcs = self.compute_genotype_pcs(genotype_file)
            covariate_dfs.append(geno_pcs)
            n_geno_pcs = len(geno_pcs)

        # Combine all covariates
        if len(covariate_dfs) == 0:
            raise ValueError("No covariates to process")

        combined = self.combine_covariates(*covariate_dfs, sample_ids=sample_ids)

        # Scale
        combined = self.scale_covariates(combined)

        # Store stats
        self._stats = CovariateStats(
            n_known_covariates=n_known,
            n_peer_factors=n_peer,
            n_genotype_pcs=n_geno_pcs,
            n_total_covariates=len(combined),
            n_samples=len(combined.columns),
        )

        # Validate
        validation = validate_covariate_matrix(combined)
        if not validation["valid"]:
            logger.warning(f"Covariate validation issues: {validation['issues']}")

        # Save
        output_path = self.save_covariates(combined, output_path)

        logger.info(f"Covariate preprocessing complete: {output_path}")
        return output_path
