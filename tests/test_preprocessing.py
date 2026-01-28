"""Tests for preprocessing modules."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eqtl_analysis.preprocessing.covariates import CovariatePreprocessor
from eqtl_analysis.preprocessing.genotypes import GenotypePreprocessor
from eqtl_analysis.preprocessing.phenotypes import PhenotypePreprocessor


class TestGenotypePreprocessor:
    """Tests for GenotypePreprocessor."""

    def test_initialization(self, temp_dir: Path) -> None:
        """Test preprocessor initialization."""
        preprocessor = GenotypePreprocessor(output_dir=temp_dir)
        assert preprocessor.output_dir == temp_dir

    def test_genotypes_to_numeric(self, temp_dir: Path) -> None:
        """Test genotype string to numeric conversion."""
        preprocessor = GenotypePreprocessor(output_dir=temp_dir)

        geno_df = pd.DataFrame({
            "S1": ["0/0", "0/1", "1/1", "./."],
            "S2": ["0|0", "0|1", "1|1", ".|."],
        })

        numeric = preprocessor._genotypes_to_numeric(geno_df)

        assert numeric.loc[0, "S1"] == 0
        assert numeric.loc[1, "S1"] == 1
        assert numeric.loc[2, "S1"] == 2
        assert pd.isna(numeric.loc[3, "S1"])

    def test_calculate_maf(self, temp_dir: Path) -> None:
        """Test MAF calculation."""
        preprocessor = GenotypePreprocessor(output_dir=temp_dir)

        # Create genotype matrix where MAF should be 0.25
        # 8 ref homozygotes, 4 het = MAF = 4 / (2*12) = 0.167
        geno_matrix = pd.DataFrame(
            [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]],
            index=["var1"],
        )

        maf = preprocessor._calculate_maf(geno_matrix)
        assert abs(maf.iloc[0] - 0.167) < 0.01


class TestPhenotypePreprocessor:
    """Tests for PhenotypePreprocessor."""

    def test_initialization(self, temp_dir: Path) -> None:
        """Test preprocessor initialization."""
        preprocessor = PhenotypePreprocessor(output_dir=temp_dir)
        assert preprocessor.min_expression == 1.0
        assert preprocessor.normalization_method == "inverse_normal"

    def test_filter_lowly_expressed(
        self,
        temp_dir: Path,
        sample_expression_data: pd.DataFrame,
    ) -> None:
        """Test filtering lowly expressed genes."""
        preprocessor = PhenotypePreprocessor(output_dir=temp_dir)

        # Set high threshold to filter some genes
        filtered = preprocessor.filter_lowly_expressed(
            sample_expression_data,
            min_expression=6.0,  # Above mean
            min_samples=0.5,
        )

        assert len(filtered) < len(sample_expression_data)

    def test_filter_low_variance(
        self,
        temp_dir: Path,
        sample_expression_data: pd.DataFrame,
    ) -> None:
        """Test filtering low variance genes."""
        preprocessor = PhenotypePreprocessor(output_dir=temp_dir)

        # Add a constant gene
        sample_expression_data.loc["CONSTANT_GENE"] = 5.0

        filtered = preprocessor.filter_low_variance(
            sample_expression_data,
            min_variance=0.01,
        )

        assert "CONSTANT_GENE" not in filtered.index

    def test_normalize_inverse_normal(
        self,
        temp_dir: Path,
        sample_expression_data: pd.DataFrame,
    ) -> None:
        """Test inverse normal transformation."""
        preprocessor = PhenotypePreprocessor(output_dir=temp_dir)

        normalized = preprocessor.normalize(sample_expression_data, method="inverse_normal")

        # Check that values follow approximately standard normal
        assert abs(normalized.values.mean()) < 0.1
        assert abs(normalized.values.std() - 1) < 0.2

    def test_detect_outlier_samples(
        self,
        temp_dir: Path,
        sample_expression_data: pd.DataFrame,
    ) -> None:
        """Test outlier sample detection."""
        preprocessor = PhenotypePreprocessor(output_dir=temp_dir)

        # Add an obvious outlier
        sample_expression_data["OUTLIER"] = 100.0

        outliers = preprocessor.detect_outlier_samples(
            sample_expression_data,
            method="correlation",
            threshold=2.0,
        )

        assert "OUTLIER" in outliers


class TestCovariatePreprocessor:
    """Tests for CovariatePreprocessor."""

    def test_initialization(self, temp_dir: Path) -> None:
        """Test preprocessor initialization."""
        preprocessor = CovariatePreprocessor(output_dir=temp_dir)
        assert preprocessor.n_peer_factors == 15
        assert preprocessor.n_genotype_pcs == 5

    def test_compute_peer_pca_fallback(
        self,
        temp_dir: Path,
        sample_expression_data: pd.DataFrame,
    ) -> None:
        """Test PEER factor computation via PCA fallback."""
        preprocessor = CovariatePreprocessor(
            n_peer_factors=5,
            output_dir=temp_dir,
        )

        factors = preprocessor._compute_peer_pca_fallback(
            sample_expression_data,
            n_factors=5,
        )

        assert factors.shape[0] == 5
        assert factors.shape[1] == sample_expression_data.shape[1]

    def test_encode_categorical_covariates(
        self,
        temp_dir: Path,
        sample_covariate_data: pd.DataFrame,
    ) -> None:
        """Test categorical covariate encoding."""
        preprocessor = CovariatePreprocessor(output_dir=temp_dir)

        encoded = preprocessor.encode_categorical_covariates(
            sample_covariate_data,
            categorical_columns=["batch"],
        )

        # Should have encoded batch into dummy variables
        assert "batch" not in encoded.index
        assert any("batch_" in idx for idx in encoded.index)

    def test_combine_covariates(
        self,
        temp_dir: Path,
    ) -> None:
        """Test combining multiple covariate matrices."""
        preprocessor = CovariatePreprocessor(output_dir=temp_dir)

        samples = ["S1", "S2", "S3"]

        cov1 = pd.DataFrame({s: [1.0, 2.0] for s in samples}, index=["cov1", "cov2"])
        cov2 = pd.DataFrame({s: [3.0] for s in samples}, index=["cov3"])

        combined = preprocessor.combine_covariates(cov1, cov2)

        assert len(combined) == 3
        assert all(c in combined.index for c in ["cov1", "cov2", "cov3"])

    def test_scale_covariates(
        self,
        temp_dir: Path,
        sample_covariate_data: pd.DataFrame,
    ) -> None:
        """Test covariate scaling."""
        preprocessor = CovariatePreprocessor(output_dir=temp_dir)

        # Get only numeric covariates
        numeric_cov = sample_covariate_data.drop(index=["batch"])
        numeric_cov = numeric_cov.astype(float)

        scaled = preprocessor.scale_covariates(numeric_cov, method="standard")

        # Each row should have approximately mean 0 and std 1
        for idx in scaled.index:
            row = scaled.loc[idx]
            assert abs(row.mean()) < 0.1
            assert abs(row.std() - 1) < 0.1
