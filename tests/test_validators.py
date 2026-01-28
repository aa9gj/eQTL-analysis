"""Tests for data validation utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eqtl_analysis.utils.validators import (
    ValidationError,
    validate_covariate_matrix,
    validate_expression_matrix,
    validate_file_exists,
    validate_sample_consistency,
    validate_vcf_format,
)


class TestValidateFileExists:
    """Tests for validate_file_exists function."""

    def test_existing_file(self, temp_dir: Path) -> None:
        """Test validation of existing file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")

        assert validate_file_exists(test_file) is True

    def test_nonexistent_file_raises(self) -> None:
        """Test that nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            validate_file_exists("/nonexistent/file.txt")

    def test_nonexistent_file_no_raise(self) -> None:
        """Test that nonexistent file returns False when raise_error=False."""
        result = validate_file_exists(
            "/nonexistent/file.txt",
            raise_error=False,
        )
        assert result is False


class TestValidateVCFFormat:
    """Tests for validate_vcf_format function."""

    def test_valid_vcf(self, sample_vcf_file: Path) -> None:
        """Test validation of valid VCF file."""
        result = validate_vcf_format(sample_vcf_file)
        assert result["valid_header"] is True
        assert result["valid_format"] is True
        assert result["has_samples"] is True
        assert result["sample_count"] == 33

    def test_invalid_vcf_missing_header(self, temp_dir: Path) -> None:
        """Test validation catches missing header."""
        bad_vcf = temp_dir / "bad.vcf"
        bad_vcf.write_text("chr1\t100\trs1\tA\tG\n")

        with pytest.raises(ValidationError):
            validate_vcf_format(bad_vcf)


class TestValidateExpressionMatrix:
    """Tests for validate_expression_matrix function."""

    def test_valid_expression(self, sample_expression_data: pd.DataFrame) -> None:
        """Test validation of valid expression matrix."""
        result = validate_expression_matrix(sample_expression_data)
        assert result["valid"] is True
        assert result["n_genes"] == 100
        assert result["n_samples"] == 33

    def test_insufficient_genes(self) -> None:
        """Test validation catches insufficient genes."""
        df = pd.DataFrame(np.random.rand(1, 10))
        result = validate_expression_matrix(df, min_genes=5)
        assert result["valid"] is False
        assert any("genes" in issue.lower() for issue in result["issues"])

    def test_insufficient_samples(self) -> None:
        """Test validation catches insufficient samples."""
        df = pd.DataFrame(np.random.rand(10, 1))
        result = validate_expression_matrix(df, min_samples=5)
        assert result["valid"] is False
        assert any("samples" in issue.lower() for issue in result["issues"])

    def test_missing_values_detected(self) -> None:
        """Test validation detects missing values."""
        df = pd.DataFrame(np.random.rand(10, 10))
        df.iloc[0, 0] = np.nan

        result = validate_expression_matrix(df, check_values=True)
        assert result["has_missing_values"] is True

    def test_negative_values_detected(self) -> None:
        """Test validation detects negative values."""
        df = pd.DataFrame(np.random.rand(10, 10) - 0.5)  # Some negative

        result = validate_expression_matrix(df, check_values=True)
        assert result["has_negative_values"] is True


class TestValidateCovariateMatrix:
    """Tests for validate_covariate_matrix function."""

    def test_valid_covariates(self, sample_covariate_data: pd.DataFrame) -> None:
        """Test validation of valid covariate matrix."""
        result = validate_covariate_matrix(sample_covariate_data)
        assert result["valid"] is True

    def test_sample_mismatch(self, sample_covariate_data: pd.DataFrame) -> None:
        """Test validation catches sample mismatch."""
        expected_samples = ["WRONG_SAMPLE_01", "WRONG_SAMPLE_02"]
        result = validate_covariate_matrix(
            sample_covariate_data,
            sample_ids=expected_samples,
        )
        assert result["valid"] is False
        assert any("sample" in issue.lower() for issue in result["issues"])

    def test_too_many_covariates(self, sample_covariate_data: pd.DataFrame) -> None:
        """Test validation catches too many covariates."""
        result = validate_covariate_matrix(
            sample_covariate_data,
            max_covariates=2,
        )
        assert result["valid"] is False

    def test_constant_covariate_detected(self) -> None:
        """Test validation detects constant covariates."""
        df = pd.DataFrame({
            "sample1": [1.0, 1.0, 5.0],
            "sample2": [2.0, 1.0, 5.0],
        }, index=["cov1", "cov2", "cov3"])

        result = validate_covariate_matrix(df)
        # cov2 and cov3 are constant
        assert any("constant" in issue.lower() for issue in result["issues"])


class TestValidateSampleConsistency:
    """Tests for validate_sample_consistency function."""

    def test_matching_samples(self) -> None:
        """Test validation with matching samples."""
        samples = ["S1", "S2", "S3"]
        result = validate_sample_consistency(samples, samples, samples)
        assert len(result["common"]) == 3

    def test_partial_overlap(self) -> None:
        """Test validation with partial sample overlap."""
        geno_samples = ["S1", "S2", "S3", "S4"]
        expr_samples = ["S2", "S3", "S4", "S5"]

        result = validate_sample_consistency(geno_samples, expr_samples)
        assert len(result["common"]) == 3
        assert "S1" in result["genotype_only"]
        assert "S5" in result["expression_only"]

    def test_no_overlap_raises(self) -> None:
        """Test validation raises when no common samples."""
        geno_samples = ["S1", "S2"]
        expr_samples = ["S3", "S4"]

        with pytest.raises(ValidationError):
            validate_sample_consistency(geno_samples, expr_samples)
