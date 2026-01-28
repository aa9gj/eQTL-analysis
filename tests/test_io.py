"""Tests for I/O utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eqtl_analysis.utils.io import (
    VCFReader,
    read_expression_matrix,
    read_vcf,
    write_expression_matrix,
    write_vcf,
)


class TestVCFReader:
    """Tests for VCFReader class."""

    def test_read_header(self, sample_vcf_file: Path) -> None:
        """Test reading VCF header."""
        reader = VCFReader(sample_vcf_file)
        assert len(reader.header) > 0
        assert len(reader.samples) == 33

    def test_iter_variants(self, sample_vcf_file: Path) -> None:
        """Test iterating over variants."""
        reader = VCFReader(sample_vcf_file)
        variants = list(reader.iter_variants())

        assert len(variants) == 1000
        assert all("chrom" in v for v in variants)
        assert all("pos" in v for v in variants)

    def test_filter_by_chrom(self, sample_vcf_file: Path) -> None:
        """Test filtering variants by chromosome."""
        reader = VCFReader(sample_vcf_file)

        # Get all unique chromosomes first
        all_variants = list(reader.iter_variants())
        chroms = set(v["chrom"] for v in all_variants)

        if len(chroms) > 0:
            test_chrom = list(chroms)[0]
            filtered = list(reader.iter_variants(chrom=test_chrom))
            assert all(v["chrom"] == test_chrom for v in filtered)


class TestReadVCF:
    """Tests for read_vcf function."""

    def test_read_as_dataframe(self, sample_vcf_file: Path) -> None:
        """Test reading VCF as DataFrame."""
        df = read_vcf(sample_vcf_file, as_dataframe=True)

        assert isinstance(df, pd.DataFrame)
        assert "chrom" in df.columns
        assert "pos" in df.columns
        assert len(df) == 1000

    def test_read_as_reader(self, sample_vcf_file: Path) -> None:
        """Test reading VCF as VCFReader."""
        reader = read_vcf(sample_vcf_file, as_dataframe=False)
        assert isinstance(reader, VCFReader)


class TestWriteVCF:
    """Tests for write_vcf function."""

    def test_write_vcf(self, temp_dir: Path, sample_genotype_data: pd.DataFrame) -> None:
        """Test writing VCF file."""
        output_path = temp_dir / "output.vcf.gz"
        sample_cols = [c for c in sample_genotype_data.columns
                      if c not in {"chrom", "pos", "id", "ref", "alt"}]

        result_path = write_vcf(
            sample_genotype_data,
            output_path,
            sample_columns=sample_cols,
        )

        assert result_path.exists()

        # Read back and verify
        df = read_vcf(result_path, as_dataframe=True)
        assert len(df) == len(sample_genotype_data)


class TestReadExpressionMatrix:
    """Tests for read_expression_matrix function."""

    def test_read_tsv(self, sample_expression_file: Path) -> None:
        """Test reading TSV expression file."""
        df = read_expression_matrix(sample_expression_file)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (100, 33)

    def test_read_csv(self, temp_dir: Path, sample_expression_data: pd.DataFrame) -> None:
        """Test reading CSV expression file."""
        csv_path = temp_dir / "expression.csv"
        sample_expression_data.to_csv(csv_path)

        df = read_expression_matrix(csv_path)
        assert df.shape == sample_expression_data.shape


class TestWriteExpressionMatrix:
    """Tests for write_expression_matrix function."""

    def test_write_tsv(
        self,
        temp_dir: Path,
        sample_expression_data: pd.DataFrame,
    ) -> None:
        """Test writing TSV expression file."""
        output_path = temp_dir / "output.tsv"

        result_path = write_expression_matrix(sample_expression_data, output_path)
        assert result_path.exists()

        # Read back and verify
        df = read_expression_matrix(result_path)
        assert df.shape == sample_expression_data.shape

    def test_write_compressed(
        self,
        temp_dir: Path,
        sample_expression_data: pd.DataFrame,
    ) -> None:
        """Test writing compressed expression file."""
        output_path = temp_dir / "output.tsv.gz"

        result_path = write_expression_matrix(
            sample_expression_data,
            output_path,
            compress=True,
        )
        assert result_path.exists()
        assert str(result_path).endswith(".gz")
