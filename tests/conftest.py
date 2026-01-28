"""Pytest configuration and fixtures for eQTL analysis tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_expression_data() -> pd.DataFrame:
    """Create sample expression data for testing."""
    np.random.seed(42)
    n_genes = 100
    n_samples = 33  # Same as canine study

    genes = [f"GENE_{i:04d}" for i in range(n_genes)]
    samples = [f"SAMPLE_{i:02d}" for i in range(n_samples)]

    # Generate expression values (log2 scale, roughly TPM-like)
    data = np.random.normal(loc=5, scale=2, size=(n_genes, n_samples))
    data = np.clip(data, 0, None)

    df = pd.DataFrame(data, index=genes, columns=samples)
    return df


@pytest.fixture
def sample_genotype_data() -> pd.DataFrame:
    """Create sample genotype data for testing."""
    np.random.seed(42)
    n_variants = 1000
    n_samples = 33

    samples = [f"SAMPLE_{i:02d}" for i in range(n_samples)]

    # Generate variant info
    chroms = np.random.choice([f"chr{i}" for i in range(1, 23)], n_variants)
    positions = np.random.randint(1, 100000000, n_variants)
    refs = np.random.choice(["A", "C", "G", "T"], n_variants)
    alts = np.random.choice(["A", "C", "G", "T"], n_variants)

    data = {
        "chrom": chroms,
        "pos": positions,
        "id": [f"rs{i}" for i in range(n_variants)],
        "ref": refs,
        "alt": alts,
    }

    # Generate genotypes (0/0, 0/1, 1/1)
    for sample in samples:
        genotypes = np.random.choice(["0/0", "0/1", "1/1"], n_variants, p=[0.5, 0.35, 0.15])
        data[sample] = genotypes

    return pd.DataFrame(data)


@pytest.fixture
def sample_covariate_data() -> pd.DataFrame:
    """Create sample covariate data for testing."""
    np.random.seed(42)
    n_samples = 33

    samples = [f"SAMPLE_{i:02d}" for i in range(n_samples)]

    covariates = {
        "age": np.random.normal(5, 2, n_samples),
        "sex": np.random.choice([0, 1], n_samples),
        "batch": np.random.choice(["batch1", "batch2", "batch3"], n_samples),
        "PC1": np.random.normal(0, 1, n_samples),
        "PC2": np.random.normal(0, 1, n_samples),
    }

    df = pd.DataFrame(covariates, index=samples).T
    return df


@pytest.fixture
def sample_vcf_file(temp_dir: Path, sample_genotype_data: pd.DataFrame) -> Path:
    """Create a sample VCF file for testing."""
    vcf_path = temp_dir / "test.vcf"

    with open(vcf_path, "w") as f:
        # Header
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=test\n")

        # Column header
        sample_cols = [c for c in sample_genotype_data.columns
                      if c not in {"chrom", "pos", "id", "ref", "alt"}]
        header_cols = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
        f.write("\t".join(header_cols + sample_cols) + "\n")

        # Data
        for _, row in sample_genotype_data.iterrows():
            fields = [
                str(row["chrom"]),
                str(row["pos"]),
                str(row["id"]),
                str(row["ref"]),
                str(row["alt"]),
                ".",
                "PASS",
                ".",
                "GT",
            ]
            for sample in sample_cols:
                fields.append(str(row[sample]))
            f.write("\t".join(fields) + "\n")

    return vcf_path


@pytest.fixture
def sample_expression_file(
    temp_dir: Path,
    sample_expression_data: pd.DataFrame,
) -> Path:
    """Create a sample expression file for testing."""
    expr_path = temp_dir / "expression.tsv"
    sample_expression_data.to_csv(expr_path, sep="\t")
    return expr_path


@pytest.fixture
def sample_covariate_file(
    temp_dir: Path,
    sample_covariate_data: pd.DataFrame,
) -> Path:
    """Create a sample covariate file for testing."""
    cov_path = temp_dir / "covariates.tsv"
    sample_covariate_data.to_csv(cov_path, sep="\t")
    return cov_path


@pytest.fixture
def sample_gene_annotation() -> pd.DataFrame:
    """Create sample gene annotation for testing."""
    np.random.seed(42)
    n_genes = 100

    genes = [f"GENE_{i:04d}" for i in range(n_genes)]

    df = pd.DataFrame({
        "chr": np.random.choice([f"chr{i}" for i in range(1, 23)], n_genes),
        "start": np.random.randint(1, 100000000, n_genes),
        "end": np.random.randint(1, 100000000, n_genes),
        "gene_name": [f"Gene{i}" for i in range(n_genes)],
    }, index=genes)

    # Ensure end > start
    df["end"] = df["start"] + np.random.randint(1000, 100000, n_genes)

    return df
