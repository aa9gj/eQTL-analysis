"""Tests for results handling."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eqtl_analysis.analysis.results import EQTLResults, EQTLSummary


@pytest.fixture
def sample_results_data() -> pd.DataFrame:
    """Create sample eQTL results for testing."""
    np.random.seed(42)
    n_results = 1000

    return pd.DataFrame({
        "phenotype_id": [f"GENE_{i % 100:04d}" for i in range(n_results)],
        "variant_id": [f"rs{i}" for i in range(n_results)],
        "pval_nominal": np.random.uniform(0, 1, n_results),
        "slope": np.random.normal(0, 1, n_results),
        "slope_se": np.random.uniform(0.1, 0.5, n_results),
    })


@pytest.fixture
def sample_results_file(temp_dir: Path, sample_results_data: pd.DataFrame) -> Path:
    """Create a sample results file for testing."""
    results_path = temp_dir / "results.tsv"
    sample_results_data.to_csv(results_path, sep="\t", index=False)
    return results_path


class TestEQTLResults:
    """Tests for EQTLResults class."""

    def test_load_results(
        self,
        temp_dir: Path,
        sample_results_file: Path,
    ) -> None:
        """Test loading results from file."""
        results = EQTLResults(output_dir=temp_dir)
        df = results.load(sample_results_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1000

    def test_apply_fdr_correction_bh(
        self,
        temp_dir: Path,
        sample_results_file: Path,
    ) -> None:
        """Test applying BH FDR correction."""
        results = EQTLResults(output_dir=temp_dir)
        results.load(sample_results_file)

        corrected = results.apply_fdr_correction(method="bh")

        assert "qval" in corrected.columns
        assert all(corrected["qval"] >= 0)
        assert all(corrected["qval"] <= 1)

    def test_apply_fdr_correction_bonferroni(
        self,
        temp_dir: Path,
        sample_results_file: Path,
    ) -> None:
        """Test applying Bonferroni correction."""
        results = EQTLResults(output_dir=temp_dir)
        results.load(sample_results_file)

        corrected = results.apply_fdr_correction(method="bonferroni")

        assert "qval" in corrected.columns
        # Bonferroni should be more conservative
        n_sig_bonf = (corrected["qval"] <= 0.05).sum()

        results.apply_fdr_correction(method="bh")
        n_sig_bh = (results.results["qval"] <= 0.05).sum()

        assert n_sig_bonf <= n_sig_bh

    def test_get_significant(
        self,
        temp_dir: Path,
        sample_results_file: Path,
    ) -> None:
        """Test getting significant results."""
        results = EQTLResults(output_dir=temp_dir, fdr_threshold=0.05)
        results.load(sample_results_file)
        results.apply_fdr_correction()

        significant = results.get_significant()

        assert len(significant) <= len(results.results)
        assert all(significant["qval"] <= 0.05)

    def test_get_lead_variants(
        self,
        temp_dir: Path,
        sample_results_file: Path,
    ) -> None:
        """Test getting lead variants per gene."""
        results = EQTLResults(output_dir=temp_dir, fdr_threshold=0.5)  # High threshold for test
        results.load(sample_results_file)
        results.apply_fdr_correction()

        lead = results.get_lead_variants()

        # Should have at most one variant per gene
        assert len(lead) == lead["phenotype_id"].nunique()

    def test_save_results(
        self,
        temp_dir: Path,
        sample_results_file: Path,
    ) -> None:
        """Test saving results to file."""
        results = EQTLResults(output_dir=temp_dir)
        results.load(sample_results_file)

        output_path = temp_dir / "saved_results.tsv"
        saved_path = results.save(output_path)

        assert saved_path.exists()

        # Read back and verify
        df = pd.read_csv(saved_path, sep="\t")
        assert len(df) == len(results.results)

    def test_summary_calculation(
        self,
        temp_dir: Path,
        sample_results_file: Path,
    ) -> None:
        """Test summary statistics calculation."""
        results = EQTLResults(output_dir=temp_dir, fdr_threshold=0.5)
        results.load(sample_results_file)
        results.apply_fdr_correction()

        assert results.summary is not None
        assert results.summary.total_associations == 1000
        assert results.summary.genes_tested == 100

    def test_generate_report(
        self,
        temp_dir: Path,
        sample_results_file: Path,
    ) -> None:
        """Test report generation."""
        results = EQTLResults(output_dir=temp_dir)
        results.load(sample_results_file)
        results.apply_fdr_correction()

        report_path = results.generate_report()

        assert report_path.exists()
        content = report_path.read_text()
        assert "eQTL Analysis Report" in content

    def test_annotate_results(
        self,
        temp_dir: Path,
        sample_results_file: Path,
    ) -> None:
        """Test result annotation."""
        results = EQTLResults(output_dir=temp_dir)
        results.load(sample_results_file)

        # Create gene annotation
        gene_ids = results.results["phenotype_id"].unique()
        gene_annotation = pd.DataFrame({
            "gene_name": [f"Gene_{i}" for i in range(len(gene_ids))],
            "biotype": ["protein_coding"] * len(gene_ids),
        }, index=gene_ids)

        annotated = results.annotate_results(gene_annotation=gene_annotation)

        assert "gene_name" in annotated.columns
        assert "biotype" in annotated.columns
