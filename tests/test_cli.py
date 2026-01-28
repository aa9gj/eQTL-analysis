"""Tests for CLI interface."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from eqtl_analysis.cli import main


@pytest.fixture
def runner() -> CliRunner:
    """Create CLI test runner."""
    return CliRunner()


class TestCLI:
    """Tests for CLI commands."""

    def test_main_help(self, runner: CliRunner) -> None:
        """Test main help command."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "eQTL Analysis Pipeline" in result.output

    def test_main_version(self, runner: CliRunner) -> None:
        """Test version command."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output

    def test_init_config(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test init-config command."""
        config_path = temp_dir / "config.yaml"
        result = runner.invoke(main, ["init-config", "-o", str(config_path)])
        assert result.exit_code == 0
        assert config_path.exists()

    def test_validate_expression(
        self,
        runner: CliRunner,
        sample_expression_file: Path,
    ) -> None:
        """Test validate command with expression file."""
        result = runner.invoke(
            main,
            ["validate", "-e", str(sample_expression_file)],
        )
        assert result.exit_code == 0
        assert "Valid format" in result.output

    def test_validate_vcf(
        self,
        runner: CliRunner,
        sample_vcf_file: Path,
    ) -> None:
        """Test validate command with VCF file."""
        result = runner.invoke(
            main,
            ["validate", "--vcf", str(sample_vcf_file)],
        )
        assert result.exit_code == 0
        assert "Valid format" in result.output

    def test_preprocess_phenotypes_help(self, runner: CliRunner) -> None:
        """Test preprocess-phenotypes help."""
        result = runner.invoke(main, ["preprocess-phenotypes", "--help"])
        assert result.exit_code == 0
        assert "expression" in result.output.lower()

    def test_preprocess_genotypes_help(self, runner: CliRunner) -> None:
        """Test preprocess-genotypes help."""
        result = runner.invoke(main, ["preprocess-genotypes", "--help"])
        assert result.exit_code == 0
        assert "axiom" in result.output.lower() or "vcf" in result.output.lower()

    def test_run_help(self, runner: CliRunner) -> None:
        """Test run help."""
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "genotypes" in result.output.lower()
        assert "phenotypes" in result.output.lower()

    def test_summarize_help(self, runner: CliRunner) -> None:
        """Test summarize help."""
        result = runner.invoke(main, ["summarize", "--help"])
        assert result.exit_code == 0
        assert "results" in result.output.lower()
