"""Tests for configuration management."""

from pathlib import Path

import pytest

from eqtl_analysis.utils.config import (
    Config,
    PipelineConfig,
    QualityControlConfig,
    TensorQTLConfig,
    load_config,
)


class TestQualityControlConfig:
    """Tests for QualityControlConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = QualityControlConfig()
        assert config.min_call_rate == 0.95
        assert config.min_maf == 0.01
        assert config.max_missing_rate == 0.05
        assert config.hwe_pvalue_threshold == 1e-6

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = QualityControlConfig(
            min_call_rate=0.9,
            min_maf=0.05,
        )
        assert config.min_call_rate == 0.9
        assert config.min_maf == 0.05


class TestTensorQTLConfig:
    """Tests for TensorQTLConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = TensorQTLConfig()
        assert config.mode == "cis"
        assert config.window == 1000000
        assert config.maf_threshold == 0.05
        assert config.fdr_threshold == 0.05
        assert config.permutations == 10000
        assert config.use_gpu is False

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = TensorQTLConfig(
            mode="trans",
            window=500000,
            use_gpu=True,
        )
        assert config.mode == "trans"
        assert config.window == 500000
        assert config.use_gpu is True


class TestConfig:
    """Tests for Config class."""

    def test_default_config(self) -> None:
        """Test creating default configuration."""
        config = Config()
        assert config.pipeline is not None
        assert isinstance(config.qc, QualityControlConfig)
        assert isinstance(config.tensorqtl, TensorQTLConfig)

    def test_load_yaml_config(self, temp_dir: Path) -> None:
        """Test loading configuration from YAML file."""
        config_content = """
output_dir: custom_results
n_jobs: 8
quality_control:
    min_call_rate: 0.9
    min_maf: 0.02
tensorqtl:
    window: 500000
    permutations: 5000
"""
        config_path = temp_dir / "config.yaml"
        config_path.write_text(config_content)

        config = Config(config_path)
        assert config.pipeline.output_dir == "custom_results"
        assert config.pipeline.n_jobs == 8
        assert config.qc.min_call_rate == 0.9
        assert config.qc.min_maf == 0.02
        assert config.tensorqtl.window == 500000
        assert config.tensorqtl.permutations == 5000

    def test_load_nonexistent_file(self) -> None:
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            Config("/nonexistent/path/config.yaml")

    def test_save_config(self, temp_dir: Path) -> None:
        """Test saving configuration to file."""
        config = Config()
        config.pipeline.output_dir = "test_output"
        config.qc.min_maf = 0.03

        save_path = temp_dir / "saved_config.yaml"
        config.save(save_path)

        assert save_path.exists()

        # Load and verify
        loaded_config = Config(save_path)
        assert loaded_config.pipeline.output_dir == "test_output"
        assert loaded_config.qc.min_maf == 0.03

    def test_validate_valid_config(self) -> None:
        """Test validation of valid configuration."""
        config = Config()
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_invalid_thresholds(self) -> None:
        """Test validation catches invalid thresholds."""
        config = Config()
        config.qc.min_call_rate = 1.5  # Invalid: > 1
        config.qc.min_maf = 0.6  # Invalid: > 0.5
        config.tensorqtl.window = -100  # Invalid: negative

        errors = config.validate()
        assert len(errors) > 0


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_default(self) -> None:
        """Test loading default configuration."""
        config = load_config()
        assert isinstance(config, Config)
        assert config.pipeline is not None

    def test_load_from_file(self, temp_dir: Path) -> None:
        """Test loading configuration from file."""
        config_content = "n_jobs: 16"
        config_path = temp_dir / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        assert config.pipeline.n_jobs == 16
