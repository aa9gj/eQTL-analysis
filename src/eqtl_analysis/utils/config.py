"""Configuration management for the eQTL analysis pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class QualityControlConfig:
    """Quality control thresholds for genotype data."""

    min_call_rate: float = 0.95
    min_maf: float = 0.01
    max_missing_rate: float = 0.05
    hwe_pvalue_threshold: float = 1e-6
    min_gq: int = 20
    min_dp: int = 10


@dataclass
class ImputationConfig:
    """Configuration for genotype imputation using Beagle."""

    beagle_jar_path: str | None = None
    reference_panel: str | None = None
    window_size: int = 50000
    overlap: int = 3000
    ne: int = 20000
    nthreads: int = 4
    memory_gb: int = 8


@dataclass
class LiftoverConfig:
    """Configuration for coordinate liftover."""

    source_assembly: str = "canfam3"
    target_assembly: str = "canfam4"
    chain_file: str | None = None
    min_match: float = 0.95


@dataclass
class TensorQTLConfig:
    """Configuration for tensorQTL analysis."""

    mode: str = "cis"
    window: int = 1000000
    maf_threshold: float = 0.05
    fdr_threshold: float = 0.05
    permutations: int = 10000
    use_gpu: bool = False
    seed: int = 42


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""

    # Input paths
    genotype_files: list[str] = field(default_factory=list)
    expression_file: str | None = None
    covariate_file: str | None = None
    sample_mapping_file: str | None = None

    # Output paths
    output_dir: str = "results"
    log_dir: str = "logs"

    # Reference files
    reference_genome: str | None = None
    gene_annotation: str | None = None

    # Sub-configurations
    quality_control: QualityControlConfig = field(default_factory=QualityControlConfig)
    imputation: ImputationConfig = field(default_factory=ImputationConfig)
    liftover: LiftoverConfig = field(default_factory=LiftoverConfig)
    tensorqtl: TensorQTLConfig = field(default_factory=TensorQTLConfig)

    # Runtime settings
    n_jobs: int = 4
    random_seed: int = 42
    verbose: bool = True


class Config:
    """Configuration manager for the eQTL analysis pipeline."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML configuration file.
                        If None, uses default configuration.
        """
        self._config = PipelineConfig()
        if config_path is not None:
            self.load(config_path)

    @property
    def pipeline(self) -> PipelineConfig:
        """Get the pipeline configuration."""
        return self._config

    @property
    def qc(self) -> QualityControlConfig:
        """Get quality control configuration."""
        return self._config.quality_control

    @property
    def imputation(self) -> ImputationConfig:
        """Get imputation configuration."""
        return self._config.imputation

    @property
    def liftover(self) -> LiftoverConfig:
        """Get liftover configuration."""
        return self._config.liftover

    @property
    def tensorqtl(self) -> TensorQTLConfig:
        """Get tensorQTL configuration."""
        return self._config.tensorqtl

    def load(self, config_path: str | Path) -> None:
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the configuration file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the configuration file is invalid.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Empty configuration file: {config_path}")

        self._update_config(data)

    def _update_config(self, data: dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        # Update main config fields
        for key, value in data.items():
            if key == "quality_control" and isinstance(value, dict):
                for qc_key, qc_value in value.items():
                    if hasattr(self._config.quality_control, qc_key):
                        setattr(self._config.quality_control, qc_key, qc_value)
            elif key == "imputation" and isinstance(value, dict):
                for imp_key, imp_value in value.items():
                    if hasattr(self._config.imputation, imp_key):
                        setattr(self._config.imputation, imp_key, imp_value)
            elif key == "liftover" and isinstance(value, dict):
                for lo_key, lo_value in value.items():
                    if hasattr(self._config.liftover, lo_key):
                        setattr(self._config.liftover, lo_key, lo_value)
            elif key == "tensorqtl" and isinstance(value, dict):
                for tq_key, tq_value in value.items():
                    if hasattr(self._config.tensorqtl, tq_key):
                        setattr(self._config.tensorqtl, tq_key, tq_value)
            elif hasattr(self._config, key):
                setattr(self._config, key, value)

    def save(self, config_path: str | Path) -> None:
        """
        Save current configuration to a YAML file.

        Args:
            config_path: Path to save the configuration.
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = self._to_dict()
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict

        return asdict(self._config)

    def validate(self) -> list[str]:
        """
        Validate the configuration.

        Returns:
            List of validation error messages. Empty if valid.
        """
        errors = []

        # Check required files exist if specified
        if self._config.expression_file:
            if not Path(self._config.expression_file).exists():
                errors.append(f"Expression file not found: {self._config.expression_file}")

        for geno_file in self._config.genotype_files:
            if not Path(geno_file).exists():
                errors.append(f"Genotype file not found: {geno_file}")

        # Validate threshold ranges
        if not 0 <= self.qc.min_call_rate <= 1:
            errors.append("min_call_rate must be between 0 and 1")

        if not 0 <= self.qc.min_maf <= 0.5:
            errors.append("min_maf must be between 0 and 0.5")

        if self.tensorqtl.window <= 0:
            errors.append("tensorqtl window must be positive")

        if not 0 < self.tensorqtl.fdr_threshold <= 1:
            errors.append("fdr_threshold must be between 0 and 1")

        return errors

    @classmethod
    def from_env(cls) -> Config:
        """
        Create configuration from environment variables.

        Environment variables should be prefixed with EQTL_.

        Returns:
            Config instance with values from environment.
        """
        config = cls()

        # Map environment variables to config
        env_mappings = {
            "EQTL_EXPRESSION_FILE": ("expression_file", str),
            "EQTL_OUTPUT_DIR": ("output_dir", str),
            "EQTL_N_JOBS": ("n_jobs", int),
            "EQTL_VERBOSE": ("verbose", lambda x: x.lower() == "true"),
            "EQTL_MIN_MAF": ("quality_control.min_maf", float),
            "EQTL_USE_GPU": ("tensorqtl.use_gpu", lambda x: x.lower() == "true"),
        }

        for env_var, (attr_path, converter) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                parts = attr_path.split(".")
                if len(parts) == 1:
                    setattr(config._config, parts[0], converter(value))
                else:
                    sub_config = getattr(config._config, parts[0])
                    setattr(sub_config, parts[1], converter(value))

        return config


def load_config(config_path: str | Path | None = None) -> Config:
    """
    Load configuration from file or create default.

    Args:
        config_path: Optional path to configuration file.

    Returns:
        Loaded or default configuration.
    """
    return Config(config_path)
