"""TensorQTL runner module for eQTL analysis.

This module provides a wrapper around tensorQTL for:
- Cis-eQTL mapping
- Trans-eQTL mapping
- Interaction QTL (ieQTL) analysis
- Conditional analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from eqtl_analysis.utils.config import TensorQTLConfig
from eqtl_analysis.utils.io import ensure_directory
from eqtl_analysis.utils.logging import get_logger, log_step
from eqtl_analysis.utils.validators import validate_file_exists

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class QTLRunStats:
    """Statistics from a QTL analysis run."""

    mode: str
    n_phenotypes: int
    n_variants_tested: int
    n_significant: int
    n_genes_with_eqtl: int
    runtime_seconds: float


class TensorQTLRunner:
    """Runner for tensorQTL eQTL analysis."""

    def __init__(
        self,
        config: TensorQTLConfig | None = None,
        output_dir: str | Path = "results/qtl",
    ) -> None:
        """
        Initialize tensorQTL runner.

        Args:
            config: TensorQTL configuration.
            output_dir: Output directory for results.
        """
        self.config = config or TensorQTLConfig()
        self.output_dir = ensure_directory(output_dir)
        self._run_stats: QTLRunStats | None = None
        self._check_dependencies()

    @property
    def run_stats(self) -> QTLRunStats | None:
        """Get statistics from last run."""
        return self._run_stats

    def _check_dependencies(self) -> None:
        """Check that required dependencies are available."""
        try:
            import tensorqtl

            logger.debug(f"tensorQTL version: {tensorqtl.__version__}")
        except ImportError as e:
            raise ImportError(
                "tensorQTL is required but not installed. "
                "Install with: pip install tensorqtl"
            ) from e

        if self.config.use_gpu:
            try:
                import torch

                if not torch.cuda.is_available():
                    logger.warning("GPU requested but CUDA not available, using CPU")
                    self.config.use_gpu = False
                else:
                    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            except ImportError:
                logger.warning("PyTorch not properly installed for GPU, using CPU")
                self.config.use_gpu = False

    def load_data(
        self,
        genotype_file: str | Path,
        phenotype_file: str | Path,
        covariate_file: str | Path | None = None,
    ) -> tuple:
        """
        Load data for QTL analysis.

        Args:
            genotype_file: Path to genotype plink files (prefix) or VCF.
            phenotype_file: Path to phenotype BED file.
            covariate_file: Path to covariate file.

        Returns:
            Tuple of (genotype_df, variant_df, phenotype_df, phenotype_pos, covariates).
        """
        import tensorqtl
        from tensorqtl import genotypeio

        genotype_file = Path(genotype_file)
        phenotype_file = Path(phenotype_file)

        # Load phenotypes
        validate_file_exists(phenotype_file, "Phenotype file")
        phenotype_df, phenotype_pos = tensorqtl.read_phenotype_bed(str(phenotype_file))
        logger.info(
            f"Loaded phenotypes: {phenotype_df.shape[0]} genes, "
            f"{phenotype_df.shape[1]} samples"
        )

        # Load genotypes
        # Check for plink files or VCF
        if (genotype_file.parent / f"{genotype_file.name}.bed").exists():
            # Plink format
            plink_prefix = str(genotype_file)
            pr = genotypeio.PlinkReader(plink_prefix)
            genotype_df = pr.load_genotypes()
            variant_df = pr.bim.set_index("snp")[["chrom", "pos"]]
        elif genotype_file.suffix in (".vcf", ".vcf.gz") or genotype_file.exists():
            # VCF format - need to convert or use plink
            genotype_df, variant_df = self._load_vcf_genotypes(genotype_file)
        else:
            raise FileNotFoundError(f"Genotype files not found: {genotype_file}")

        logger.info(
            f"Loaded genotypes: {genotype_df.shape[0]} variants, "
            f"{genotype_df.shape[1]} samples"
        )

        # Load covariates
        covariates = None
        if covariate_file:
            validate_file_exists(covariate_file, "Covariate file")
            covariates = pd.read_csv(covariate_file, sep="\t", index_col=0)

            # Ensure covariates x samples orientation
            if covariates.shape[0] > covariates.shape[1]:
                covariates = covariates.T

            logger.info(f"Loaded covariates: {covariates.shape[0]} covariates")

        # Match samples across data types
        common_samples = (
            set(genotype_df.columns) &
            set(phenotype_df.columns)
        )
        if covariates is not None:
            common_samples &= set(covariates.columns)

        common_samples = sorted(common_samples)
        if len(common_samples) == 0:
            raise ValueError("No common samples found across data types")

        logger.info(f"Using {len(common_samples)} common samples")

        # Subset to common samples
        genotype_df = genotype_df[common_samples]
        phenotype_df = phenotype_df[common_samples]
        if covariates is not None:
            covariates = covariates[common_samples]

        return genotype_df, variant_df, phenotype_df, phenotype_pos, covariates

    def _load_vcf_genotypes(
        self,
        vcf_file: Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load genotypes from VCF file."""
        from eqtl_analysis.utils.io import read_vcf

        df = read_vcf(vcf_file, as_dataframe=True)

        # Separate metadata from genotypes
        meta_cols = ["chrom", "pos", "id", "ref", "alt"]
        sample_cols = [c for c in df.columns if c not in meta_cols]

        # Convert genotypes to dosage
        def parse_gt(gt: str) -> float:
            if pd.isna(gt) or gt in ("./.", ".|."):
                return np.nan
            parts = gt.replace("|", "/").split("/")
            try:
                return float(sum(int(a) for a in parts))
            except ValueError:
                return np.nan

        genotype_df = df[sample_cols].apply(lambda col: col.map(parse_gt))
        genotype_df.index = df["id"]

        variant_df = df[["chrom", "pos"]].copy()
        variant_df.index = df["id"]

        return genotype_df, variant_df

    def run_cis(
        self,
        genotype_file: str | Path,
        phenotype_file: str | Path,
        covariate_file: str | Path | None = None,
        output_prefix: str | None = None,
        mode: Literal["nominal", "permutation", "independent"] = "permutation",
    ) -> Path:
        """
        Run cis-eQTL mapping.

        Args:
            genotype_file: Path to genotype data.
            phenotype_file: Path to phenotype BED file.
            covariate_file: Path to covariate file.
            output_prefix: Prefix for output files.
            mode: Analysis mode (nominal, permutation, or independent).

        Returns:
            Path to results file.
        """
        import time

        import tensorqtl
        from tensorqtl import cis

        log_step("Cis-eQTL Analysis", logger)

        start_time = time.time()

        # Load data
        genotype_df, variant_df, phenotype_df, phenotype_pos, covariates = self.load_data(
            genotype_file, phenotype_file, covariate_file
        )

        if output_prefix is None:
            output_prefix = f"cis_{mode}"

        # Set seed for reproducibility
        np.random.seed(self.config.seed)

        # Run analysis based on mode
        if mode == "nominal":
            results = cis.map_nominal(
                genotype_df,
                variant_df,
                phenotype_df,
                phenotype_pos,
                covariates_df=covariates,
                window=self.config.window,
                maf_threshold=self.config.maf_threshold,
            )
            output_file = self.output_dir / f"{output_prefix}.parquet"
            results.to_parquet(output_file)

        elif mode == "permutation":
            results = cis.map_cis(
                genotype_df,
                variant_df,
                phenotype_df,
                phenotype_pos,
                covariates_df=covariates,
                window=self.config.window,
                maf_threshold=self.config.maf_threshold,
                nperm=self.config.permutations,
                seed=self.config.seed,
            )
            output_file = self.output_dir / f"{output_prefix}.txt.gz"
            results.to_csv(output_file, sep="\t", compression="gzip")

        elif mode == "independent":
            # First run permutation to get significant genes
            perm_results = cis.map_cis(
                genotype_df,
                variant_df,
                phenotype_df,
                phenotype_pos,
                covariates_df=covariates,
                window=self.config.window,
                maf_threshold=self.config.maf_threshold,
                nperm=self.config.permutations,
                seed=self.config.seed,
            )

            # Apply FDR correction
            tensorqtl.calculate_qvalues(perm_results, fdr=self.config.fdr_threshold)

            # Get significant genes
            sig_genes = perm_results[
                perm_results["qval"] <= self.config.fdr_threshold
            ]["phenotype_id"].tolist()

            if len(sig_genes) > 0:
                # Run independent eQTL mapping
                results = cis.map_independent(
                    genotype_df,
                    variant_df,
                    phenotype_df.loc[sig_genes],
                    phenotype_pos.loc[sig_genes],
                    perm_results.loc[sig_genes],
                    covariates_df=covariates,
                    window=self.config.window,
                    maf_threshold=self.config.maf_threshold,
                    nperm=self.config.permutations,
                    seed=self.config.seed,
                )
                output_file = self.output_dir / f"{output_prefix}.txt.gz"
                results.to_csv(output_file, sep="\t", compression="gzip")
            else:
                logger.warning("No significant genes found for independent analysis")
                output_file = self.output_dir / f"{output_prefix}_permutation.txt.gz"
                perm_results.to_csv(output_file, sep="\t", compression="gzip")
                results = perm_results

        else:
            raise ValueError(f"Unknown cis mode: {mode}")

        runtime = time.time() - start_time

        # Calculate statistics
        n_significant = 0
        n_genes_with_eqtl = 0

        if "qval" in results.columns:
            n_significant = (results["qval"] <= self.config.fdr_threshold).sum()
            n_genes_with_eqtl = results[
                results["qval"] <= self.config.fdr_threshold
            ]["phenotype_id"].nunique()
        elif "pval_beta" in results.columns:
            # Estimate from beta distribution p-values
            n_significant = (results["pval_beta"] <= 0.05).sum()

        self._run_stats = QTLRunStats(
            mode=f"cis_{mode}",
            n_phenotypes=len(phenotype_df),
            n_variants_tested=len(genotype_df),
            n_significant=n_significant,
            n_genes_with_eqtl=n_genes_with_eqtl,
            runtime_seconds=runtime,
        )

        logger.info(
            f"Cis-eQTL analysis complete: {n_significant} significant associations "
            f"({n_genes_with_eqtl} genes) in {runtime:.1f}s"
        )
        logger.info(f"Results saved to: {output_file}")

        return output_file

    def run_trans(
        self,
        genotype_file: str | Path,
        phenotype_file: str | Path,
        covariate_file: str | Path | None = None,
        output_prefix: str | None = None,
        batch_size: int = 50000,
        pval_threshold: float = 1e-5,
    ) -> Path:
        """
        Run trans-eQTL mapping.

        Args:
            genotype_file: Path to genotype data.
            phenotype_file: Path to phenotype BED file.
            covariate_file: Path to covariate file.
            output_prefix: Prefix for output files.
            batch_size: Number of variants per batch.
            pval_threshold: P-value threshold for output.

        Returns:
            Path to results file.
        """
        import time

        from tensorqtl import trans

        log_step("Trans-eQTL Analysis", logger)

        start_time = time.time()

        # Load data
        genotype_df, variant_df, phenotype_df, phenotype_pos, covariates = self.load_data(
            genotype_file, phenotype_file, covariate_file
        )

        if output_prefix is None:
            output_prefix = "trans"

        # Run trans mapping
        results = trans.map_trans(
            genotype_df,
            phenotype_df,
            covariates_df=covariates,
            batch_size=batch_size,
            pval_threshold=pval_threshold,
            maf_threshold=self.config.maf_threshold,
        )

        runtime = time.time() - start_time

        # Save results
        output_file = self.output_dir / f"{output_prefix}.txt.gz"
        results.to_csv(output_file, sep="\t", compression="gzip", index=False)

        self._run_stats = QTLRunStats(
            mode="trans",
            n_phenotypes=len(phenotype_df),
            n_variants_tested=len(genotype_df),
            n_significant=len(results),
            n_genes_with_eqtl=results["phenotype_id"].nunique() if len(results) > 0 else 0,
            runtime_seconds=runtime,
        )

        logger.info(
            f"Trans-eQTL analysis complete: {len(results)} associations "
            f"in {runtime:.1f}s"
        )
        logger.info(f"Results saved to: {output_file}")

        return output_file

    def run_interaction(
        self,
        genotype_file: str | Path,
        phenotype_file: str | Path,
        interaction_file: str | Path,
        covariate_file: str | Path | None = None,
        output_prefix: str | None = None,
    ) -> Path:
        """
        Run interaction eQTL (ieQTL) mapping.

        Args:
            genotype_file: Path to genotype data.
            phenotype_file: Path to phenotype BED file.
            interaction_file: Path to interaction term file.
            covariate_file: Path to covariate file.
            output_prefix: Prefix for output files.

        Returns:
            Path to results file.
        """
        import time

        from tensorqtl import cis

        log_step("Interaction eQTL Analysis", logger)

        start_time = time.time()

        # Load data
        genotype_df, variant_df, phenotype_df, phenotype_pos, covariates = self.load_data(
            genotype_file, phenotype_file, covariate_file
        )

        # Load interaction term
        validate_file_exists(interaction_file, "Interaction file")
        interaction_s = pd.read_csv(
            interaction_file,
            sep="\t",
            index_col=0,
            header=None,
            squeeze=True,
        )

        # Match samples
        common_samples = (
            set(genotype_df.columns) &
            set(phenotype_df.columns) &
            set(interaction_s.index)
        )
        common_samples = sorted(common_samples)

        genotype_df = genotype_df[common_samples]
        phenotype_df = phenotype_df[common_samples]
        interaction_s = interaction_s[common_samples]
        if covariates is not None:
            covariates = covariates[common_samples]

        if output_prefix is None:
            output_prefix = "interaction"

        # Run interaction mapping
        results = cis.map_nominal(
            genotype_df,
            variant_df,
            phenotype_df,
            phenotype_pos,
            covariates_df=covariates,
            interaction_s=interaction_s,
            window=self.config.window,
            maf_threshold=self.config.maf_threshold,
        )

        runtime = time.time() - start_time

        # Save results
        output_file = self.output_dir / f"{output_prefix}.parquet"
        results.to_parquet(output_file)

        self._run_stats = QTLRunStats(
            mode="interaction",
            n_phenotypes=len(phenotype_df),
            n_variants_tested=len(genotype_df),
            n_significant=0,  # Need to apply threshold
            n_genes_with_eqtl=0,
            runtime_seconds=runtime,
        )

        logger.info(
            f"Interaction eQTL analysis complete in {runtime:.1f}s"
        )
        logger.info(f"Results saved to: {output_file}")

        return output_file

    def run(
        self,
        genotype_file: str | Path,
        phenotype_file: str | Path,
        covariate_file: str | Path | None = None,
        mode: Literal["cis", "trans", "both"] = "cis",
        cis_mode: Literal["nominal", "permutation", "independent"] = "permutation",
    ) -> dict[str, Path]:
        """
        Run complete eQTL analysis.

        Args:
            genotype_file: Path to genotype data.
            phenotype_file: Path to phenotype BED file.
            covariate_file: Path to covariate file.
            mode: Analysis mode (cis, trans, or both).
            cis_mode: Cis analysis mode.

        Returns:
            Dictionary of output file paths.
        """
        results = {}

        if mode in ("cis", "both"):
            results["cis"] = self.run_cis(
                genotype_file,
                phenotype_file,
                covariate_file,
                mode=cis_mode,
            )

        if mode in ("trans", "both"):
            results["trans"] = self.run_trans(
                genotype_file,
                phenotype_file,
                covariate_file,
            )

        return results
