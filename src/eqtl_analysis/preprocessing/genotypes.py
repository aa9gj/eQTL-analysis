"""Genotype preprocessing module for eQTL analysis.

This module handles:
- Conversion of Axiom genotype TSV files to VCF format
- Coordinate liftover between genome assemblies
- Genotype imputation using Beagle
- Quality control filtering
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from eqtl_analysis.utils.config import (
    ImputationConfig,
    LiftoverConfig,
    QualityControlConfig,
)
from eqtl_analysis.utils.io import ensure_directory, write_vcf
from eqtl_analysis.utils.logging import get_logger
from eqtl_analysis.utils.validators import validate_file_exists, validate_vcf_format

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class GenotypeQCStats:
    """Statistics from genotype quality control."""

    total_variants: int
    variants_after_call_rate: int
    variants_after_maf: int
    variants_after_hwe: int
    final_variants: int
    total_samples: int
    samples_after_missing: int
    final_samples: int


class GenotypePreprocessor:
    """Preprocessor for genotype data in eQTL analysis pipeline."""

    def __init__(
        self,
        qc_config: QualityControlConfig | None = None,
        imputation_config: ImputationConfig | None = None,
        liftover_config: LiftoverConfig | None = None,
        output_dir: str | Path = "results/genotypes",
    ) -> None:
        """
        Initialize genotype preprocessor.

        Args:
            qc_config: Quality control configuration.
            imputation_config: Imputation configuration.
            liftover_config: Liftover configuration.
            output_dir: Output directory for processed files.
        """
        self.qc_config = qc_config or QualityControlConfig()
        self.imputation_config = imputation_config or ImputationConfig()
        self.liftover_config = liftover_config or LiftoverConfig()
        self.output_dir = ensure_directory(output_dir)
        self._qc_stats: GenotypeQCStats | None = None

    @property
    def qc_stats(self) -> GenotypeQCStats | None:
        """Get QC statistics from last run."""
        return self._qc_stats

    def convert_axiom_to_vcf(
        self,
        calls_file: str | Path,
        annotations_file: str | Path,
        output_path: str | Path | None = None,
        sample_mapping: dict[str, str] | None = None,
    ) -> Path:
        """
        Convert Axiom genotype TSV files to VCF format.

        This handles the conversion when bcftools affy2vcf is not available,
        working directly with the raw Axiom output files.

        Args:
            calls_file: Path to Axiom calls TSV file.
            annotations_file: Path to Axiom annotations TSV file.
            output_path: Output VCF path. If None, auto-generated.
            sample_mapping: Optional mapping of sample IDs.

        Returns:
            Path to the generated VCF file.

        Raises:
            FileNotFoundError: If input files don't exist.
            ValueError: If file format is invalid.
        """
        calls_file = Path(calls_file)
        annotations_file = Path(annotations_file)

        validate_file_exists(calls_file, "Axiom calls file")
        validate_file_exists(annotations_file, "Axiom annotations file")

        logger.info(f"Converting Axiom files to VCF: {calls_file.name}")

        # Read annotations for SNP metadata
        annotations = self._read_axiom_annotations(annotations_file)
        logger.info(f"Loaded {len(annotations)} SNP annotations")

        # Read genotype calls
        calls_df = self._read_axiom_calls(calls_file)
        logger.info(f"Loaded calls for {len(calls_df.columns)} samples")

        # Apply sample mapping if provided
        if sample_mapping:
            calls_df.columns = [sample_mapping.get(c, c) for c in calls_df.columns]

        # Merge annotations with calls
        merged = annotations.join(calls_df, how="inner")

        if len(merged) == 0:
            raise ValueError("No matching SNPs between annotations and calls files")

        logger.info(f"Merged {len(merged)} variants")

        # Convert to VCF DataFrame format
        vcf_df = self._create_vcf_dataframe(merged)

        # Write VCF
        if output_path is None:
            output_path = self.output_dir / "genotypes.vcf.gz"

        output_path = write_vcf(
            vcf_df,
            output_path,
            sample_columns=[c for c in vcf_df.columns if c not in
                           {"chrom", "pos", "id", "ref", "alt"}],
        )

        logger.info(f"Created VCF with {len(vcf_df)} variants")
        return output_path

    def _read_axiom_annotations(self, file_path: Path) -> pd.DataFrame:
        """Read Axiom annotation file and extract SNP metadata."""
        # Try to auto-detect format
        df = pd.read_csv(file_path, sep="\t", comment="#")

        # Map common column names
        column_mapping = {
            "Probe Set ID": "probe_id",
            "ProbeSetID": "probe_id",
            "probeset_id": "probe_id",
            "Chromosome": "chrom",
            "Chr": "chrom",
            "Physical Position": "pos",
            "Position": "pos",
            "Strand": "strand",
            "Ref Allele": "ref",
            "Reference Allele": "ref",
            "Alt Allele": "alt",
            "Alternate Allele": "alt",
            "dbSNP RS ID": "id",
            "rsID": "id",
        }

        df = df.rename(columns=column_mapping)

        # Ensure required columns exist
        required = {"probe_id", "chrom", "pos", "ref", "alt"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in annotations: {missing}")

        # Set probe ID as index
        df = df.set_index("probe_id")

        # Add ID column if not present
        if "id" not in df.columns:
            df["id"] = df.index

        return df[["chrom", "pos", "ref", "alt", "id"]]

    def _read_axiom_calls(self, file_path: Path) -> pd.DataFrame:
        """Read Axiom calls file."""
        df = pd.read_csv(file_path, sep="\t", index_col=0)

        # Convert call codes to genotypes
        # Axiom uses: 0=AA, 1=AB, 2=BB, -1=NoCall
        call_mapping = {0: "0/0", 1: "0/1", 2: "1/1", -1: "./.", "NoCall": "./."}

        for col in df.columns:
            df[col] = df[col].map(lambda x: call_mapping.get(x, "./."))

        return df

    def _create_vcf_dataframe(self, merged: pd.DataFrame) -> pd.DataFrame:
        """Create VCF-formatted DataFrame from merged data."""
        # Get sample columns (everything except annotation columns)
        annotation_cols = {"chrom", "pos", "ref", "alt", "id"}
        sample_cols = [c for c in merged.columns if c not in annotation_cols]

        # Build VCF DataFrame
        vcf_df = pd.DataFrame({
            "chrom": merged["chrom"],
            "pos": merged["pos"].astype(int),
            "id": merged["id"],
            "ref": merged["ref"],
            "alt": merged["alt"],
        })

        # Add genotype columns
        for sample in sample_cols:
            vcf_df[sample] = merged[sample]

        # Sort by chromosome and position
        vcf_df = vcf_df.sort_values(["chrom", "pos"]).reset_index(drop=True)

        return vcf_df

    def liftover(
        self,
        vcf_path: str | Path,
        output_path: str | Path | None = None,
        chain_file: str | Path | None = None,
    ) -> Path:
        """
        Perform coordinate liftover to target genome assembly.

        Args:
            vcf_path: Input VCF file path.
            output_path: Output VCF path. If None, auto-generated.
            chain_file: Path to chain file. Uses config if not provided.

        Returns:
            Path to lifted-over VCF file.

        Raises:
            RuntimeError: If liftover fails.
        """
        vcf_path = Path(vcf_path)
        validate_file_exists(vcf_path, "VCF file for liftover")

        chain_file = chain_file or self.liftover_config.chain_file
        if not chain_file:
            raise ValueError("Chain file required for liftover")

        validate_file_exists(chain_file, "Chain file")

        logger.info(
            f"Lifting over from {self.liftover_config.source_assembly} "
            f"to {self.liftover_config.target_assembly}"
        )

        if output_path is None:
            output_path = self.output_dir / f"genotypes_{self.liftover_config.target_assembly}.vcf.gz"

        # Check for picard or CrossMap
        liftover_tool = self._detect_liftover_tool()

        if liftover_tool == "picard":
            return self._liftover_picard(vcf_path, output_path, Path(chain_file))
        elif liftover_tool == "crossmap":
            return self._liftover_crossmap(vcf_path, output_path, Path(chain_file))
        else:
            return self._liftover_manual(vcf_path, output_path, Path(chain_file))

    def _detect_liftover_tool(self) -> str:
        """Detect available liftover tool."""
        tools = [
            ("picard", ["picard", "--version"]),
            ("crossmap", ["CrossMap.py", "--version"]),
        ]

        for name, cmd in tools:
            try:
                subprocess.run(cmd, capture_output=True, check=False)
                return name
            except FileNotFoundError:
                continue

        logger.warning("No external liftover tool found, using manual implementation")
        return "manual"

    def _liftover_picard(
        self,
        vcf_path: Path,
        output_path: Path,
        chain_file: Path,
    ) -> Path:
        """Perform liftover using Picard LiftoverVcf."""
        reject_path = output_path.parent / f"{output_path.stem}_rejected.vcf"

        cmd = [
            "picard", "LiftoverVcf",
            f"I={vcf_path}",
            f"O={output_path}",
            f"CHAIN={chain_file}",
            f"REJECT={reject_path}",
        ]

        if self.liftover_config.min_match:
            cmd.append(f"LIFTOVER_MIN_MATCH={self.liftover_config.min_match}")

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Picard liftover failed: {result.stderr}")

        logger.info(f"Liftover complete: {output_path}")
        return Path(output_path)

    def _liftover_crossmap(
        self,
        vcf_path: Path,
        output_path: Path,
        chain_file: Path,
    ) -> Path:
        """Perform liftover using CrossMap."""
        cmd = [
            "CrossMap.py", "vcf",
            str(chain_file),
            str(vcf_path),
            str(self.liftover_config.target_assembly),  # Reference genome
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"CrossMap liftover failed: {result.stderr}")

        logger.info(f"Liftover complete: {output_path}")
        return Path(output_path)

    def _liftover_manual(
        self,
        vcf_path: Path,
        output_path: Path,
        chain_file: Path,
    ) -> Path:
        """
        Manual liftover implementation using chain file.

        This is a fallback when external tools are not available.
        """
        # Parse chain file
        chains = self._parse_chain_file(chain_file)

        # Read and transform VCF
        from eqtl_analysis.utils.io import VCFReader

        reader = VCFReader(vcf_path)
        transformed_records = []
        unmapped_count = 0

        for variant in reader.iter_variants():
            chrom = str(variant["chrom"])
            pos = int(variant["pos"])

            # Find mapping
            new_chrom, new_pos = self._map_coordinates(chains, chrom, pos)

            if new_chrom is not None and new_pos is not None:
                variant["chrom"] = new_chrom
                variant["pos"] = new_pos
                transformed_records.append(variant)
            else:
                unmapped_count += 1

        logger.info(
            f"Liftover: {len(transformed_records)} mapped, {unmapped_count} unmapped"
        )

        # Create DataFrame and write
        df = pd.DataFrame([
            {
                "chrom": r["chrom"],
                "pos": r["pos"],
                "id": r["id"],
                "ref": r["ref"],
                "alt": r["alt"],
                **{s: g for s, g in zip(reader.samples, r["genotypes"])},
            }
            for r in transformed_records
        ])

        return write_vcf(df, output_path, sample_columns=reader.samples)

    def _parse_chain_file(self, chain_file: Path) -> list[dict]:
        """Parse UCSC chain file format."""
        chains = []
        current_chain = None

        with open(chain_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("chain"):
                    parts = line.split()
                    current_chain = {
                        "score": int(parts[1]),
                        "source_chrom": parts[2],
                        "source_size": int(parts[3]),
                        "source_strand": parts[4],
                        "source_start": int(parts[5]),
                        "source_end": int(parts[6]),
                        "target_chrom": parts[7],
                        "target_size": int(parts[8]),
                        "target_strand": parts[9],
                        "target_start": int(parts[10]),
                        "target_end": int(parts[11]),
                        "blocks": [],
                    }
                    chains.append(current_chain)
                elif current_chain is not None and line[0].isdigit():
                    parts = line.split()
                    block = {"size": int(parts[0])}
                    if len(parts) > 1:
                        block["dt"] = int(parts[1])
                        block["dq"] = int(parts[2])
                    current_chain["blocks"].append(block)

        return chains

    def _map_coordinates(
        self,
        chains: list[dict],
        chrom: str,
        pos: int,
    ) -> tuple[str | None, int | None]:
        """Map coordinates using chain data."""
        for chain in chains:
            if chain["source_chrom"] != chrom:
                continue
            if pos < chain["source_start"] or pos >= chain["source_end"]:
                continue

            # Calculate offset within chain
            source_offset = pos - chain["source_start"]
            target_offset = 0
            current_source = 0

            for block in chain["blocks"]:
                if current_source + block["size"] > source_offset:
                    # Position is within this block
                    block_offset = source_offset - current_source
                    target_pos = chain["target_start"] + target_offset + block_offset
                    return chain["target_chrom"], target_pos

                current_source += block["size"] + block.get("dt", 0)
                target_offset += block["size"] + block.get("dq", 0)

        return None, None

    def run_imputation(
        self,
        vcf_path: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Run genotype imputation using Beagle.

        Args:
            vcf_path: Input VCF file path.
            output_path: Output path prefix.

        Returns:
            Path to imputed VCF file.

        Raises:
            RuntimeError: If Beagle is not available or fails.
        """
        vcf_path = Path(vcf_path)
        validate_file_exists(vcf_path, "VCF file for imputation")

        if output_path is None:
            output_path = self.output_dir / "genotypes_imputed"
        else:
            output_path = Path(output_path)

        logger.info("Running Beagle imputation")

        # Build Beagle command
        beagle_jar = self.imputation_config.beagle_jar_path
        if not beagle_jar:
            # Try to find Beagle in PATH
            beagle_jar = self._find_beagle()

        if not beagle_jar:
            raise RuntimeError(
                "Beagle not found. Please set beagle_jar_path in configuration."
            )

        cmd = [
            "java",
            f"-Xmx{self.imputation_config.memory_gb}g",
            "-jar", str(beagle_jar),
            f"gt={vcf_path}",
            f"out={output_path}",
            f"nthreads={self.imputation_config.nthreads}",
            f"window={self.imputation_config.window_size}",
            f"overlap={self.imputation_config.overlap}",
            f"ne={self.imputation_config.ne}",
        ]

        # Add reference panel if provided
        if self.imputation_config.reference_panel:
            cmd.append(f"ref={self.imputation_config.reference_panel}")

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Beagle imputation failed: {result.stderr}")

        output_vcf = Path(str(output_path) + ".vcf.gz")
        logger.info(f"Imputation complete: {output_vcf}")
        return output_vcf

    def _find_beagle(self) -> str | None:
        """Try to find Beagle JAR in common locations."""
        common_paths = [
            Path.home() / "beagle.jar",
            Path("/usr/local/share/beagle/beagle.jar"),
            Path("/opt/beagle/beagle.jar"),
        ]

        for path in common_paths:
            if path.exists():
                return str(path)

        return None

    def run_quality_control(
        self,
        vcf_path: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Run quality control filtering on genotype data.

        Applies filters for:
        - Variant call rate
        - Minor allele frequency (MAF)
        - Hardy-Weinberg equilibrium
        - Sample missing rate

        Args:
            vcf_path: Input VCF file path.
            output_path: Output VCF path.

        Returns:
            Path to QC-filtered VCF file.
        """
        vcf_path = Path(vcf_path)
        validate_file_exists(vcf_path, "VCF file for QC")

        logger.info("Running genotype quality control")

        # Read VCF
        from eqtl_analysis.utils.io import read_vcf

        df = read_vcf(vcf_path, as_dataframe=True)

        # Get sample columns
        meta_cols = {"chrom", "pos", "id", "ref", "alt"}
        sample_cols = [c for c in df.columns if c not in meta_cols]

        # Convert genotypes to numeric for calculations
        geno_matrix = self._genotypes_to_numeric(df[sample_cols])

        initial_variants = len(df)
        initial_samples = len(sample_cols)

        # 1. Filter by variant call rate
        variant_call_rate = geno_matrix.notna().mean(axis=1)
        mask_call_rate = variant_call_rate >= self.qc_config.min_call_rate
        df = df[mask_call_rate]
        geno_matrix = geno_matrix[mask_call_rate]
        variants_after_call_rate = len(df)

        logger.info(
            f"Call rate filter: {initial_variants} -> {variants_after_call_rate} "
            f"(min rate: {self.qc_config.min_call_rate})"
        )

        # 2. Filter by MAF
        maf = self._calculate_maf(geno_matrix)
        mask_maf = maf >= self.qc_config.min_maf
        df = df[mask_maf]
        geno_matrix = geno_matrix[mask_maf]
        variants_after_maf = len(df)

        logger.info(
            f"MAF filter: {variants_after_call_rate} -> {variants_after_maf} "
            f"(min MAF: {self.qc_config.min_maf})"
        )

        # 3. Filter by HWE
        hwe_pvalues = self._calculate_hwe(geno_matrix)
        mask_hwe = hwe_pvalues >= self.qc_config.hwe_pvalue_threshold
        df = df[mask_hwe]
        geno_matrix = geno_matrix[mask_hwe]
        variants_after_hwe = len(df)

        logger.info(
            f"HWE filter: {variants_after_maf} -> {variants_after_hwe} "
            f"(min p-value: {self.qc_config.hwe_pvalue_threshold})"
        )

        # 4. Filter samples by missing rate
        sample_missing_rate = geno_matrix.isna().mean(axis=0)
        good_samples = sample_missing_rate[
            sample_missing_rate <= self.qc_config.max_missing_rate
        ].index.tolist()

        samples_after_missing = len(good_samples)
        logger.info(
            f"Sample filter: {initial_samples} -> {samples_after_missing} "
            f"(max missing: {self.qc_config.max_missing_rate})"
        )

        # Store QC stats
        self._qc_stats = GenotypeQCStats(
            total_variants=initial_variants,
            variants_after_call_rate=variants_after_call_rate,
            variants_after_maf=variants_after_maf,
            variants_after_hwe=variants_after_hwe,
            final_variants=variants_after_hwe,
            total_samples=initial_samples,
            samples_after_missing=samples_after_missing,
            final_samples=samples_after_missing,
        )

        # Keep only good samples
        final_cols = list(meta_cols & set(df.columns)) + good_samples
        df = df[[c for c in df.columns if c in final_cols]]

        # Write output
        if output_path is None:
            output_path = self.output_dir / "genotypes_qc.vcf.gz"

        output_path = write_vcf(df, output_path, sample_columns=good_samples)

        logger.info(f"QC complete: {output_path}")
        return output_path

    def _genotypes_to_numeric(self, geno_df: pd.DataFrame) -> pd.DataFrame:
        """Convert genotype strings to numeric dosage values."""
        def parse_gt(gt: str) -> float | None:
            if pd.isna(gt) or gt in ("./.", ".|."):
                return None
            parts = gt.replace("|", "/").split("/")
            try:
                return sum(int(a) for a in parts)
            except ValueError:
                return None

        return geno_df.apply(lambda col: col.map(parse_gt))

    def _calculate_maf(self, geno_matrix: pd.DataFrame) -> pd.Series:
        """Calculate minor allele frequency for each variant."""
        allele_freq = geno_matrix.mean(axis=1) / 2  # Divide by 2 for diploid
        maf = allele_freq.apply(lambda x: min(x, 1 - x) if pd.notna(x) else 0)
        return maf

    def _calculate_hwe(self, geno_matrix: pd.DataFrame) -> pd.Series:
        """Calculate Hardy-Weinberg equilibrium p-values."""
        from scipy import stats

        pvalues = []

        for _, row in geno_matrix.iterrows():
            valid = row.dropna()
            if len(valid) < 10:
                pvalues.append(1.0)
                continue

            # Count genotypes
            n_aa = (valid == 0).sum()
            n_ab = (valid == 1).sum()
            n_bb = (valid == 2).sum()
            n = n_aa + n_ab + n_bb

            if n == 0:
                pvalues.append(1.0)
                continue

            # Calculate allele frequencies
            p = (2 * n_aa + n_ab) / (2 * n)
            q = 1 - p

            # Expected counts under HWE
            exp_aa = n * p * p
            exp_ab = n * 2 * p * q
            exp_bb = n * q * q

            # Chi-square test
            observed = [n_aa, n_ab, n_bb]
            expected = [exp_aa, exp_ab, exp_bb]

            # Avoid division by zero
            if any(e == 0 for e in expected):
                pvalues.append(1.0)
                continue

            chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
            pvalue = 1 - stats.chi2.cdf(chi2, df=1)
            pvalues.append(pvalue)

        return pd.Series(pvalues, index=geno_matrix.index)

    def preprocess(
        self,
        calls_file: str | Path | None = None,
        annotations_file: str | Path | None = None,
        vcf_file: str | Path | None = None,
        run_liftover: bool = True,
        run_imputation: bool = True,
        run_qc: bool = True,
    ) -> Path:
        """
        Run complete genotype preprocessing pipeline.

        Args:
            calls_file: Axiom calls TSV file.
            annotations_file: Axiom annotations TSV file.
            vcf_file: Existing VCF file (alternative to Axiom files).
            run_liftover: Whether to perform liftover.
            run_imputation: Whether to run imputation.
            run_qc: Whether to run quality control.

        Returns:
            Path to final processed VCF file.
        """
        logger.info("Starting genotype preprocessing pipeline")

        # Step 1: Get or create VCF
        if vcf_file:
            current_vcf = Path(vcf_file)
            validate_file_exists(current_vcf, "Input VCF")
        elif calls_file and annotations_file:
            current_vcf = self.convert_axiom_to_vcf(calls_file, annotations_file)
        else:
            raise ValueError("Either vcf_file or both calls_file and annotations_file required")

        # Step 2: Liftover
        if run_liftover and self.liftover_config.chain_file:
            current_vcf = self.liftover(current_vcf)

        # Step 3: Quality control (before imputation)
        if run_qc:
            current_vcf = self.run_quality_control(current_vcf)

        # Step 4: Imputation
        if run_imputation:
            current_vcf = self.run_imputation(current_vcf)

        logger.info(f"Genotype preprocessing complete: {current_vcf}")
        return current_vcf
