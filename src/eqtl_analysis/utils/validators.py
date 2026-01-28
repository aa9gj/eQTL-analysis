"""Data validation utilities for the eQTL analysis pipeline."""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from eqtl_analysis.utils.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)


class ValidationError(Exception):
    """Raised when data validation fails."""

    pass


def validate_file_exists(
    file_path: str | Path,
    description: str = "file",
    raise_error: bool = True,
) -> bool:
    """
    Validate that a file exists.

    Args:
        file_path: Path to the file.
        description: Description of the file for error messages.
        raise_error: If True, raise an error on failure.

    Returns:
        True if file exists.

    Raises:
        FileNotFoundError: If file does not exist and raise_error is True.
    """
    path = Path(file_path)
    if not path.exists():
        msg = f"{description} not found: {file_path}"
        if raise_error:
            raise FileNotFoundError(msg)
        logger.warning(msg)
        return False
    return True


def validate_vcf_format(
    vcf_path: str | Path,
    check_samples: bool = True,
    check_variants: bool = True,
    max_lines_to_check: int = 100,
) -> dict[str, bool | int | list[str]]:
    """
    Validate VCF file format.

    Args:
        vcf_path: Path to the VCF file.
        check_samples: Whether to validate sample information.
        check_variants: Whether to validate variant records.
        max_lines_to_check: Maximum variant lines to validate.

    Returns:
        Dictionary with validation results.

    Raises:
        ValidationError: If critical validation fails.
    """
    vcf_path = Path(vcf_path)
    validate_file_exists(vcf_path, "VCF file")

    results: dict[str, bool | int | list[str]] = {
        "valid_header": False,
        "valid_format": False,
        "has_samples": False,
        "sample_count": 0,
        "samples": [],
        "variant_count": 0,
        "issues": [],
    }

    # Determine if gzipped
    opener = gzip.open if str(vcf_path).endswith(".gz") else open

    try:
        with opener(vcf_path, "rt", encoding="utf-8") as f:
            header_lines = []
            sample_line = None
            variant_count = 0

            for line in f:
                line = line.strip()

                if line.startswith("##"):
                    header_lines.append(line)
                elif line.startswith("#CHROM"):
                    sample_line = line
                    results["valid_header"] = True
                else:
                    # Variant line
                    if check_variants and variant_count < max_lines_to_check:
                        is_valid, issue = _validate_variant_line(line, sample_line)
                        if not is_valid and issue:
                            issues = results["issues"]
                            if isinstance(issues, list):
                                issues.append(f"Line {variant_count + 1}: {issue}")
                    variant_count += 1

            results["variant_count"] = variant_count

            # Validate sample line
            if sample_line and check_samples:
                fields = sample_line.split("\t")
                if len(fields) >= 9:
                    results["has_samples"] = len(fields) > 9
                    samples = fields[9:]
                    results["sample_count"] = len(samples)
                    results["samples"] = samples
                    results["valid_format"] = True

    except Exception as e:
        raise ValidationError(f"Failed to read VCF file: {e}") from e

    # Check for critical issues
    if not results["valid_header"]:
        raise ValidationError("VCF file missing required header line starting with #CHROM")

    return results


def _validate_variant_line(line: str, sample_line: str | None) -> tuple[bool, str | None]:
    """
    Validate a single variant line in VCF.

    Args:
        line: Variant line from VCF.
        sample_line: Header line with sample names.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not line:
        return True, None

    fields = line.split("\t")

    # Basic field count check
    if len(fields) < 8:
        return False, "Insufficient fields (minimum 8 required)"

    # Validate chromosome
    chrom = fields[0]
    if not chrom:
        return False, "Empty chromosome field"

    # Validate position
    try:
        pos = int(fields[1])
        if pos < 1:
            return False, f"Invalid position: {pos}"
    except ValueError:
        return False, f"Non-integer position: {fields[1]}"

    # Validate reference allele
    ref = fields[3]
    if not re.match(r"^[ACGTN]+$", ref, re.IGNORECASE):
        return False, f"Invalid reference allele: {ref}"

    # Validate alternate allele
    alt = fields[4]
    if alt != "." and not re.match(r"^[ACGTN,*]+$", alt, re.IGNORECASE):
        return False, f"Invalid alternate allele: {alt}"

    # Check sample count matches header
    if sample_line:
        expected_fields = len(sample_line.split("\t"))
        if len(fields) != expected_fields:
            return False, f"Field count mismatch: got {len(fields)}, expected {expected_fields}"

    return True, None


def validate_expression_matrix(
    expression_data: pd.DataFrame,
    require_gene_ids: bool = True,
    min_genes: int = 1,
    min_samples: int = 2,
    check_values: bool = True,
) -> dict[str, bool | int | list[str]]:
    """
    Validate expression matrix format and content.

    Args:
        expression_data: Expression matrix as DataFrame.
        require_gene_ids: Whether gene IDs are required in index.
        min_genes: Minimum number of genes required.
        min_samples: Minimum number of samples required.
        check_values: Whether to validate expression values.

    Returns:
        Dictionary with validation results.

    Raises:
        ValidationError: If critical validation fails.
    """
    results: dict[str, bool | int | list[str]] = {
        "valid": True,
        "n_genes": expression_data.shape[0],
        "n_samples": expression_data.shape[1],
        "has_gene_ids": False,
        "has_missing_values": False,
        "has_negative_values": False,
        "issues": [],
    }

    issues: list[str] = []

    # Check dimensions
    if expression_data.shape[0] < min_genes:
        issues.append(f"Insufficient genes: {expression_data.shape[0]} < {min_genes}")

    if expression_data.shape[1] < min_samples:
        issues.append(f"Insufficient samples: {expression_data.shape[1]} < {min_samples}")

    # Check gene IDs
    if require_gene_ids:
        if expression_data.index.name is None and not expression_data.index.is_unique:
            issues.append("Gene IDs not unique")
        else:
            results["has_gene_ids"] = True

    # Check values
    if check_values:
        # Check for missing values
        missing_count = expression_data.isna().sum().sum()
        if missing_count > 0:
            results["has_missing_values"] = True
            missing_pct = (missing_count / expression_data.size) * 100
            issues.append(f"Contains {missing_count} missing values ({missing_pct:.2f}%)")

        # Check for negative values (unusual for expression data)
        numeric_data = expression_data.select_dtypes(include=[np.number])
        if (numeric_data < 0).any().any():
            results["has_negative_values"] = True
            issues.append("Contains negative values")

        # Check for infinite values
        if np.isinf(numeric_data.values).any():
            issues.append("Contains infinite values")

    results["issues"] = issues
    results["valid"] = len(issues) == 0

    if not results["valid"]:
        logger.warning(f"Expression matrix validation issues: {issues}")

    return results


def validate_covariate_matrix(
    covariates: pd.DataFrame,
    sample_ids: list[str] | None = None,
    max_covariates: int | None = None,
) -> dict[str, bool | int | list[str]]:
    """
    Validate covariate matrix format.

    Args:
        covariates: Covariate matrix as DataFrame.
        sample_ids: Expected sample IDs to match.
        max_covariates: Maximum allowed number of covariates.

    Returns:
        Dictionary with validation results.
    """
    results: dict[str, bool | int | list[str]] = {
        "valid": True,
        "n_covariates": covariates.shape[0],
        "n_samples": covariates.shape[1],
        "issues": [],
    }

    issues: list[str] = []

    # Check sample ID match
    if sample_ids is not None:
        covariate_samples = set(covariates.columns)
        expected_samples = set(sample_ids)
        missing = expected_samples - covariate_samples
        extra = covariate_samples - expected_samples

        if missing:
            issues.append(f"Missing samples in covariates: {len(missing)}")
        if extra:
            issues.append(f"Extra samples in covariates: {len(extra)}")

    # Check covariate count
    if max_covariates is not None and covariates.shape[0] > max_covariates:
        issues.append(
            f"Too many covariates: {covariates.shape[0]} > {max_covariates}"
        )

    # Check for constant covariates
    for cov_name in covariates.index:
        if covariates.loc[cov_name].nunique() <= 1:
            issues.append(f"Constant covariate: {cov_name}")

    results["issues"] = issues
    results["valid"] = len(issues) == 0

    return results


def validate_sample_consistency(
    genotype_samples: list[str],
    expression_samples: list[str],
    covariate_samples: list[str] | None = None,
) -> dict[str, set[str]]:
    """
    Validate sample consistency across data types.

    Args:
        genotype_samples: Sample IDs from genotype data.
        expression_samples: Sample IDs from expression data.
        covariate_samples: Sample IDs from covariate data.

    Returns:
        Dictionary with sample overlap information.

    Raises:
        ValidationError: If no common samples exist.
    """
    geno_set = set(genotype_samples)
    expr_set = set(expression_samples)

    common = geno_set & expr_set
    geno_only = geno_set - expr_set
    expr_only = expr_set - geno_set

    if covariate_samples is not None:
        cov_set = set(covariate_samples)
        common = common & cov_set
        cov_only = cov_set - geno_set - expr_set
    else:
        cov_only = set()

    if len(common) == 0:
        raise ValidationError(
            "No common samples found between genotype and expression data. "
            f"Genotype samples: {len(geno_set)}, Expression samples: {len(expr_set)}"
        )

    logger.info(
        f"Sample overlap: {len(common)} common, "
        f"{len(geno_only)} genotype-only, {len(expr_only)} expression-only"
    )

    return {
        "common": common,
        "genotype_only": geno_only,
        "expression_only": expr_only,
        "covariate_only": cov_only,
    }
