"""I/O utilities for the eQTL analysis pipeline."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np
import pandas as pd

from eqtl_analysis.utils.logging import get_logger
from eqtl_analysis.utils.validators import validate_file_exists

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class VCFReader:
    """Reader for VCF files with streaming support."""

    def __init__(self, vcf_path: str | Path) -> None:
        """
        Initialize VCF reader.

        Args:
            vcf_path: Path to VCF file.
        """
        self.vcf_path = Path(vcf_path)
        validate_file_exists(self.vcf_path, "VCF file")

        self._is_gzipped = str(self.vcf_path).endswith(".gz")
        self._header_lines: list[str] = []
        self._sample_names: list[str] = []
        self._parse_header()

    @property
    def samples(self) -> list[str]:
        """Get sample names from VCF."""
        return self._sample_names

    @property
    def header(self) -> list[str]:
        """Get header lines."""
        return self._header_lines

    def _parse_header(self) -> None:
        """Parse VCF header to extract metadata and sample names."""
        opener = gzip.open if self._is_gzipped else open

        with opener(self.vcf_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("##"):
                    self._header_lines.append(line)
                elif line.startswith("#CHROM"):
                    fields = line.split("\t")
                    if len(fields) > 9:
                        self._sample_names = fields[9:]
                    break

    def iter_variants(
        self,
        chrom: str | None = None,
        start: int | None = None,
        end: int | None = None,
    ) -> Iterator[dict[str, str | int | list[str]]]:
        """
        Iterate over variants in VCF.

        Args:
            chrom: Filter by chromosome.
            start: Filter by start position.
            end: Filter by end position.

        Yields:
            Dictionary with variant information.
        """
        opener = gzip.open if self._is_gzipped else open

        with opener(self.vcf_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue

                fields = line.split("\t")
                if len(fields) < 8:
                    continue

                variant_chrom = fields[0]
                variant_pos = int(fields[1])

                # Apply filters
                if chrom is not None and variant_chrom != chrom:
                    continue
                if start is not None and variant_pos < start:
                    continue
                if end is not None and variant_pos > end:
                    continue

                yield {
                    "chrom": variant_chrom,
                    "pos": variant_pos,
                    "id": fields[2],
                    "ref": fields[3],
                    "alt": fields[4],
                    "qual": fields[5],
                    "filter": fields[6],
                    "info": fields[7],
                    "format": fields[8] if len(fields) > 8 else "",
                    "genotypes": fields[9:] if len(fields) > 9 else [],
                }


def read_vcf(
    vcf_path: str | Path,
    sample_subset: list[str] | None = None,
    chrom: str | None = None,
    as_dataframe: bool = True,
) -> pd.DataFrame | VCFReader:
    """
    Read VCF file.

    Args:
        vcf_path: Path to VCF file.
        sample_subset: Subset of samples to read.
        chrom: Filter by chromosome.
        as_dataframe: If True, return as DataFrame.

    Returns:
        VCF data as DataFrame or VCFReader object.
    """
    reader = VCFReader(vcf_path)

    if not as_dataframe:
        return reader

    # Read into DataFrame
    records = []
    for variant in reader.iter_variants(chrom=chrom):
        record = {
            "chrom": variant["chrom"],
            "pos": variant["pos"],
            "id": variant["id"],
            "ref": variant["ref"],
            "alt": variant["alt"],
        }

        # Parse genotypes
        genotypes = variant["genotypes"]
        if isinstance(genotypes, list):
            for i, sample in enumerate(reader.samples):
                if sample_subset is None or sample in sample_subset:
                    if i < len(genotypes):
                        gt_field = genotypes[i].split(":")[0]
                        record[sample] = gt_field

        records.append(record)

    return pd.DataFrame(records)


def write_vcf(
    df: pd.DataFrame,
    output_path: str | Path,
    sample_columns: list[str] | None = None,
    header_lines: list[str] | None = None,
    compress: bool = True,
) -> Path:
    """
    Write VCF file from DataFrame.

    Args:
        df: DataFrame with variant data.
        output_path: Output file path.
        sample_columns: Columns containing sample genotypes.
        header_lines: Additional header lines.
        compress: Whether to gzip output.

    Returns:
        Path to written file.
    """
    output_path = Path(output_path)
    if compress and not str(output_path).endswith(".gz"):
        output_path = Path(str(output_path) + ".gz")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine sample columns
    required_cols = {"chrom", "pos", "id", "ref", "alt"}
    if sample_columns is None:
        sample_columns = [c for c in df.columns if c not in required_cols]

    # Build header
    header = [
        "##fileformat=VCFv4.2",
        f"##source=eqtl_analysis",
    ]
    if header_lines:
        header.extend(header_lines)

    # Column header
    base_cols = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
    column_line = "\t".join(base_cols + sample_columns)

    opener = gzip.open if compress else open

    with opener(output_path, "wt", encoding="utf-8") as f:
        # Write header
        for line in header:
            f.write(line + "\n")
        f.write(column_line + "\n")

        # Write variants
        for _, row in df.iterrows():
            fields = [
                str(row.get("chrom", ".")),
                str(row.get("pos", ".")),
                str(row.get("id", ".")),
                str(row.get("ref", ".")),
                str(row.get("alt", ".")),
                ".",  # QUAL
                "PASS",  # FILTER
                ".",  # INFO
                "GT",  # FORMAT
            ]

            # Add genotypes
            for sample in sample_columns:
                gt = row.get(sample, "./.")
                fields.append(str(gt) if pd.notna(gt) else "./.")

            f.write("\t".join(fields) + "\n")

    logger.info(f"Wrote VCF to {output_path}")
    return output_path


def read_expression_matrix(
    file_path: str | Path,
    gene_column: str | None = None,
    transpose: bool = False,
    sep: str | None = None,
) -> pd.DataFrame:
    """
    Read expression matrix from file.

    Args:
        file_path: Path to expression file.
        gene_column: Column name containing gene IDs.
        transpose: Whether to transpose the matrix.
        sep: Column separator. If None, auto-detect.

    Returns:
        Expression matrix as DataFrame (genes x samples).
    """
    file_path = Path(file_path)
    validate_file_exists(file_path, "Expression file")

    # Auto-detect separator
    if sep is None:
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            sep = ","
        elif suffix in (".tsv", ".txt"):
            sep = "\t"
        else:
            # Try to detect from first line
            opener = gzip.open if str(file_path).endswith(".gz") else open
            with opener(file_path, "rt", encoding="utf-8") as f:
                first_line = f.readline()
                sep = "\t" if "\t" in first_line else ","

    # Read file
    df = pd.read_csv(file_path, sep=sep, index_col=None)

    # Set gene IDs as index
    if gene_column is not None and gene_column in df.columns:
        df = df.set_index(gene_column)
    elif df.columns[0] in ("gene_id", "gene", "Gene", "GeneID", "gene_name"):
        df = df.set_index(df.columns[0])

    # Transpose if needed (ensure genes x samples)
    if transpose:
        df = df.T

    logger.info(f"Read expression matrix: {df.shape[0]} genes x {df.shape[1]} samples")
    return df


def write_expression_matrix(
    df: pd.DataFrame,
    output_path: str | Path,
    sep: str = "\t",
    compress: bool = False,
) -> Path:
    """
    Write expression matrix to file.

    Args:
        df: Expression matrix DataFrame.
        output_path: Output file path.
        sep: Column separator.
        compress: Whether to gzip output.

    Returns:
        Path to written file.
    """
    output_path = Path(output_path)
    if compress and not str(output_path).endswith(".gz"):
        output_path = Path(str(output_path) + ".gz")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, sep=sep, index=True)
    logger.info(f"Wrote expression matrix to {output_path}")
    return output_path


def read_bed_file(file_path: str | Path) -> pd.DataFrame:
    """
    Read BED file for gene annotations.

    Args:
        file_path: Path to BED file.

    Returns:
        DataFrame with gene annotations.
    """
    file_path = Path(file_path)
    validate_file_exists(file_path, "BED file")

    # BED files have specific column names
    columns = ["chr", "start", "end", "gene_id", "score", "strand"]

    opener = gzip.open if str(file_path).endswith(".gz") else open

    # Try reading with header first
    with opener(file_path, "rt", encoding="utf-8") as f:
        first_line = f.readline().strip()
        has_header = first_line.startswith("#") or first_line.startswith("chr\t")

    if has_header:
        df = pd.read_csv(file_path, sep="\t", comment="#")
        if len(df.columns) >= 6:
            df.columns = columns[: len(df.columns)]
    else:
        df = pd.read_csv(file_path, sep="\t", header=None, names=columns)

    logger.info(f"Read BED file with {len(df)} entries")
    return df


def read_sample_mapping(file_path: str | Path) -> dict[str, str]:
    """
    Read sample ID mapping file.

    Args:
        file_path: Path to mapping file (TSV with two columns).

    Returns:
        Dictionary mapping old IDs to new IDs.
    """
    file_path = Path(file_path)
    validate_file_exists(file_path, "Sample mapping file")

    df = pd.read_csv(file_path, sep="\t", header=None, names=["old_id", "new_id"])
    mapping = dict(zip(df["old_id"], df["new_id"]))

    logger.info(f"Read sample mapping with {len(mapping)} entries")
    return mapping


def ensure_directory(path: str | Path) -> Path:
    """
    Ensure directory exists, creating if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
