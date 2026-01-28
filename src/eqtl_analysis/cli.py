"""Command-line interface for the eQTL analysis pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from eqtl_analysis import __version__
from eqtl_analysis.utils.config import Config, load_config
from eqtl_analysis.utils.logging import setup_logging, get_logger

console = Console()
logger = get_logger(__name__)


def print_banner() -> None:
    """Print application banner."""
    console.print(
        "\n[bold blue]eQTL Analysis Pipeline[/bold blue] "
        f"[dim]v{__version__}[/dim]\n"
    )


@click.group()
@click.version_option(version=__version__, prog_name="eqtl-analysis")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="Path to configuration file.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output.",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Suppress non-error output.",
)
@click.option(
    "--log-file",
    type=click.Path(),
    help="Path to log file.",
)
@click.pass_context
def main(
    ctx: click.Context,
    config: Optional[str],
    verbose: bool,
    quiet: bool,
    log_file: Optional[str],
) -> None:
    """
    eQTL Analysis Pipeline.

    A production-grade pipeline for Expression Quantitative Trait Loci (eQTL)
    identification, designed for analyzing genetic variants affecting gene expression.
    """
    ctx.ensure_object(dict)

    # Set up logging
    log_level = "DEBUG" if verbose else "WARNING" if quiet else "INFO"
    setup_logging(level=log_level, log_file=log_file)

    # Load configuration
    ctx.obj["config"] = load_config(config)
    ctx.obj["verbose"] = verbose

    if not quiet:
        print_banner()


@main.command()
@click.option(
    "--calls", "-c",
    type=click.Path(exists=True),
    help="Axiom genotype calls file.",
)
@click.option(
    "--annotations", "-a",
    type=click.Path(exists=True),
    help="Axiom annotations file.",
)
@click.option(
    "--vcf", "-v",
    type=click.Path(exists=True),
    help="Existing VCF file (alternative to Axiom files).",
)
@click.option(
    "--chain-file",
    type=click.Path(exists=True),
    help="Chain file for liftover.",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="results/genotypes",
    help="Output directory.",
)
@click.option(
    "--skip-liftover",
    is_flag=True,
    help="Skip liftover step.",
)
@click.option(
    "--skip-imputation",
    is_flag=True,
    help="Skip imputation step.",
)
@click.option(
    "--skip-qc",
    is_flag=True,
    help="Skip quality control step.",
)
@click.pass_context
def preprocess_genotypes(
    ctx: click.Context,
    calls: Optional[str],
    annotations: Optional[str],
    vcf: Optional[str],
    chain_file: Optional[str],
    output_dir: str,
    skip_liftover: bool,
    skip_imputation: bool,
    skip_qc: bool,
) -> None:
    """
    Preprocess genotype data.

    Convert Axiom TSV files to VCF, perform liftover, QC, and imputation.
    """
    from eqtl_analysis.preprocessing import GenotypePreprocessor

    config = ctx.obj["config"]

    # Update config with chain file if provided
    if chain_file:
        config.liftover.chain_file = chain_file

    preprocessor = GenotypePreprocessor(
        qc_config=config.qc,
        imputation_config=config.imputation,
        liftover_config=config.liftover,
        output_dir=output_dir,
    )

    try:
        output_path = preprocessor.preprocess(
            calls_file=calls,
            annotations_file=annotations,
            vcf_file=vcf,
            run_liftover=not skip_liftover,
            run_imputation=not skip_imputation,
            run_qc=not skip_qc,
        )

        # Print QC stats
        if preprocessor.qc_stats:
            stats = preprocessor.qc_stats
            table = Table(title="Genotype QC Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")

            table.add_row("Initial variants", f"{stats.total_variants:,}")
            table.add_row("After call rate filter", f"{stats.variants_after_call_rate:,}")
            table.add_row("After MAF filter", f"{stats.variants_after_maf:,}")
            table.add_row("After HWE filter", f"{stats.variants_after_hwe:,}")
            table.add_row("Final variants", f"{stats.final_variants:,}")
            table.add_row("Initial samples", f"{stats.total_samples:,}")
            table.add_row("Final samples", f"{stats.final_samples:,}")

            console.print(table)

        console.print(f"\n[green]Output:[/green] {output_path}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.option(
    "--expression", "-e",
    type=click.Path(exists=True),
    required=True,
    help="Expression data file.",
)
@click.option(
    "--gene-annotation", "-g",
    type=click.Path(exists=True),
    help="Gene annotation file.",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="results/phenotypes",
    help="Output directory.",
)
@click.option(
    "--normalization",
    type=click.Choice(["tmm", "quantile", "inverse_normal"]),
    default="inverse_normal",
    help="Normalization method.",
)
@click.option(
    "--min-expression",
    type=float,
    default=1.0,
    help="Minimum expression threshold.",
)
@click.option(
    "--remove-outliers/--keep-outliers",
    default=True,
    help="Remove outlier samples.",
)
@click.pass_context
def preprocess_phenotypes(
    ctx: click.Context,
    expression: str,
    gene_annotation: Optional[str],
    output_dir: str,
    normalization: str,
    min_expression: float,
    remove_outliers: bool,
) -> None:
    """
    Preprocess phenotype (expression) data.

    Filter lowly expressed genes, normalize, and prepare for tensorQTL.
    """
    from eqtl_analysis.preprocessing import PhenotypePreprocessor

    preprocessor = PhenotypePreprocessor(
        min_expression=min_expression,
        normalization_method=normalization,
        output_dir=output_dir,
    )

    try:
        output_path = preprocessor.preprocess(
            expression_file=expression,
            gene_annotation_file=gene_annotation,
            remove_outliers=remove_outliers,
        )

        # Print QC stats
        if preprocessor.qc_stats:
            stats = preprocessor.qc_stats
            table = Table(title="Phenotype QC Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")

            table.add_row("Initial genes", f"{stats.total_genes:,}")
            table.add_row("After expression filter", f"{stats.genes_after_expression_filter:,}")
            table.add_row("After variance filter", f"{stats.genes_after_variance_filter:,}")
            table.add_row("Final genes", f"{stats.final_genes:,}")
            table.add_row("Initial samples", f"{stats.total_samples:,}")
            table.add_row("After outlier filter", f"{stats.samples_after_outlier_filter:,}")
            table.add_row("Final samples", f"{stats.final_samples:,}")

            console.print(table)

        console.print(f"\n[green]Output:[/green] {output_path}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.option(
    "--expression", "-e",
    type=click.Path(exists=True),
    required=True,
    help="Expression data file.",
)
@click.option(
    "--genotypes", "-g",
    type=click.Path(exists=True),
    help="Genotype file for computing PCs.",
)
@click.option(
    "--known-covariates", "-k",
    type=click.Path(exists=True),
    help="Known covariates file.",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="results/covariates",
    help="Output directory.",
)
@click.option(
    "--n-peer-factors",
    type=int,
    default=15,
    help="Number of PEER factors.",
)
@click.option(
    "--n-genotype-pcs",
    type=int,
    default=5,
    help="Number of genotype PCs.",
)
@click.pass_context
def preprocess_covariates(
    ctx: click.Context,
    expression: str,
    genotypes: Optional[str],
    known_covariates: Optional[str],
    output_dir: str,
    n_peer_factors: int,
    n_genotype_pcs: int,
) -> None:
    """
    Preprocess covariates.

    Compute PEER factors, genotype PCs, and combine with known covariates.
    """
    from eqtl_analysis.preprocessing import CovariatePreprocessor

    preprocessor = CovariatePreprocessor(
        n_peer_factors=n_peer_factors,
        n_genotype_pcs=n_genotype_pcs,
        output_dir=output_dir,
    )

    try:
        output_path = preprocessor.preprocess(
            expression_file=expression,
            genotype_file=genotypes,
            known_covariates_file=known_covariates,
        )

        # Print stats
        if preprocessor.stats:
            stats = preprocessor.stats
            table = Table(title="Covariate Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")

            table.add_row("Known covariates", f"{stats.n_known_covariates:,}")
            table.add_row("PEER factors", f"{stats.n_peer_factors:,}")
            table.add_row("Genotype PCs", f"{stats.n_genotype_pcs:,}")
            table.add_row("Total covariates", f"{stats.n_total_covariates:,}")
            table.add_row("Samples", f"{stats.n_samples:,}")

            console.print(table)

        console.print(f"\n[green]Output:[/green] {output_path}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.option(
    "--genotypes", "-g",
    type=click.Path(exists=True),
    required=True,
    help="Genotype file (plink prefix or VCF).",
)
@click.option(
    "--phenotypes", "-p",
    type=click.Path(exists=True),
    required=True,
    help="Phenotype BED file.",
)
@click.option(
    "--covariates", "-c",
    type=click.Path(exists=True),
    help="Covariate file.",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="results/qtl",
    help="Output directory.",
)
@click.option(
    "--mode",
    type=click.Choice(["cis", "trans", "both"]),
    default="cis",
    help="Analysis mode.",
)
@click.option(
    "--cis-mode",
    type=click.Choice(["nominal", "permutation", "independent"]),
    default="permutation",
    help="Cis analysis mode.",
)
@click.option(
    "--window",
    type=int,
    default=1000000,
    help="Cis window size (bp).",
)
@click.option(
    "--maf-threshold",
    type=float,
    default=0.05,
    help="Minimum MAF threshold.",
)
@click.option(
    "--fdr-threshold",
    type=float,
    default=0.05,
    help="FDR threshold for significance.",
)
@click.option(
    "--permutations",
    type=int,
    default=10000,
    help="Number of permutations.",
)
@click.option(
    "--use-gpu/--no-gpu",
    default=False,
    help="Use GPU acceleration.",
)
@click.pass_context
def run(
    ctx: click.Context,
    genotypes: str,
    phenotypes: str,
    covariates: Optional[str],
    output_dir: str,
    mode: str,
    cis_mode: str,
    window: int,
    maf_threshold: float,
    fdr_threshold: float,
    permutations: int,
    use_gpu: bool,
) -> None:
    """
    Run eQTL analysis using tensorQTL.

    Perform cis-eQTL and/or trans-eQTL mapping.
    """
    from eqtl_analysis.analysis import TensorQTLRunner
    from eqtl_analysis.utils.config import TensorQTLConfig

    config = TensorQTLConfig(
        mode=mode,
        window=window,
        maf_threshold=maf_threshold,
        fdr_threshold=fdr_threshold,
        permutations=permutations,
        use_gpu=use_gpu,
    )

    runner = TensorQTLRunner(config=config, output_dir=output_dir)

    try:
        results = runner.run(
            genotype_file=genotypes,
            phenotype_file=phenotypes,
            covariate_file=covariates,
            mode=mode,
            cis_mode=cis_mode,
        )

        # Print stats
        if runner.run_stats:
            stats = runner.run_stats
            table = Table(title="eQTL Analysis Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")

            table.add_row("Mode", stats.mode)
            table.add_row("Phenotypes tested", f"{stats.n_phenotypes:,}")
            table.add_row("Variants tested", f"{stats.n_variants_tested:,}")
            table.add_row("Significant associations", f"{stats.n_significant:,}")
            table.add_row("Genes with eQTL", f"{stats.n_genes_with_eqtl:,}")
            table.add_row("Runtime (seconds)", f"{stats.runtime_seconds:.1f}")

            console.print(table)

        console.print("\n[green]Output files:[/green]")
        for name, path in results.items():
            console.print(f"  {name}: {path}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if ctx.obj.get("verbose"):
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@main.command()
@click.option(
    "--results", "-r",
    type=click.Path(exists=True),
    required=True,
    help="Results file from eQTL analysis.",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="results/summary",
    help="Output directory.",
)
@click.option(
    "--fdr-threshold",
    type=float,
    default=0.05,
    help="FDR threshold for significance.",
)
@click.option(
    "--generate-plots/--no-plots",
    default=True,
    help="Generate visualization plots.",
)
@click.pass_context
def summarize(
    ctx: click.Context,
    results: str,
    output_dir: str,
    fdr_threshold: float,
    generate_plots: bool,
) -> None:
    """
    Summarize and visualize eQTL results.

    Generate summary statistics, reports, and plots.
    """
    from eqtl_analysis.analysis import EQTLResults

    try:
        results_handler = EQTLResults(
            results_file=results,
            fdr_threshold=fdr_threshold,
            output_dir=output_dir,
        )

        # Print summary
        if results_handler.summary:
            summary = results_handler.summary
            table = Table(title="eQTL Results Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")

            table.add_row("Total associations", f"{summary.total_associations:,}")
            table.add_row("Significant associations", f"{summary.significant_associations:,}")
            table.add_row("Genes tested", f"{summary.genes_tested:,}")
            table.add_row("Genes with eQTL", f"{summary.genes_with_eqtl:,}")
            table.add_row("Lead variants", f"{summary.lead_variants:,}")
            table.add_row("FDR threshold", f"{summary.fdr_threshold}")

            console.print(table)

        # Generate report
        report_path = results_handler.generate_report()
        console.print(f"\n[green]Report:[/green] {report_path}")

        # Generate plots
        if generate_plots:
            manhattan_path = results_handler.plot_manhattan()
            qq_path = results_handler.plot_qq()

            if manhattan_path:
                console.print(f"[green]Manhattan plot:[/green] {manhattan_path}")
            if qq_path:
                console.print(f"[green]Q-Q plot:[/green] {qq_path}")

        # Save significant results
        sig_path = results_handler.save(significant_only=True)
        console.print(f"[green]Significant results:[/green] {sig_path}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="config.yaml",
    help="Output configuration file path.",
)
def init_config(output: str) -> None:
    """
    Generate a default configuration file.

    Creates a YAML configuration file with all available options.
    """
    config = Config()
    config.save(output)
    console.print(f"[green]Configuration saved to:[/green] {output}")


@main.command()
@click.option(
    "--calls", "-c",
    type=click.Path(exists=True),
    help="Axiom genotype calls file.",
)
@click.option(
    "--vcf",
    type=click.Path(exists=True),
    help="Existing VCF file.",
)
@click.option(
    "--expression", "-e",
    type=click.Path(exists=True),
    help="Expression data file.",
)
@click.option(
    "--covariates",
    type=click.Path(exists=True),
    help="Covariate file.",
)
@click.pass_context
def validate(
    ctx: click.Context,
    calls: Optional[str],
    vcf: Optional[str],
    expression: Optional[str],
    covariates: Optional[str],
) -> None:
    """
    Validate input files.

    Check format and integrity of input data files.
    """
    from eqtl_analysis.utils.validators import (
        validate_file_exists,
        validate_vcf_format,
        validate_expression_matrix,
        validate_covariate_matrix,
    )
    from eqtl_analysis.utils.io import read_expression_matrix

    all_valid = True

    if vcf:
        console.print(f"\n[cyan]Validating VCF:[/cyan] {vcf}")
        try:
            result = validate_vcf_format(vcf)
            if result["valid_format"]:
                console.print(f"  [green]Valid format[/green]")
                console.print(f"  Samples: {result['sample_count']}")
                console.print(f"  Variants: {result['variant_count']}")
            else:
                console.print(f"  [red]Invalid format[/red]")
                all_valid = False
            if result["issues"]:
                console.print(f"  [yellow]Issues:[/yellow] {result['issues'][:3]}")
        except Exception as e:
            console.print(f"  [red]Error:[/red] {e}")
            all_valid = False

    if expression:
        console.print(f"\n[cyan]Validating expression:[/cyan] {expression}")
        try:
            df = read_expression_matrix(expression)
            result = validate_expression_matrix(df)
            if result["valid"]:
                console.print(f"  [green]Valid format[/green]")
                console.print(f"  Genes: {result['n_genes']}")
                console.print(f"  Samples: {result['n_samples']}")
            else:
                console.print(f"  [yellow]Issues:[/yellow] {result['issues']}")
                all_valid = False
        except Exception as e:
            console.print(f"  [red]Error:[/red] {e}")
            all_valid = False

    if covariates:
        console.print(f"\n[cyan]Validating covariates:[/cyan] {covariates}")
        try:
            import pandas as pd
            df = pd.read_csv(covariates, sep="\t", index_col=0)
            result = validate_covariate_matrix(df)
            if result["valid"]:
                console.print(f"  [green]Valid format[/green]")
                console.print(f"  Covariates: {result['n_covariates']}")
                console.print(f"  Samples: {result['n_samples']}")
            else:
                console.print(f"  [yellow]Issues:[/yellow] {result['issues']}")
        except Exception as e:
            console.print(f"  [red]Error:[/red] {e}")
            all_valid = False

    if all_valid:
        console.print("\n[green]All validations passed![/green]")
    else:
        console.print("\n[red]Some validations failed.[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
