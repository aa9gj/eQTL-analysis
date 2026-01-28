# eQTL Analysis Pipeline

[![CI](https://github.com/aa9gj/eQTL_analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/aa9gj/eQTL_analysis/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade pipeline for **Expression Quantitative Trait Loci (eQTL)** analysis, designed for identifying genetic variants that influence gene expression. Originally developed for a population of 33 healthy canines.

## Features

- **Genotype Preprocessing**
  - Convert Axiom genotype TSV files to VCF format
  - Coordinate liftover between genome assemblies (e.g., canfam3 to canfam4)
  - Genotype imputation using Beagle
  - Quality control filtering (call rate, MAF, HWE)

- **Phenotype Preprocessing**
  - Expression data normalization (TMM, quantile, inverse normal)
  - Low expression filtering
  - Outlier sample detection
  - tensorQTL-ready BED format output

- **Covariate Computation**
  - PEER factor estimation
  - Genotype principal components
  - Known covariate integration
  - Categorical variable encoding

- **eQTL Mapping**
  - Cis-eQTL analysis (nominal, permutation, conditional)
  - Trans-eQTL analysis
  - Interaction eQTL (ieQTL) analysis
  - Multiple testing correction

- **Results Analysis**
  - Summary statistics and reports
  - Manhattan and Q-Q plots
  - Lead variant identification
  - Result annotation

## Installation

### From PyPI (recommended)

```bash
pip install eqtl-analysis
```

### From source

```bash
git clone https://github.com/aa9gj/eQTL_analysis.git
cd eQTL_analysis
pip install -e ".[dev]"
```

### Dependencies

The pipeline requires:
- Python >= 3.9
- tensorQTL
- PyTorch
- pandas, numpy, scipy
- pysam (for VCF handling)

Optional dependencies:
- Beagle (for imputation)
- PEER (for hidden factor estimation)

## Quick Start

### 1. Initialize Configuration

```bash
eqtl-analysis init-config -o config.yaml
```

### 2. Preprocess Genotypes

```bash
# From Axiom TSV files
eqtl-analysis preprocess-genotypes \
    --calls genotype_calls.tsv \
    --annotations snp_annotations.tsv \
    --chain-file canfam3_to_canfam4.chain \
    --output-dir results/genotypes

# From existing VCF
eqtl-analysis preprocess-genotypes \
    --vcf genotypes.vcf.gz \
    --skip-liftover \
    --output-dir results/genotypes
```

### 3. Preprocess Phenotypes

```bash
eqtl-analysis preprocess-phenotypes \
    --expression expression_matrix.tsv \
    --gene-annotation genes.bed \
    --normalization inverse_normal \
    --output-dir results/phenotypes
```

### 4. Compute Covariates

```bash
eqtl-analysis preprocess-covariates \
    --expression expression_matrix.tsv \
    --genotypes results/genotypes/genotypes_qc.vcf.gz \
    --known-covariates sample_info.tsv \
    --n-peer-factors 15 \
    --n-genotype-pcs 5 \
    --output-dir results/covariates
```

### 5. Run eQTL Analysis

```bash
eqtl-analysis run \
    --genotypes results/genotypes/genotypes_qc \
    --phenotypes results/phenotypes/expression.bed.gz \
    --covariates results/covariates/covariates.txt \
    --mode cis \
    --cis-mode permutation \
    --output-dir results/qtl
```

### 6. Summarize Results

```bash
eqtl-analysis summarize \
    --results results/qtl/cis_permutation.txt.gz \
    --fdr-threshold 0.05 \
    --generate-plots \
    --output-dir results/summary
```

## Pipeline Workflow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Genotypes     │    │   Expression    │    │   Covariates    │
│   (Axiom/VCF)   │    │   (counts/TPM)  │    │   (metadata)    │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  QC & Liftover  │    │  Normalization  │    │   PEER Factors  │
│   & Imputation  │    │   & Filtering   │    │   & Geno PCs    │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │    tensorQTL eQTL     │
                    │       Mapping         │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Results & Reports   │
                    │   Plots & Summaries   │
                    └───────────────────────┘
```

## Configuration

Create a `config.yaml` file to customize the pipeline:

```yaml
# Output settings
output_dir: results
log_dir: logs
n_jobs: 4

# Quality control
quality_control:
  min_call_rate: 0.95
  min_maf: 0.01
  max_missing_rate: 0.05
  hwe_pvalue_threshold: 1e-6

# Imputation settings
imputation:
  window_size: 50000
  ne: 20000
  nthreads: 4

# Liftover settings
liftover:
  source_assembly: canfam3
  target_assembly: canfam4

# tensorQTL settings
tensorqtl:
  mode: cis
  window: 1000000
  maf_threshold: 0.05
  fdr_threshold: 0.05
  permutations: 10000
  use_gpu: false
```

## Python API

```python
from eqtl_analysis.preprocessing import (
    GenotypePreprocessor,
    PhenotypePreprocessor,
    CovariatePreprocessor,
)
from eqtl_analysis.analysis import TensorQTLRunner, EQTLResults

# Preprocess genotypes
geno_processor = GenotypePreprocessor()
vcf_path = geno_processor.preprocess(
    vcf_file="genotypes.vcf.gz",
    run_qc=True,
)

# Preprocess phenotypes
pheno_processor = PhenotypePreprocessor(
    normalization_method="inverse_normal"
)
bed_path = pheno_processor.preprocess(
    expression_file="expression.tsv",
    gene_annotation_file="genes.bed",
)

# Compute covariates
cov_processor = CovariatePreprocessor(
    n_peer_factors=15,
    n_genotype_pcs=5,
)
cov_path = cov_processor.preprocess(
    expression_file="expression.tsv",
    genotype_file=vcf_path,
)

# Run eQTL mapping
runner = TensorQTLRunner()
results_path = runner.run_cis(
    genotype_file=vcf_path,
    phenotype_file=bed_path,
    covariate_file=cov_path,
    mode="permutation",
)

# Analyze results
results = EQTLResults(results_path)
results.apply_fdr_correction()
significant = results.get_significant(threshold=0.05)
results.plot_manhattan()
results.generate_report()
```

## Input File Formats

### Genotypes
- **VCF/VCF.gz**: Standard VCF format
- **Axiom TSV**: Calls and annotations files from Axiom arrays

### Expression
- **TSV/CSV**: Matrix with genes as rows, samples as columns
- **BED**: tensorQTL phenotype BED format

### Covariates
- **TSV**: Matrix with covariates as rows, samples as columns

### Gene Annotation
- **BED**: Standard BED format with gene coordinates

## Output Files

| File | Description |
|------|-------------|
| `genotypes_qc.vcf.gz` | QC-filtered genotypes |
| `expression.bed.gz` | Normalized expression in BED format |
| `covariates.txt` | Combined covariate matrix |
| `cis_permutation.txt.gz` | Cis-eQTL results |
| `eqtl_report.txt` | Summary report |
| `manhattan_plot.png` | Manhattan plot |
| `qq_plot.png` | Q-Q plot |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/eqtl_analysis --cov-report=html

# Run specific test module
pytest tests/test_preprocessing.py -v
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{eqtl_analysis,
  title = {eQTL Analysis Pipeline},
  author = {eQTL Analysis Team},
  year = {2024},
  url = {https://github.com/aa9gj/eQTL_analysis}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [tensorQTL](https://github.com/broadinstitute/tensorqtl) for GPU-accelerated QTL mapping
- [Beagle](https://faculty.washington.edu/browning/beagle/beagle.html) for genotype imputation
- The GTEx Consortium for methodological guidance
