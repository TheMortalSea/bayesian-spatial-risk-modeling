# Modelling Spatial Risk for Suicides by LSOA in Cornwall County

A Bayesian spatial risk assessment using Intrinsic Conditional Autoregressive (ICAR) modeling to analyze suicide risk patterns across Lower Layer Super Output Areas (LSOAs) in Cornwall, England.

## Project Overview

This project investigates the spatial variation in suicide risk across Cornwall County using a Zero-Inflated Negative Binomial (ZINB) model with spatial random effects. Cornwall presents a unique case study as it recorded the second-highest suicide counts in 2022 despite being only the 40th most populous county in England.

## Research Objectives

- Analyze spatial patterns of suicide risk at the LSOA level in Cornwall
- Investigate the relationship between area-level deprivation and suicide incidence
- Account for spatial autocorrelation and overdispersion in sparse count data
- Identify areas with elevated suicide risk for potential public health intervention

## Data Sources

### Outcome Variable
- **Suicide Counts (2023)**: Number of registered suicides per LSOA
  - Source: Office for National Statistics (ONS)
  - Geographic Unit: 336 LSOAs in Cornwall County
  - Average population per LSOA: ~1,500 people

### Covariates
- **Index of Multiple Deprivation (IMD) 2019**: Composite deprivation measure across seven domains
  - Source: Ministry of Housing, Communities and Local Government
  - Domains: Income, employment, health, education, crime, housing, environment

- **Small Area Mental Health Index (SAMHI)**: Composite annual mental health measure
  - Source: SAMHI dataset
  - Components: NHS mental health attendances, antidepressant prescriptions, QOF depression data, DWP mental health benefit claims

## Methodology

### Model Selection
Two modeling approaches were compared using Leave-One-Out (LOO) cross-validation:
1. Negative Binomial with ICAR spatial effects
2. **Zero-Inflated Negative Binomial (ZINB) with ICAR spatial effects** *(selected model)*

### Model Specification

The final ZINB-ICAR model addresses:
- **Excess zeros**: High proportion of LSOAs with zero suicide counts
- **Overdispersion**: Variance exceeding that expected under Poisson distribution
- **Spatial autocorrelation**: Dependence between neighboring areas

**Model Structure:**
```
ϕ ~ ICAR(N, node1, node2)
Y ~ Negative Binomial(log(E) + α + β_imd*X + β_samhi*L + σ*ϕ)

Priors:
α ~ Normal(0.0, 1.0)
β_imd ~ Normal(0.0, 1.0)
β_samhi ~ Normal(0.0, 1.0)
σ ~ Normal(0.0, 1.0)
Sum(ϕ) ~ Normal(0.0, 0.001*N)
```

## Key Findings

### Model Parameters
| Parameter | Mean | 95% CI | Interpretation |
|-----------|------|--------|----------------|
| α (Baseline) | -1.38 | (-2.83, -0.06) | Generally low suicide counts |
| β_imd (Deprivation) | 0.52 | (-0.52, 1.50) | Positive but non-significant association |
| β_samhi (Mental Health) | -0.30 | (-1.40, 0.79) | Negative but non-significant association |
| σ (Spatial variation) | 1.88 | (1.20, 2.62) | Significant spatial overdispersion |
| φ_nb (Zero-inflation) | 18.25 | (0.63, 53.96) | High proportion of zero counts |

### Spatial Results
- **No LSOAs** showed statistically significant deviations in suicide risk
- Substantial variation in point estimates suggests potential spatial heterogeneity
- ICAR spatial smoothing may have attenuated extreme local values

## Technical Requirements

### Software Dependencies
- **Stan**: Bayesian modeling framework
- **R**: Statistical computing environment
- Required R packages:
  - `rstan` or `cmdstanr`
  - `loo` (model comparison)
  - `sf` (spatial data handling)
  - `ggplot2` (visualization)
  - `bayesplot` (Bayesian diagnostics)

### Data Format
- LSOA boundary files (shapefile format)
- Suicide count data (CSV format)
- IMD scores by LSOA
- SAMHI scores by LSOA
- Spatial adjacency matrix for ICAR model

### Model Performance
- Successfully handles excess zeros and overdispersion
- ICAR component provides spatial smoothing while preserving local patterns
- Zero-inflation parameter indicates model appropriately captures data structure

