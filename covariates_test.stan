data {
  int<lower=0> N;                         // Number of spatial units
  int<lower=0> N_edges;                   // Number of edges between spatial units
  array[N_edges] int<lower=1, upper=N> node1;
  array[N_edges] int<lower=1, upper=N> node2;
  array[N] int<lower=0> Y;                // Observed counts
  int<lower=1> K;                         // Number of covariates
  matrix[N, K] X;                         // Covariate matrix
  vector<lower=0>[N] Off_set;             // Offset
}

transformed data {
  vector[N] log_Offset = log(Off_set);   // Log of offset
}

parameters {
  // Count model
  real alpha;                             // Intercept
  vector[K] beta;                         // Covariate effects
  real<lower=0> sigma;                    // SD of spatial+nonspatial random effect
  real<lower=0, upper=1> rho;             // Mixing of spatial vs nonspatial effect
  vector[N] theta;                        // Nonspatial (unstructured) random effect
  vector[N] phi;                          // Spatially structured random effect

  // Zero-inflation model
  real gamma_0;                           // Intercept for zero-inflation
  vector[K] gamma;                        // Covariate effects for zero-inflation

  // Dispersion
  real<lower=0> phi_nb;                   // Negative binomial dispersion
}

transformed parameters {
  vector[N] combined;                     // Combined random effect
  vector[N] eta;                          // Linear predictor for count
  vector[N] mu;                           // Expected counts
  vector[N] p_zero;                       // Zero-inflation probabilities

  // Combined spatial + unstructured
  combined = sqrt(1 - rho) * theta + sqrt(rho) * phi;

  // Main linear predictor
  eta = log_Offset + alpha + X * beta + combined * sigma;
  mu = exp(eta);

  // Zero-inflation
  for (i in 1:N) {
    p_zero[i] = inv_logit(gamma_0 + X[i] * gamma);
  }
}

model {
  // Priors
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);
  theta ~ normal(0, 1);
  rho ~ beta(2.0, 2.0);

  gamma_0 ~ normal(0, 1);
  gamma ~ normal(0, 1);

  phi_nb ~ gamma(2.0, 0.1);

  // Spatial CAR prior
  target += -0.5 * dot_self(phi[node1] - phi[node2]);
  sum(phi) ~ normal(0, 0.001 * N);

  // Likelihood (zero-inflated negative binomial)
  for (i in 1:N) {
    if (Y[i] == 0) {
      target += log_sum_exp(
        bernoulli_lpmf(1 | p_zero[i]),
        bernoulli_lpmf(0 | p_zero[i]) + neg_binomial_2_lpmf(0 | mu[i], phi_nb)
      );
    } else {
      target += bernoulli_lpmf(0 | p_zero[i]) +
                neg_binomial_2_lpmf(Y[i] | mu[i], phi_nb);
    }
  }
}

generated quantities {
  vector[N] rr_mu = exp(alpha + X * beta + combined * sigma);  // Relative risks per area
  vector[K] rr_beta = exp(beta);                               // RR for each covariate
  real rr_alpha = exp(alpha);                                  // Baseline RR

  vector[K] or_gamma = exp(gamma);                             // OR for zero-inflation

  // Log-likelihoods for model diagnostics
  vector[N] log_lik;
  for (i in 1:N) {
    if (Y[i] == 0) {
      log_lik[i] = log_sum_exp(
        bernoulli_lpmf(1 | p_zero[i]),
        bernoulli_lpmf(0 | p_zero[i]) + neg_binomial_2_lpmf(0 | mu[i], phi_nb)
      );
    } else {
      log_lik[i] = bernoulli_lpmf(0 | p_zero[i]) +
                   neg_binomial_2_lpmf(Y[i] | mu[i], phi_nb);
    }
  }
}

