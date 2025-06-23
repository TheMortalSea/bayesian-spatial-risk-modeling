data {
  int<lower=0> N;                                                        // number of spatial units or neighbourhoods
  int<lower=0> N_edges;                                                  // number of edges connecting adjacent areas using Queens contiguity
  array[N_edges] int<lower=1, upper=N> node1;                            // list of index areas showing which spatial units are neighbours
  array[N_edges] int<lower=1, upper=N> node2;                            // list of neighbouring areas showing the connection to index spatial unit
  array[N] int<lower=0> Y;                                               // dependent variable
  vector[N] X;                                                          // first covariate
  vector[N] X2;                                                         // second covariate
  vector<lower=0>[N] Off_set;                                            // offset variable
}

transformed data {
  vector[N] log_Offset = log(Off_set);                                   // use the expected cases as an offset
}

parameters {
  // Parameters for the count component
  real alpha;                                                            // intercept for count model
  real beta;                                                             // slope for first covariate (count model)
  real beta2;                                                            // slope for second covariate (count model)
  real<lower=0> sigma;                                                   // overall standard deviation
  real<lower=0, upper=1> rho;                                            // proportion unstructured vs. spatially structured variance
  vector[N] theta;                                                       // unstructured random effects
  vector[N] phi;                                                         // structured spatial random effects
  
  // Parameters for the zero-inflation component
  real gamma_0;                                                          // intercept for zero-inflation model
  real gamma_1;                                                          // slope for first covariate (zero-inflation)
  real gamma_2;                                                          // slope for second covariate (zero-inflation)
  
  // Negative binomial dispersion parameter
  real<lower=0> phi_nb;                                                  // dispersion parameter for negative binomial
}

transformed parameters {
  vector[N] combined;                                                    // combined spatial effects
  vector[N] eta;                                                         // linear predictor for count model
  vector[N] mu;                                                          // expected count
  vector[N] p_zero;                                                      // probability of structural zeros
  
  // Combine random effects
  combined = sqrt(1 - rho) * theta + sqrt(rho) * phi;
  
  // Linear predictor and expected counts
  eta = log_Offset + alpha + X * beta + X2 * beta2 + combined * sigma;
  mu = exp(eta);
  
  // Zero-inflation probabilities
  for (i in 1:N) {
    p_zero[i] = inv_logit(gamma_0 + gamma_1 * X[i] + gamma_2 * X2[i]);
  }
}

model {
  // Priors for the count component
  alpha ~ normal(0.0, 1.0);
  beta ~ normal(0.0, 1.0);
  beta2 ~ normal(0.0, 1.0);
  theta ~ normal(0.0, 1.0);
  sigma ~ normal(0.0, 1.0);
  rho ~ beta(0.5, 0.5);

  // Priors for the zero-inflation component
  gamma_0 ~ normal(0.0, 1.0);
  gamma_1 ~ normal(0.0, 1.0);
  gamma_2 ~ normal(0.0, 1.0);

  // Prior for dispersion parameter
  phi_nb ~ gamma(2.0, 0.1);

  // Spatial structure priors
  target += -0.5 * dot_self(phi[node1] - phi[node2]);
  sum(phi) ~ normal(0, 0.001 * N);
  
  // Zero-inflated Negative Binomial likelihood
  for (i in 1:N) {
    if (Y[i] == 0) {
      target += bernoulli_lpmf(1 | p_zero[i]);                              // Zero-inflation: excess zeros
      target += bernoulli_lpmf(0 | p_zero[i]) + neg_binomial_2_log_lpmf(0 | eta[i], phi_nb);  // zero count with negative binomial
    } else {
      target += bernoulli_lpmf(0 | p_zero[i]);                              // Zero-inflation: non-zero counts
      target += neg_binomial_2_log_lpmf(Y[i] | eta[i], phi_nb);                 // negative binomial likelihood for non-zero counts
    }
  }
}

generated quantities {
  // Relative risks and odds ratios
  vector[N] rr_mu = exp(alpha + X * beta + X2 * beta2 + combined * sigma);    // relative risks per area
  real rr_beta = exp(beta);                                                   // risk ratio for first covariate
  real rr_beta2 = exp(beta2);                                                  // risk ratio for second covariate
  real rr_alpha = exp(alpha);                                                  // baseline risk
  real or_gamma_1 = exp(gamma_1);                                               // odds ratio for first covariate (zero-inflation)
  real or_gamma_2 = exp(gamma_2);                                               // odds ratio for second covariate (zero-inflation)

  // Log-likelihood for model checking (e.g., for LOO/WAIC)
  vector[N] log_lik;
  
  for (i in 1:N) {
    if (Y[i] == 0) {
      log_lik[i] = log_sum_exp(
        bernoulli_lpmf(1 | p_zero[i]),
        bernoulli_lpmf(0 | p_zero[i]) + neg_binomial_2_log_lpmf(0 | eta[i], phi_nb)
      );
    } else {
      log_lik[i] = bernoulli_lpmf(0 | p_zero[i]) + neg_binomial_2_log_lpmf(Y[i] | eta[i], phi_nb);
    }
  }
}

