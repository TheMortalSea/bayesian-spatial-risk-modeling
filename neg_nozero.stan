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
  
  // Negative binomial dispersion parameter
  real<lower=0> phi_nb;                                                  // dispersion parameter for negative binomial
}

transformed parameters {
  vector[N] combined;                                                    // combined spatial effects
  vector[N] eta;                                                         // linear predictor for count model
  vector[N] mu;                                                          // expected count
  
  // Combine random effects
  combined = sqrt(1 - rho) * theta + sqrt(rho) * phi;
  
  // Linear predictor and expected counts
  eta = log_Offset + alpha + X * beta + X2 * beta2 + combined * sigma;
  mu = exp(eta);                                                         // expected count
}

model {
  // Priors for the count component
  alpha ~ normal(0.0, 1.0);
  beta ~ normal(0.0, 1.0);
  beta2 ~ normal(0.0, 1.0);
  theta ~ normal(0.0, 1.0);
  sigma ~ normal(0.0, 1.0);
  rho ~ beta(0.5, 0.5);

  // Prior for dispersion parameter
  phi_nb ~ gamma(2.0, 0.1);

  // Spatial structure priors
  target += -0.5 * dot_self(phi[node1] - phi[node2]);
  sum(phi) ~ normal(0, 0.001 * N);
  
  // Negative Binomial likelihood
  for (i in 1:N) {
    target += neg_binomial_2_lpmf(Y[i] | mu[i], phi_nb);  // likelihood for non-zero counts
  }
}

generated quantities {
  // Relative risks
  vector[N] rr_mu = exp(alpha + X * beta + X2 * beta2 + combined * sigma);    // relative risks per area
  real rr_beta = exp(beta);                                                   // risk ratio for first covariate
  real rr_beta2 = exp(beta2);                                                 // risk ratio for second covariate
  real rr_alpha = exp(alpha);                                                 // baseline risk

  // Log-likelihood for model checking (e.g., for LOO/WAIC)
  vector[N] log_lik;
  
  for (i in 1:N) {
    log_lik[i] = neg_binomial_2_lpmf(Y[i] | mu[i], phi_nb); // log-likelihood for Negative Binomial
  }
}



