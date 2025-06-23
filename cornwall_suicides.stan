data {
  int<lower=0> N;
  int<lower=0> N_edges;
  array[N_edges] int<lower=1, upper=N> node1;
  array[N_edges] int<lower=1, upper=N> node2;
  array[N] int<lower=0> Y;
  vector[N] X;
  vector[N] X2;                      // <-- NEW
  vector<lower=0>[N] Off_set;
}

transformed data {
    vector[N] log_Offset = log(Off_set);                                  // use the expected cases as an offset and add to the regression model
}

parameters {
  real alpha;
  real beta;
  real beta2;                        // <-- NEW
  real<lower=0> sigma;
  real<lower=0, upper=1> rho;
  vector[N] theta;
  vector[N] phi;
}

transformed parameters {
  vector[N] combined;                                                    // values derived from adding the unstructure and structured effect of each area
  combined = sqrt(1 - rho) * theta + sqrt(rho) * phi;                    // formulation for the combined random effect
}

model {
  Y ~ poisson_log(log_Offset + alpha + X * beta + X2 * beta2 + combined * sigma);    // likelihood function: multivariable Poisson ICAR regression model
                                                                        // setting priors
  alpha ~ normal(0.0, 1.0);                                             // prior for alpha: weakly informative
  beta ~ normal(0.0, 1.0);                                              // prior for betas: weakly informative
  theta ~ normal(0.0, 1.0);                                             // prior for theta: weakly informative
  sigma ~ normal(0.0, 1.0);                                             // prior for sigma: weakly informative
  rho ~ beta(0.5, 0.5);                                                 // prior for rho
  target += -0.5 * dot_self(phi[node1] - phi[node2]);                   // calculates the spatial weights
  sum(phi) ~ normal(0, 0.001 * N);                                      // priors for phi
}

generated quantities {
  vector[N] eta = alpha + X * beta + X2 * beta2 + combined * sigma;
  vector[N] rr_mu = exp(eta);
  real rr_beta = exp(beta);
  real rr_beta2 = exp(beta2);        // <-- NEW
  real rr_alpha = exp(alpha);
}
