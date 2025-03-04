data {
  int<lower=1> T;           // Number of days
  int<lower=1> D;           // Maximum delay
  int<lower=0> y[T, D];     // Observed cases matrix (time x delay)
  
  // Priors for hierarchical components
  real<lower=0> sigma;      
  real<lower=0> phi;        // Overdispersion
  
  // Regression component
  real<lower=0> beta0;
  real beta1;
  vector[T] m; // External data stream (e.g., ICU admissions)
}

parameters {
  vector[T] log_lambda;   // Epidemic curve (log scale)
  simplex[D] p_delay[T];  // Delay probabilities
}

model {
  // Priors
  log_lambda[1] ~ normal(0, 1);
  for (t in 2:T) {
    log_lambda[t] ~ normal(beta0 * log_lambda[t-1] + beta1 * log(m[t-1]), sigma);
  }

  // Likelihood
  for (t in 1:T) {
    for (d in 1:D) {
      y[t, d] ~ neg_binomial_2(exp(log_lambda[t]) * p_delay[t, d], phi);
    }
  }
}
