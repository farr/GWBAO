functions {
  real bao_kernel(real r, real alpha, real bao_amp, real bao_loc, real bao_width) {
    real r0 = 0.001; /* Turnover radius for regularized power law. */
    real dr_bao = r - bao_loc;
    real bao_xi = bao_amp*exp(-0.5*dr_bao*dr_bao/(bao_width*bao_width));
    real pl_xi;

    if (r < r0) {
      pl_xi = 1 - 0.1*(r/r0);
    } else {
      pl_xi = 0.9*(r/r0)^(-alpha);
    }

    return (pl_xi + bao_xi);
  }

  matrix bao_corr_matrix(vector[] rs, real alpha, real bao_amp, real bao_loc, real bao_width) {
    int N = size(rs);
    matrix[N,N] C;

    real k0 = bao_kernel(0.0, alpha, bao_amp, bao_loc, bao_width);

    for (i in 1:N) {
      for (j in i:N) {
        vector[3] dr = rs[i] - rs[j];
        real r = sqrt(dr'*dr);
        real ker = bao_kernel(r, alpha, bao_amp, bao_loc, bao_width)/k0;

        C[i,j] = ker;
        C[j,i] = ker;
      }
    }

    return C;
  }
}

data {
  int nobs;

  vector[3] pts[nobs];

  /* For the prior. */
  real mu_n0;
  real sigma_n0;

  /* Volume out to cutoff */
  real V;
}

parameters {
  real logN0;

  real<lower=0, upper=0.1> A;

  real<lower=1> alpha;

  real<lower=0, upper=0.1> bao_amp;
  real<lower=0> bao_loc;
  real<lower=0> bao_width;

  vector[nobs] logN_unit;
}

transformed parameters {
  vector[nobs] logN;
  {
    matrix[nobs, nobs] C;

    C = bao_corr_matrix(pts, alpha, bao_amp, bao_loc, bao_width);

    logN = logN0 + A*cholesky_decompose(C)*logN_unit;
  }
}

model {
  /* Priors */
  logN0 ~ normal(mu_n0, sigma_n0);

  A ~ lognormal(log(3e-2), 0.5);

  alpha ~ normal(1.8, 0.5);

  bao_amp ~ lognormal(log(1e-2), 0.5);
  bao_loc ~ lognormal(log(0.14), 0.05/0.14);
  bao_width ~ lognormal(log(2e-2), 0.5);

  logN_unit ~ normal(0,1); /* => logN ~ multi_normal(logN0, C) */


  /* Likelihood */
  for (i in 1:nobs) {
    target += logN[i];
  }
  target += -V*exp(logN0 + A*A/2);
}
