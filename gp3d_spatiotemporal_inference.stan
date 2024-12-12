
functions {
  matrix rbf_cov(real[] x, real alpha, real rho) {
    int N = size(x);
    matrix[N,N] K;
    for (i in 1:N) {
      for (j in 1:N) {
        real dist_sq = (x[i] - x[j])^2;
        K[i,j] = alpha * exp(-0.5 * dist_sq / (rho^2));
      }
    }
    return K;
  }

  matrix periodic_cov(real[] t, real alpha, real rho, real period) {
    int nt = size(t);
    matrix[nt,nt] K;
    for (i in 1:nt) {
      for (j in 1:nt) {
        real diff = fabs(t[i]-t[j]);
        K[i,j] = alpha * exp(-2 * square(sin(pi()*diff/period)) / (rho^2));
      }
    }
    return K;
  }

  // Kronecker multiplication: (A ⊗ B)*vec(V)
  matrix kron_mvprod(matrix A, matrix B, matrix V) {
    return (B * V) * transpose(A);
  }

  // Compute full 3D GP:
  // gp3d constructs f from z using L_K1, L_K2, L_Ktime
  // f_mat is (n1*n2, nt)
  matrix gp3d(matrix L_K1, matrix L_K2, matrix L_Ktime, matrix z, int n1, int n2, int nt) {
    matrix[n1*n2, nt] f_spatial;
    for (tt in 1:nt) {
      matrix[n1,n2] z_slice = to_matrix(z[,tt], n1, n2);
      matrix[n1,n2] f_slice = kron_mvprod(L_K2, L_K1, z_slice);
      f_spatial[,tt] = to_vector(f_slice);
    }
    return f_spatial * L_Ktime';
  }
}

data {
  int<lower=1> n1;
  int<lower=1> n2;
  int<lower=1> nt;
  real x1[n1];
  real x2[n2];
  real t[nt];
  real period;
  int<lower=0> y[n1,n2,nt];
  real sigma_nugget;
}

parameters {
  // Spatial parameters
  real<lower=0> alpha_spatial;
  real<lower=0> rho1_spatial;
  real<lower=0> rho2_spatial;

  // Temporal parameters (seasonal + long-term)
  real<lower=0> alpha_season;
  real<lower=0> rho_season;
  real<lower=0> alpha_longterm;
  real<lower=0> rho_longterm;

  matrix[n1*n2, nt] z;
}

transformed parameters {
  // Build covariance matrices
  matrix[n1,n1] K1 = rbf_cov(x1, alpha_spatial, rho1_spatial) + diag_matrix(rep_vector(sigma_nugget, n1));
  matrix[n2,n2] K2 = rbf_cov(x2, alpha_spatial, rho2_spatial) + diag_matrix(rep_vector(sigma_nugget, n2));

  matrix[nt,nt] Kseason = periodic_cov(t, alpha_season, rho_season, period);
  matrix[nt,nt] Klongterm = rbf_cov(t, alpha_longterm, rho_longterm);
  matrix[nt,nt] Ktime = Kseason + Klongterm + diag_matrix(rep_vector(sigma_nugget, nt));

  matrix[n1,n1] L_K1 = cholesky_decompose(K1);
  matrix[n2,n2] L_K2 = cholesky_decompose(K2);
  matrix[nt,nt] L_Ktime = cholesky_decompose(Ktime);

  matrix[n1*n2, nt] f_mat = gp3d(L_K1, L_K2, L_Ktime, z, n1, n2, nt);
}

model {
  // Priors
  alpha_spatial ~ normal(1, 1);
  rho1_spatial ~ normal(1, 1);
  rho2_spatial ~ normal(1, 1);
  alpha_season ~ normal(0.5, 0.5);
  rho_season ~ normal(1, 0.5);
  alpha_longterm ~ normal(0.8, 0.5);
  rho_longterm ~ normal(2, 1);

  to_vector(z) ~ normal(0,1);

  // Likelihood
  for (tt in 1:nt) {
    for (i in 1:n1) {
      for (j in 1:n2) {
        int idx = j + (i-1)*n2;
        y[i,j,tt] ~ poisson_log(f_mat[idx, tt]);
      }
    }
  }
}

generated quantities {
  // Posterior predictive samples for f
  // Already computed: f_mat
  // If needed, can generate replicate counts, etc.
}

