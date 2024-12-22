functions {
  matrix kron_mvprod(matrix A, matrix B, matrix V) {
    return (B * V) * transpose(A);
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
  real sigma_nugget;

  int<lower=1> N_obs;
  int obs_i[N_obs];
  int obs_j[N_obs];
  int obs_t[N_obs];
  int<lower=0> y_obs[N_obs];

  int<lower=1> N_miss;
  int miss_i[N_miss];
  int miss_j[N_miss];
  int miss_t[N_miss];
}

parameters {
  // spatial parameters
  real<lower=0> alpha_spatial;
  real<lower=0> rho1_spatial;
  real<lower=0> rho2_spatial;

  // temporal parameters (seasonal + long-term)
  real<lower=0> alpha_season;
  real<lower=0> rho_season;
  real<lower=0> alpha_longterm;
  real<lower=0> rho_longterm;

  matrix[n1*n2, nt] z;
}

transformed parameters {
  matrix[n1,n1] K1 = gp_exp_quad_cov(x1, alpha_spatial, rho1_spatial) + diag_matrix(rep_vector(sigma_nugget, n1));
  matrix[n2,n2] K2 = gp_exp_quad_cov(x2, alpha_spatial, rho2_spatial) + diag_matrix(rep_vector(sigma_nugget, n2));
  
  matrix[nt,nt] Kseason = periodic_cov(t, alpha_season, rho_season, period);
  matrix[nt,nt] Klongterm = gp_exp_quad_cov(t, alpha_longterm, rho_longterm);
  matrix[nt,nt] Ktime = Kseason + Klongterm + diag_matrix(rep_vector(sigma_nugget, nt));

  matrix[n1,n1] L_K1 = cholesky_decompose(K1);
  matrix[n2,n2] L_K2 = cholesky_decompose(K2);
  matrix[nt,nt] L_Ktime = cholesky_decompose(Ktime);

  matrix[n1*n2, nt] f_mat = gp3d(L_K1, L_K2, L_Ktime, z, n1, n2, nt);
}

model {
  alpha_spatial ~ normal(0.5, 0.5);
  rho1_spatial ~ normal(0.5, 0.5);
  rho2_spatial ~ normal(0.5, 0.5);
  alpha_season ~ normal(0.5, 0.2);
  rho_season ~ normal(1, 0.2);
  alpha_longterm ~ normal(0.8, 0.2);
  rho_longterm ~ normal(1, 0.5);

  to_vector(z) ~ normal(0,1);

  // Poisson likelihood only for observed data
  for (o in 1:N_obs) {
    int idx = obs_j[o] + (obs_i[o]-1)*n2;
    y_obs[o] ~ poisson_log(f_mat[idx, obs_t[o]]);
  }
}

generated quantities {
  // Impute missing values using posterior predictive distribution
  int y_miss_pred[N_miss];
  {
    for (m in 1:N_miss) {
      int idx = miss_j[m] + (miss_i[m]-1)*n2;
      y_miss_pred[m] = poisson_log_rng(f_mat[idx, miss_t[m]]);
    }
  }
}


