functions {
  matrix kron_mvprod(matrix A, matrix B, matrix V) {
    return (B * V) * transpose(A);
  }

  matrix gp2d_exp_quad(real[] x1, real[] x2, real alpha, real rho1, real rho2, matrix z) {
    int n1 = size(x1);
    int n2 = size(x2);
    matrix[n1,n1] K1 = gp_exp_quad_cov(x1, alpha, rho1) + diag_matrix(rep_vector(1e-9, n1));
    matrix[n2,n2] K2 = gp_exp_quad_cov(x2, alpha, rho2) + diag_matrix(rep_vector(1e-9, n2));
    matrix[n1,n1] L_K1 = cholesky_decompose(K1);
    matrix[n2,n2] L_K2 = cholesky_decompose(K2);
    matrix[n1,n2] f = kron_mvprod(L_K2, L_K1, z);
    return f;
  }
}

data {
  int<lower=1> n1;
  int<lower=1> n2;
  real x1[n1];
  real x2[n2];
  int<lower=0> y[n1,n2]; 
}

parameters {
  real<lower=0> alpha;
  real<lower=0> rho1;
  real<lower=0> rho2;
  matrix[n1,n2] z;
}

model {
  alpha ~ normal(0, 1);
  rho1 ~ normal(1, 1);
  rho2 ~ normal(1, 1);

  to_vector(z) ~ normal(0,1);

  matrix[n1,n2] f = gp2d_exp_quad(x1, x2, alpha, rho1, rho2, z);

  // Poisson likelihood with log link
  for (i in 1:n1) {
    for (j in 1:n2) {
      y[i,j] ~ poisson_log(f[i,j]);
    }
  }
}

generated quantities {
  matrix[n1,n2] f_rep = gp2d_exp_quad(x1, x2, alpha, rho1, rho2, z);
}
