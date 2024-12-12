library(MASS)
library(rstan)
options(mc.cores = parallel::detectCores())

#---------------------------
# 1) Simulate data from a 3D GP (space + time) with Poisson counts
#---------------------------
set.seed(123)

# Basic RBF kernel
rbf_cov <- function(x, alpha, rho) {
  D <- as.matrix(dist(x))
  K <- alpha * exp(-0.5 * (D^2) / (rho^2))
  return(K)
}

# Seasonal (periodic) kernel
periodic_cov <- function(t, alpha, rho, period) {
  # t: time points
  # periodic kernel: K(i,j) = alpha * exp(-2 * sin^2(pi * |t_i - t_j| / period) / rho^2)
  # ensures periodicity with given period
  D <- outer(t, t, FUN = function(a,b) abs(a-b))
  K <- alpha * exp(-2 * (sin(pi*D/period))^2 / (rho^2))
  return(K)
}

# Long-term kernel (RBF for time)
longterm_cov <- function(t, alpha, rho) {
  rbf_cov(t, alpha, rho)
}

# Spatial grid
x1 <- seq(-5, 5, length.out = 20)
x2 <- seq(-5, 5, length.out = 20)
n1 <- length(x1)
n2 <- length(x2)

# Time dimension
nt <- 12
t <- 1:nt
period <- 12.0  # Assume a yearly seasonal period for illustration

# True parameters
alpha_spatial <- 1.0
rho1_spatial <- 1.5
rho2_spatial <- 1.0
sigma_nugget <- 1e-9

alpha_season <- 0.5
rho_season <- 1.0
alpha_longterm <- 0.8
rho_longterm <- 2.0

K1 <- rbf_cov(x1, alpha_spatial, rho1_spatial) + diag(sigma_nugget, n1)
K2 <- rbf_cov(x2, alpha_spatial, rho2_spatial) + diag(sigma_nugget, n2)

# Temporal covariance as sum of seasonal + long-term
Kseason <- periodic_cov(t, alpha_season, rho_season, period)
Klongterm <- longterm_cov(t, alpha_longterm, rho_longterm)
Ktime <- Kseason + Klongterm + diag(sigma_nugget, nt)

# Full covariance: K = Ktime ⊗ K2 ⊗ K1
# Flatten f as vector of length n1*n2*nt
K_spatial <- kronecker(K2, K1)
K_full <- kronecker(Ktime, K_spatial)

f_vec <- mvrnorm(1, rep(0, n1*n2*nt), K_full)

# Reshape f into a 3D array: (n1, n2, nt)
f_true <- array(f_vec, dim = c(n1, n2, nt))

lambda <- exp(f_true)
y <- array(rpois(n1*n2*nt, lambda), dim = c(n1, n2, nt))

#---------------------------
# Visualize one time slice (just as an example)
#---------------------------
image(x1, x2, log(y[,,1]+1), col = terrain.colors(50), xlab = "x1", ylab = "x2",
      main = "Poisson counts (log scale) at time t=1")

#---------------------------
# 2) Stan model
# We now include:
# - Precomputed K1, K2, Ktime from R
# - 3D structure: We'll index data in vectorized form for Stan
#
# The model will form:
# f as: f = (Ktime^{1/2} ⊗ K2^{1/2} ⊗ K1^{1/2}) * z
# We'll do this in a stepwise manner using kron_mvprod multiple times.
#
# The Stan model expects data in a vectorized form.
# We'll provide K1, K2, Ktime, and coordinates (x1,x2,t).
# The model will sample z and reconstruct f.
#---------------------------

stan_code <- "
functions {
  matrix kron_mvprod(matrix A, matrix B, matrix V) {
    // (A ⊗ B)*vec(V)
    return (B * V) * transpose(A);
  }

  // 3D GP: f ~ GP(0, Ktime ⊗ K2 ⊗ K1)
  // We'll do: f_mat = kron_mvprod(L_K2, L_K1, z_slice) for each time slice
  // Then combine time dimension using L_Ktime.
  // To handle 3D: We vectorize f as (n1*n2)*nt matrix (each column a time slice)
  matrix gp3d(matrix L_K1, matrix L_K2, matrix L_Ktime, matrix z, int n1, int n2, int nt) {
    // z is (n1*n2, nt)
    // First apply spatial decomposition:
    // For each time slice t, f_spatial = (L_K2 ⊗ L_K1)*z_t
    // Then apply L_Ktime across time dimension.
    matrix[n1*n2, nt] f_spatial;
    for (tt in 1:nt) {
      matrix[n1,n2] z_slice = to_matrix(z[,tt], n1, n2);
      matrix[n1,n2] f_slice = kron_mvprod(L_K2, L_K1, z_slice);
      f_spatial[,tt] = to_vector(f_slice);
    }
    // Now apply the temporal correlation:
    // f = f_spatial * L_Ktime^T (since we have f_spatial as N*(nt) and L_Ktime as nt*nt)
    matrix[n1*n2, nt] f_full = f_spatial * L_Ktime';
    return f_full;
  }
}

data {
  int<lower=1> n1;
  int<lower=1> n2;
  int<lower=1> nt;
  matrix[n1,n1] K1;
  matrix[n2,n2] K2;
  matrix[nt,nt] Ktime;
  int<lower=0> y[n1,n2,nt];
}

transformed data {
  matrix[n1,n1] L_K1 = cholesky_decompose(K1);
  matrix[n2,n2] L_K2 = cholesky_decompose(K2);
  matrix[nt,nt] L_Ktime = cholesky_decompose(Ktime);
}

parameters {
  matrix[n1*n2, nt] z; // Latent standard normals for space-time field
}

model {
  to_vector(z) ~ normal(0,1);

  matrix[n1*n2, nt] f_mat = gp3d(L_K1, L_K2, L_Ktime, z, n1, n2, nt);

  // Poisson likelihood with log link
  // f_mat(:, tt) corresponds to time slice tt vectorized
  // Need to index back:
  for (tt in 1:nt) {
    for (i in 1:n1) {
      for (j in 1:n2) {
        int idx = j + (i-1)*n2; // vector index in f_mat
        y[i,j,tt] ~ poisson_log(f_mat[idx, tt]);
      }
    }
  }
}

generated quantities {
  matrix[n1*n2, nt] f_mat = gp3d(L_K1, L_K2, L_Ktime, z, n1, n2, nt);
  // f_mat is the latent field in vector form
  // Convert to 3D array if needed outside Stan. Stan doesn't have a 3D array output directly.
  // We'll just leave it in f_mat form here.
}
"

writeLines(stan_code, con = "gp3d_spatiotemporal_model.stan")

#---------------------------
# 3) Fit the model
#---------------------------
# Flatten the observed y for convenience. The model loops over them, but we pass as is.
stan_data <- list(
  n1 = n1,
  n2 = n2,
  nt = nt,
  K1 = K1,
  K2 = K2,
  Ktime = Ktime,
  y = y
)

mod <- stan_model(file = "gp3d_spatiotemporal_model.stan")
fit <- sampling(mod, data = stan_data, iter = 2000, chains = 2)

#---------------------------
# 4) Visualize results
#---------------------------
f_samples <- rstan::extract(fit, "f_mat")$f_mat
# f_mat is (n1*n2, nt) for each iteration
# Compute posterior mean
f_mean_mat <- apply(f_samples, c(2,3), mean) # average over draws
# Convert back to 3D
f_hat_mean <- array(0, dim = c(n1, n2, nt))
for (tt in 1:nt) {
  # Inverse vectorization
  for (i in 1:n1) {
    for (j in 1:n2) {
      idx <- j + (i-1)*n2
      f_hat_mean[i,j,tt] <- f_mean_mat[idx, tt]
    }
  }
}

# Compare f_true and f_hat_mean for a single time slice
slice_to_view <- 1
zlim <- range(f_true[,,slice_to_view], f_hat_mean[,,slice_to_view])

par(mfrow = c(1,3))
image(x1, x2, log(y[,,slice_to_view]+1), col = terrain.colors(50),
      main = paste("y (log scale), t=", slice_to_view), xlab="x1", ylab="x2")
image(x1, x2, f_true[,,slice_to_view], col = terrain.colors(50),
      main = paste("true f, t=", slice_to_view), xlab="x1", ylab="x2", zlim=zlim)
image(x1, x2, f_hat_mean[,,slice_to_view], col = terrain.colors(50),
      main = paste("inferred f mean, t=", slice_to_view), xlab="x1", ylab="x2", zlim=zlim)
