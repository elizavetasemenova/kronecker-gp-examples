library(MASS)
library(rstan)
options(mc.cores = parallel::detectCores())

#---------------------------
# 1) simulate data from a 3d GP (space + time) with Poisson counts
#---------------------------
set.seed(123)

exp_quad_cov <- function(x, alpha, rho) {
  D <- as.matrix(dist(x))
  K <- alpha * exp(-0.5 * (D^2) / (rho^2))
  return(K)
}

# seasonal (periodic) kernel
periodic_cov <- function(t, alpha, rho, period) {
  # periodic kernel: K(i,j) = alpha * exp(-2 * sin^2(pi * |t_i - t_j| / period) / rho^2)
  D <- outer(t, t, FUN = function(a,b) abs(a-b))
  K <- alpha * exp(-2 * (sin(pi*D/period))^2 / (rho^2))
  return(K)
}

# long-term kernel (RBF for time)
longterm_cov <- function(t, alpha, rho) {
  exp_quad_cov(t, alpha, rho)
}

# spatial grid
x1 <- seq(-5, 5, length.out = 20)
x2 <- seq(-5, 5, length.out = 20)
n1 <- length(x1)
n2 <- length(x2)

# time dimension
nt <- 12
t <- 1:nt
period <- 12.0  # yearly seasonal period


alpha_spatial <- 1.0
rho1_spatial <- 1.5
rho2_spatial <- 1.0
sigma_nugget <- 1e-9

alpha_season <- 0.5
rho_season <- 1.0
alpha_longterm <- 0.8
rho_longterm <- 2.0

K1 <- exp_quad_cov(x1, alpha_spatial, rho1_spatial) + diag(sigma_nugget, n1)
K2 <- exp_quad_cov(x2, alpha_spatial, rho2_spatial) + diag(sigma_nugget, n2)

# temporal covariance = seasonal + long-term
Kseason <- periodic_cov(t, alpha_season, rho_season, period)
Klongterm <- longterm_cov(t, alpha_longterm, rho_longterm)
Ktime <- Kseason + Klongterm + diag(sigma_nugget, nt)

# full covariance: K = Ktime (x) K2 (x) K1
K_spatial <- kronecker(K2, K1)
K_full <- kronecker(Ktime, K_spatial)

f_vec <- mvrnorm(1, rep(0, n1*n2*nt), K_full)
f_true <- array(f_vec, dim = c(n1, n2, nt))

lambda <- exp(f_true)
y <- array(rpois(n1*n2*nt, lambda), dim = c(n1, n2, nt))

#---------------------------
# 2) fit the model(assuming parameters fixed):
# 
# - precomputed K1, K2, Ktime 
# - f = (Ktime^{1/2} (x) K2^{1/2} (x) K1^{1/2}) * z
#---------------------------

stan_data <- list(
  n1 = n1,
  n2 = n2,
  nt = nt,
  K1 = K1,
  K2 = K2,
  Ktime = Ktime,
  y = y
)

mod <- stan_model(file = "stan_models/04_gp3d_Poisson_fixed_params.stan")
fit <- sampling(mod, data = stan_data, iter = 2000, chains = 2)

#---------------------------
# 3) visualise
#---------------------------

f_samples <- rstan::extract(fit, "f_mat")$f_mat
# f_mat is (n1*n2, nt) for each iteration
print(dim(f_samples))

f_mean_mat <- apply(f_samples, c(2,3), mean)

f_hat_mean <- array(0, dim = c(n1, n2, nt))
for (tt in 1:nt) {
  # unstacking
  for (i in 1:n1) {
    for (j in 1:n2) {
      idx <- j + (i-1)*n2
      f_hat_mean[i,j,tt] <- f_mean_mat[idx, tt]
    }
  }
}


# choose two time points to visualize
slice_to_view1 <- 1
slice_to_view2 <- 3

zlim_all <- range(f_true[,,slice_to_view1], f_hat_mean[,,slice_to_view1],
                  f_true[,,slice_to_view2], f_hat_mean[,,slice_to_view2])

par(mfrow = c(2, 3), mar = c(5,4,4,2) + 0.1)

# first time slice
image(x1, x2, log(y[,,slice_to_view1] + 1), col = terrain.colors(50),
      main = paste("log (y), t =", slice_to_view1), xlab="x1", ylab="x2")
image(x1, x2, f_true[,,slice_to_view1], col = terrain.colors(50),
      main = paste("true f, t=", slice_to_view1), xlab="x1", ylab="x2", zlim=zlim_all)
image(x1, x2, f_hat_mean[,,slice_to_view1], col = terrain.colors(50),
      main = paste("inferred f mean, t =", slice_to_view1), xlab="x1", ylab="x2", zlim=zlim_all)

# second time slice
image(x1, x2, log(y[,,slice_to_view2] + 1), col = terrain.colors(50),
      main = paste("log (y), t =", slice_to_view2), xlab="x1", ylab="x2")
image(x1, x2, f_true[,,slice_to_view2], col = terrain.colors(50),
      main = paste("true f, t=", slice_to_view2), xlab="x1", ylab="x2", zlim=zlim_all)
image(x1, x2, f_hat_mean[,,slice_to_view2], col = terrain.colors(50),
      main = paste("inferred f mean, t =", slice_to_view2), xlab="x1", ylab="x2", zlim=zlim_all)


