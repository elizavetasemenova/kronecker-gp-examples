library(MASS)
library(rstan)
options(mc.cores = parallel::detectCores())

#---------------------------
# 1) simulate data from a 3d GP (space + time) with Poisson counts
#---------------------------
set.seed(123)

exp_quad_cov <- function(x, alpha, rho) {
  D <- as.matrix(dist(x))
  K <- alpha * exp(-0.5 * (D^2) / rho^2)
  return(K)
}

periodic_cov <- function(t, alpha, rho, period) {
  D <- outer(t, t, FUN=function(a,b) abs(a-b))
  K <- alpha * exp(-2 * (sin(pi*D/period))^2 / (rho^2))
  return(K)
}

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
period <- 12.0

# true parameters for simulation
alpha_spatial_true <- 1.0
rho1_spatial_true <- 1.5
rho2_spatial_true <- 1.0
sigma_nugget <- 1e-6

alpha_season_true <- 0.5
rho_season_true <- 1.0
alpha_longterm_true <- 0.8
rho_longterm_true <- 2.0

K1 <- exp_quad_cov(x1, alpha_spatial_true, rho1_spatial_true) + diag(sigma_nugget, n1)
K2 <- exp_quad_cov(x2, alpha_spatial_true, rho2_spatial_true) + diag(sigma_nugget, n2)
Kseason <- periodic_cov(t, alpha_season_true, rho_season_true, period)
Klongterm <- longterm_cov(t, alpha_longterm_true, rho_longterm_true)
Ktime <- Kseason + Klongterm + diag(sigma_nugget, nt)

K_spatial <- kronecker(K2, K1)
K_full <- kronecker(Ktime, K_spatial)

f_vec <- MASS::mvrnorm(1, rep(0, n1*n2*nt), K_full)
f_true <- array(f_vec, dim = c(n1, n2, nt))
lambda <- exp(f_true)
y_full <- array(rpois(n1*n2*nt, lambda), dim = c(n1, n2, nt))

# introduce missing values
set.seed(999)
missing_ratio <- 0.1
N <- n1*n2*nt
n_miss <- round(N * missing_ratio)
miss_idx <- sample(seq_len(N), size = n_miss, replace = FALSE)

y <- y_full
y[miss_idx] <- NA

# observed indices
obs_idx <- which(!is.na(y))
N_obs <- length(obs_idx)
N_miss <- length(miss_idx)

# convert obs_idx and miss_idx to (i,j,tt) coordinates
obs_iii <- ((obs_idx - 1) %% n1) + 1
obs_jjj <- ((obs_idx - 1) %/% n1) %% n2 + 1
obs_ttt <- ((obs_idx - 1) %/% (n1*n2)) + 1

miss_iii <- ((miss_idx - 1) %% n1) + 1
miss_jjj <- ((miss_idx - 1) %/% n1) %% n2 + 1
miss_ttt <- ((miss_idx - 1) %/% (n1*n2)) + 1

y_obs <- y[obs_idx]

#---------------------------
# 2) fit the model (infer parameters with missing data)
#---------------------------
stan_data <- list(
  n1 = n1,
  n2 = n2,
  nt = nt,
  x1 = x1,
  x2 = x2,
  t = t,
  period = period,
  sigma_nugget = sigma_nugget,
  N_obs = N_obs,
  obs_i = obs_iii,
  obs_j = obs_jjj,
  obs_t = obs_ttt,
  y_obs = y_obs,
  N_miss = N_miss,
  miss_i = miss_iii,
  miss_j = miss_jjj,
  miss_t = miss_ttt
)

mod <- stan_model(file = "stan_models/06_gp3d_Poisson_missing.stan")
fit <- sampling(mod, data = stan_data, iter = 2000, chains = 2)

print(fit, pars = c("alpha_spatial", "rho1_spatial", "rho2_spatial",
                    "alpha_season", "rho_season", "alpha_longterm", "rho_longterm"))

#---------------------------
# 3) visualise
#---------------------------
f_samples <- rstan::extract(fit, "f_mat")$f_mat
f_mean_mat <- apply(f_samples, c(2,3), mean)
f_hat_mean <- array(0, dim = c(n1, n2, nt))
for (tt in 1:nt) {
  for (i in 1:n1) {
    for (j in 1:n2) {
      idx <- j + (i-1)*n2
      f_hat_mean[i,j,tt] <- f_mean_mat[idx, tt]
    }
  }
}

# imputed values
y_miss_pred_samples <- rstan::extract(fit, "y_miss_pred")$y_miss_pred
y_miss_pred_mean <- apply(y_miss_pred_samples, 2, mean)

y_imputed <- y
y_imputed[is.na(y_imputed)] <- y_miss_pred_mean

slice_to_view1 <- 1
slice_to_view2 <- 3

zlim_all <- range(f_true[,,slice_to_view1], f_hat_mean[,,slice_to_view1],
                  f_true[,,slice_to_view2], f_hat_mean[,,slice_to_view2])

par(mfrow = c(2, 3), mar = c(5,4,4,2) + 0.1)

# first time slice
image(x1, x2, log(y_imputed[,,slice_to_view1] + 1), col = terrain.colors(50),
      main = paste("log (imputed y), t =", slice_to_view1), xlab="x1", ylab="x2")
image(x1, x2, f_true[,,slice_to_view1], col = terrain.colors(50),
      main = paste("true f, t=", slice_to_view1), xlab="x1", ylab="x2", zlim=zlim_all)
image(x1, x2, f_hat_mean[,,slice_to_view1], col = terrain.colors(50),
      main = paste("inferred f mean, t =", slice_to_view1), xlab="x1", ylab="x2", zlim=zlim_all)

# second time slice
image(x1, x2, log(y_imputed[,,slice_to_view2] + 1), col = terrain.colors(50),
      main = paste("log (imputed y), t =", slice_to_view2), xlab="x1", ylab="x2")
image(x1, x2, f_true[,,slice_to_view2], col = terrain.colors(50),
      main = paste("true f, t=", slice_to_view2), xlab="x1", ylab="x2", zlim=zlim_all)
image(x1, x2, f_hat_mean[,,slice_to_view2], col = terrain.colors(50),
      main = paste("inferred f mean, t =", slice_to_view2), xlab="x1", ylab="x2", zlim=zlim_all)


