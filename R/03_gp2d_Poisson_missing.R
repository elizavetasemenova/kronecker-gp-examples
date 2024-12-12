library(MASS)     
library(rstan)
options(mc.cores = parallel::detectCores())

#---------------------------
# 1) simulate data from a 2D RBF Gaussian Process (Poisson counts)
#---------------------------
set.seed(123)

exp_quad_cov <- function(x, alpha, rho) {
  D <- as.matrix(dist(x))
  K <- alpha * exp(-0.5 * (D^2) / (rho^2))
  return(K)
}

x1 <- seq(-5, 5, length.out = 20)
x2 <- seq(-5, 5, length.out = 20)

n1 <- length(x1)
n2 <- length(x2)

alpha_true <- 1.0
rho1_true <- 1.5
rho2_true <- 1.0
sigma_nugget <- 1e-9

K1 <- exp_quad_cov(x1, alpha_true, rho1_true) + diag(sigma_nugget, n1)
K2 <- exp_quad_cov(x2, alpha_true, rho2_true) + diag(sigma_nugget, n2)
K <- kronecker(K2, K1)

f_vec <- MASS::mvrnorm(1, rep(0, n1*n2), K)
f_true <- matrix(f_vec, n1, n2)

lambda <- exp(f_true)
y_full <- matrix(rpois(n1*n2, lambda), n1, n2)

# introduce missing data (e.g., 20% missing)
set.seed(999)
missing_ratio <- 0.2
N <- n1*n2
n_miss <- round(N * missing_ratio)
miss_idx <- sample(seq_len(N), size = n_miss, replace = FALSE)

y <- y_full
y[miss_idx] <- NA  # mask missing values

# get indices of observed and missing data
obs_idx <- which(!is.na(y))
miss_idx <- which(is.na(y))

N_obs <- length(obs_idx)
N_miss <- length(miss_idx)

# convert 2D indices to row/col
obs_rows <- ((obs_idx - 1) %% n1) + 1
obs_cols <- ((obs_idx - 1) %/% n1) + 1
miss_rows <- ((miss_idx - 1) %% n1) + 1
miss_cols <- ((miss_idx - 1) %/% n1) + 1

# vector of observed y
y_obs <- y[obs_idx]

#---------------------------
# 2) fit the model
#---------------------------

stan_data <- list(
  n1 = n1,
  n2 = n2,
  x1 = x1,
  x2 = x2,
  N_obs = N_obs,
  obs_rows = obs_rows,
  obs_cols = obs_cols,
  y_obs = y_obs,
  N_miss = N_miss,
  miss_rows = miss_rows,
  miss_cols = miss_cols
)

mod <- stan_model(file = "stan_models/03.stan")

fit <- sampling(mod, data = stan_data, iter = 2000, chains = 2)

print(fit, pars = c("alpha", "rho1", "rho2"))

#---------------------------
# 3) visualise
#---------------------------

f_samples <- rstan::extract(fit, "f_rep")$f_rep

f_hat_mean <- apply(f_samples, c(2, 3), mean)

# extract imputed values
y_miss_samples <- rstan::extract(fit, "y_miss_pred")$y_miss_pred
y_miss_hat <- apply(y_miss_samples, 2, mean) 

# reconstruct full y matrix with imputations
y_imputed <- y
y_imputed[is.na(y_imputed)] <- y_miss_hat

zlim <- range(f_true, f_hat_mean)

layout(matrix(c(1,2,3,4), nrow = 2, byrow = TRUE))
par(mar = c(5,4,4,2) + 0.1)

# observed + missing (original)
image(x1, x2, log(y+1), col = terrain.colors(50), xlab = "x1", ylab = "x2",
      main = "observed data (with NAs)", zlim = range(log(y+1), na.rm=TRUE))

# imputed data
image(x1, x2, log(y_imputed+1), col = terrain.colors(50), xlab = "x1", ylab = "x2",
      main = "imputed data (log scale)", zlim = range(log(y_imputed+1)))

# true latent f
image(x1, x2, f_true, col = terrain.colors(50), xlab = "x1", ylab = "x2",
      main = "true latent f", zlim = zlim)

# inferred latent f (mean)
image(x1, x2, f_hat_mean, col = terrain.colors(50), xlab = "x1", ylab = "x2",
      main = "inferred f (mean)", zlim = zlim)
