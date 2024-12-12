library(MASS)     
library(rstan)
options(mc.cores = parallel::detectCores())

#---------------------------
# 1) simulate data from a 2D RBF Gaussian Process
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

obs_sigma <- 0.1
y <- f_true + matrix(rnorm(n1*n2, 0, obs_sigma), n1, n2)

#---------------------------
# 2) fit the model
#---------------------------
stan_data <- list(
  n1 = n1,
  n2 = n2,
  x1 = x1,
  x2 = x2,
  y = y,
  obs_sigma = obs_sigma
)

mod <- stan_model(file = "stan_models/01_gp2d_Normal.stan")

fit <- sampling(mod, data = stan_data, iter = 2000, chains = 2)

print(fit, pars = c("alpha", "rho1", "rho2"))

#---------------------------
# 3) visualise
#---------------------------

f_samples <- rstan::extract(fit, "f_rep")$f_rep

f_hat_mean <- apply(f_samples, c(2, 3), mean)

zlim <- range(y, f_hat_mean)

layout(matrix(c(1,2,3), nrow = 1, byrow = TRUE), widths = c(1,1,1))

par(mar = c(5,4,4,2) + 0.1)
image(x1, x2, y, col = terrain.colors(50), xlab = "x1", ylab = "x2",
      main = "observed y", zlim = zlim)

par(mar = c(5,4,4,2) + 0.1)
image(x1, x2, f_true, col = terrain.colors(50), xlab = "x1", ylab = "x2",
      main = "true latent f", zlim = zlim)

par(mar = c(5,4,4,2) + 0.1)
image(x1, x2, f_hat_mean, col = terrain.colors(50), xlab = "x1", ylab = "x2",
      main = "inferred f (mean)", zlim = zlim)



