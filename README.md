# Kronecker Demos

These examples demonstrate Gaussian Processes (GPs) with Kronecker-structured covariance matrices.

## Key idea

If $\mathbf{x} = (x_1, x_2)$ and the kernel factorizes as:

$$
k((x_1,x_2),(x_1',x_2')) = k_1(x_1,x_1')\,k_2(x_2,x_2'),
$$

then the covariance matrix over grids $X_1 \times X_2$ can be written as:

$$
K = K_2 \otimes K_1.
$$

For three dimensions (e.g., space $\times$ space $\times$ time):

$$
k((x_1,x_2,t),(x_1',x_2',t')) = k_1(x_1,x_1')\,k_2(x_2,x_2')\,k_t(t,t'),
$$

implying:

$$
K = K_t \otimes K_2 \otimes K_1.
$$

This structure enables efficient computations (e.g. Cholesky) at $O(n^{3/d})$ instead of $O(n^3)$.

## Likelihoods

- **Normal**: 
  
  $$
  y(\mathbf{x}) = f(\mathbf{x}) + \epsilon, \quad \epsilon \sim \mathcal{N}(0,\sigma^2).
  $$
  

- **Poisson**:

  $$
  y(\mathbf{x}) \sim \text{Poisson}(\exp(f(\mathbf{x}))).
  $$

## Examples

- **01**: 2d GP with Normal likelihood, no missingness
- **02**: 2d GP with Poisson likelihood, no missingness
- **03**: 2d GP with Poisson likelihood, with missing values  (impute from posterior)
- **04**: 3d GP (e.g. space-time), Poisson likelihood, no missingness, fixed parameters
- **05**: 3d GP with Poisson likelihood, no missingness, parameter inference
