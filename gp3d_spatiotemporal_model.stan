
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

