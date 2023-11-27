logistic <- function(x){
  1 / (1 + exp(-x))
}

softplus <- function(x){
  log(1 + exp(x))
}

computeDimL <- function(n){
  
  n * (n + 1) / 2
  
}

createDiagMat <- function(x){
  diag(softplus(x), nrow = length(x))
}

extractDiagElement <- function(numCov,
                               idx_blocks_start,
                               idx_blocks_end,
                               useSparse,
                               useDiag){
  
  idx_elem <- rep(NA, numCov)
  
  if(useDiag){
    idx_elem <- 1:numCov
  } else if(useSparse) {
    numBlocks <- length(idx_blocks_start)
    
    l1 <- 0
    l2 <- 0
    for (k in 1:numBlocks) {
      numCovBlock <- idx_blocks_end[k] - idx_blocks_start[k] + 1
      for (i in 1:numCovBlock) {
        idx_elem[l1 + i] <- l2 + i
        l2 <- l2 + i
      }
      l1 <- l1 + numCovBlock
    }
  } else {
    idx_elem <- sapply(1:numCov, function(i){
      i * (i + 1) / 2 
    })
  }
  
  idx_elem
  
}

createSparseMat <- function(x, idx_blocks_start, idx_blocks_end){
  
  L <- matrix(0, numCov, numCov)
  
  numBlocks <- length(idx_blocks_start)
  
  l1 <- 0
  l2 <- 0
  for (k in 1:numBlocks) {
    numCovBlock <- idx_blocks_end[k] - idx_blocks_start[k] + 1
    for (i in 1:numCovBlock) {
      L[l1 + i,l1 + 1:i] <- x[l2 + 1:i]
      l2 <- l2 + i
    }
    l1 <- l1 + numCovBlock
  }
  
  diag(L) <- softplus(diag(L))
  
  L
}

createCholMat <- function(x, numCov){
  
  L <- matrix(0, numCov, numCov)
  
  # L[!upper.tri(L)] <- x
  
  l <- 0
  for (i in 1:numCov) {
    L[i,1:i] <- x[l + 1:i]
    l <- l + i
  }
  
  diag(L) <- softplus(diag(L))
  
  L
}


buildGrid <- function(XY_sp, gridStep){
  
  x_grid <- seq(min(XY_sp[,1]) - (1.5) * gridStep, 
                max(XY_sp[,1]) + (1.5) * gridStep, by = gridStep)
  y_grid <- seq(min(XY_sp[,2]) - (1.5) * gridStep, 
                max(XY_sp[,2]) + (1.5) * gridStep, by = gridStep)
  
  pointInGrid <- matrix(T, nrow = length(x_grid), ncol = length(y_grid))
  
  for (i in 2:(length(x_grid) - 1)) {
    
    for (j in 2:(length(y_grid) - 1)) {
      
      isAnyPointInBandRight <- isPointInBandRight(XY_sp, x_grid, y_grid, i - 1, j - 1)
      
      isAnyPointInBandLeft <- isPointInBandLeft(XY_sp, x_grid, y_grid, i - 1, j - 1)
      
      isAnyPointInBandUp <- isPointInBandUp(XY_sp, x_grid, y_grid, i - 1, j - 1)
      
      isAnyPointInBandDown <- isPointInBandDown(XY_sp, x_grid, y_grid, i - 1, j - 1)
      
      if(!isAnyPointInBandRight | !isAnyPointInBandLeft | !isAnyPointInBandUp | !isAnyPointInBandDown){
        pointInGrid[i,j] <- F
      }
      
    } 
    
  }
  
  pointInGrid <- pointInGrid[-c(1,nrow(pointInGrid)),]
  pointInGrid <- pointInGrid[,-c(1,ncol(pointInGrid))]
  x_grid <- x_grid[-c(1,length(x_grid))]
  y_grid <- y_grid[-c(1,length(y_grid))]
  
  allPoints <- cbind(expand.grid(x_grid, y_grid), as.vector((pointInGrid)))
  allPoints <- allPoints[allPoints[,3],-3]  
  
  allPoints
}

# OUTPUT ----

simulateFromModel <- function(model_mean, model_cov, 
                              useDiag, useSparse, idx_covs_flipped, 
                              numSims = 1000){
  
  if(useDiag){
    L_mat <- createDiagMat(model_cov)
  } else if(useSparse) {
    L_mat <- createSparseMat(model_cov, idx_blocks_start, idx_blocks_end)
  } else {
    L_mat <- createCholMat(model_cov, numCov)
  }
  
  cov_mat <- L_mat %*% t(L_mat)
  
  beta_samples <- mvrnorm(numSims, mu = model_mean, Sigma = cov_mat)
  
  if(useSparse){
    beta_samples <- beta_samples[,idx_covs_flipped]
  }
  
  beta_samples
}

generateTrend <- function(beta_samples, df_time_occ){
  
  n_t <- ncol(df_time_occ)
  n_y <- nrow(df_time_occ)
  n_samples <- nrow(beta_samples)
  
  logit_b_samples <- 
    matrix(beta_samples[,1], n_y, n_samples, byrow = T) +
    as.matrix(df_time_occ) %*% t(beta_samples[,2:(n_t + 1)])
  # as.matrix(df_time_occ) %*% t(beta_samples[,1:n_t])
  
  b_t_samples <- logistic(logit_b_samples)
  
  b_t_qtl <- apply(b_t_samples, 1, function(x){
    quantile(x, probs = c(0.025, 0.975, .5))
  })
  
  b_t_qtl
  
}


plotCovariatesQtl <- function(beta_samples, idx_cov){
  
  beta_samples_subset <- beta_samples[,idx_cov, drop = F]
  
  beta_qtl <- apply(beta_samples_subset, 2, function(x){
    quantile(x, probs = c(0.025, 0.975))
  })
  
  beta_qtl
  
}

generateYearlyDet <- function(beta_samples, df_year_det){
  
  # number of covariates
  n_t <- ncol(df_year_det)
  n_y <- nrow(df_year_det)
  n_samples <- nrow(beta_samples)
  
  idx_intercept <- which(covNames == "P intercept")
  idx_pintercepts <- grep("p_Year",covNames)
  
  logit_b_samples <- 
    matrix(beta_samples[,idx_intercept], n_y, n_samples, byrow = T) + 
    as.matrix(df_year_det) %*% t(beta_samples[,idx_pintercepts])
  
  b_t_samples <- logistic(logit_b_samples)
  
  b_t_qtl <- apply(b_t_samples, 1, function(x){
    quantile(x, probs = c(0.025, 0.975))
  })
  
  b_t_qtl
  
}

generateTrend_det <- function(beta_samples, df_time_det, year){
  
  n_t <- ncol(df_time_det)
  n_y <- nrow(df_time_det)
  n_samples <- nrow(beta_samples)
  
  idx_intercept <- which(covNames == "P intercept")
  idx_year <- grep(year, covNames) 
  idx_covspline <- grep("Wspline",covNames)
  
  logit_b_samples <- 
    matrix(beta_samples[,idx_intercept] + beta_samples[,idx_year], n_y, n_samples, byrow = T) + 
    as.matrix(df_time_det) %*% t(beta_samples[,idx_covspline])
  
  b_t_samples <- logistic(logit_b_samples)
  
  b_t_qtl <- apply(b_t_samples, 1, function(x){
    quantile(x, probs = c(0.025, 0.975, .5))
  })
  
  b_t_qtl
  
}
