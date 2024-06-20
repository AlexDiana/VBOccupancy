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
  diag(x, nrow = length(x))
  # diag(softplus(x), nrow = length(x))
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

computeELBOsimple_r <- function(model_mean, L, eps_beta, sigma){
  
  z <- sapply(1:n_latent, function(l){
    
    model_mean + L %*% eps_beta[l,]
    
  })
  
  list_lossval_stan <- computeELBOsimple(y, X, sigma, z, model_mean, L %*% t(L))
  list_lossval_stan$loglik - list_lossval_stan$entropy
  
}

createSparseMat <- function(x, idx_blocks_start, idx_blocks_end){
  
  TT <- matrix(0, numCov, numCov)
  
  numBlocks <- length(idx_blocks_start)
  
  l1 <- 0
  l2 <- 0
  for (k in 1:numBlocks) {
    numCovBlock <- idx_blocks_end[k] - idx_blocks_start[k] + 1
    for (i in 1:numCovBlock) {
      TT[l1 + i,l1 + 1:i] <- x[l2 + 1:i]
      l2 <- l2 + i
    }
    l1 <- l1 + numCovBlock
  }
  
  diag(TT) <- diag(TT) * diag(TT)
  
  TT
}

createNonZeroElem <- function(idx_blocks_start, idx_blocks_end){
  
  TT <- matrix(0, numCov, numCov)
  
  numBlocks <- length(idx_blocks_start)
  
  l1 <- 0
  l2 <- 0
  for (k in 1:numBlocks) {
    numCovBlock <- idx_blocks_end[k] - idx_blocks_start[k] + 1
    for (i in 1:numCovBlock) {
      TT[l1 + i,l1 + 1:i] <- 1
      l2 <- l2 + i
    }
    l1 <- l1 + numCovBlock
  }
  
  TT
}

createCholMat <- function(x, numCov){
  
  L <- matrix(0, numCov, numCov)
  
  # L[!upper.tri(L)] <- x
  
  l <- 0
  for (i in 1:numCov) {
    L[i,1:i] <- x[l + 1:i]
    l <- l + i
  }
  
  # diag(L) <- softplus(diag(L))
  
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

simulateData <- function(n, M, 
                         ncov_psi, ncov_theta, 
                         meanPsi, meanTheta){
  
  N <- sum(M)
  
  beta_psi_true <- c(meanPsi, sample(c(-1,1), ncov_psi, replace = T))
  beta_theta_true <- c(meanTheta, sample(c(-1,1), ncov_theta, replace = T))
  
  X_psi <- matrix(rnorm(n * ncov_psi), n, ncov_psi)
  X_psi <- cbind(1, X_psi)
  X_theta <- matrix(rnorm(N * ncov_theta), N, ncov_theta)
  X_theta <- cbind(1, X_theta)
  
  psi <- logistic(X_psi %*% beta_psi_true)
  theta <- logistic(X_theta %*% beta_theta_true)
  
  z <- sapply(1:n, function(i){
    rbinom(1, 1, psi[i])
  })
  
  y <- rep(0, N)
  for (i in 1:n) {
    for (m in 1:M[i]) {
      if(z[i] == 1){
        y[sum(M[seq_len(i-1)]) + m] <- 
          rbinom(1, 1, theta[sum(M[seq_len(i-1)]) + m])
      } 
    }
  }
  
  data <- list("X_psi" = X_psi,
               "X_theta" = X_theta,
               "y" = y)
  
  trueParams <- list(
    "beta_psi" = beta_psi_true,
    "beta_theta" = beta_theta_true)  
  
  list("trueParams" = trueParams,
       "data" = data)
}

fitModelSparse <- function(y, M, sumM, occ, X_psi, X_p, 
                     useDiag, useSparse,
                     epochs, n_latent, 
                     verbose = F,
                     model_mean_start = NULL, TT_start = NULL){
  
  ncov_psi <- ncol(X_psi)
  ncov_p <- ncol(X_p)
  
  # blocks
  {
    # idx_covs <- 1:(ncov_psi + ncov_p)
    idx_covs <- c(1,ncov_psi + 1,2:ncov_psi,ncov_psi + 2:ncov_p)
    idx_covs_flipped <- order(idx_covs)
    
    idx_blocks_start <- c(1,3:length(idx_covs))
    idx_blocks_end <- c(2,3:length(idx_covs))
    # idx_blocks_start <- 1:length(idx_covs)
    # idx_blocks_end <- 1:length(idx_covs)
    
    nonZeroElems <- createNonZeroElem(idx_blocks_start, idx_blocks_end)
    
  }
  
  # dim L
  {
    # numCov <- ncov_psi + ncov_p
    # if(useDiag){
    #   dimT <- numCov 
    # } else if(useSparse) {
    #   dimT <- sum(
    #     sapply(seq_along(idx_blocks_start), function(i) {
    #       computeDimL(idx_blocks_end[i] - idx_blocks_start[i] + 1)    
    #     })
    #   )
    # } else {
    #   dimT <- numCov * (numCov + 1) / 2
    # }  
  }
  
  # starting point 
  {
    if(is.null(model_mean_start)){
      model_mean <- rep(0, numCov)  
    } else {
      model_mean <- model_mean_start
    }
    
    if(is.null(TT_start)){
      TT <- diag(1 / (.5), nrow = numCov)
    } else {
      TT <- TT_start
    }
    
  }
  
  # adadelta
  {
    rho <- .95
    eps <- 10^(-6)
    egmu <- edeltamu <- rep(0, numCov)
    egT <- edeltaT <- matrix(0, numCov, numCov)
    
    eta_mu <- 10
    eta_T <- 10
    
    # normal algo
    rho_t_mu <- .01
    rho_t_T <- .01
  }
  
  loss_values <- rep(NA, epochs)
  loss_values1 <- rep(NA, epochs)
  loss_values2 <- rep(NA, epochs)
  model_mean_output <- matrix(NA, numCov, epochs)
  TT_output <- array(NA, dim = c(numCov, numCov, epochs))
  model_sigma_output <- matrix(NA, numCov * numCov, epochs)
  model_diag_output <- matrix(NA, numCov, epochs)
  
  for (i in 1:epochs) {
    
    if(verbose){
      print(paste0("Epoch = ",i))  
    }
    
    {
      eps_beta <- matrix(rnorm(n_latent * numCov), n_latent, numCov)
      # eps_beta <- matrix(rnorm(n_latent / 2 * numCov), n_latent / 2, numCov)
      # eps_beta <- rbind(eps_beta, - eps_beta)
    }
    
    # create L matrix
    {
      # if(useDiag){
      #   TT <- createDiagMat(model_cov)
      # } else if(useSparse) {
      #   TT <- createSparseMat(model_cov, idx_blocks_start, idx_blocks_end)
      # } else {
      #   TT <- createCholMat(model_cov, numCov)
      # }
      
      tTm1 <- t(solve(TT))
    }
    
    # create latent variables
    {
      
      z_flipped <- sapply(1:n_latent, function(l){
        
        model_mean + tTm1 %*% eps_beta[l,]
        
      })
      
      z <- z_flipped[idx_covs_flipped,,drop=F]
      
      beta_psi_flipped <- z_flipped[1:ncov_psi,,drop=F]
      beta_p_flipped <- z_flipped[ncov_psi + 1:ncov_p,,drop=F]
      
      beta_psi <- z[1:ncov_psi,,drop=F]
      beta_p <- z[ncov_psi + 1:ncov_p,,drop=F]
      
      logit_psi <-  X_psi %*% beta_psi
      logit_p <-  X_p %*% beta_p
      
    }
    
    # loglikelihood contribution
    deltaldeltabeta <- computedeltaldeltabeta(y, 
                                              M, 
                                              eps_beta, 
                                              sumM, occ, 
                                              logit_psi, logit_p, 
                                              beta_psi, beta_p, 
                                              X_psi, X_p, 
                                              idx_covs - 1, idx_blocks_start - 1, 
                                              idx_blocks_end - 1, 
                                              n_latent)
    
    
    
    # prior
    # time covariates
    # deltaldeltamu[1 + 1:Y] <- deltaldeltamu[1 + 1:Y] - 
    # model_mean[1 + 1:Y] %*% invSigma_gp
    
    # standard  covariates
    # deltaldeltamu[-c(1 + 1:Y)] <- deltaldeltamu[-c(1 + 1:Y)] - 
    #   model_mean[-c(1 + 1:Y)]
    
    # deltaldeltamu <- deltaldeltamu - model_mean
    
    # normal algo 
    if(F){
      gmu <- deltaldeltabeta + t(TT %*% t(eps_beta))
      
      gT <- lapply(1:n_latent, function(n_lat){
        - tTm1 %*% eps_beta[n_lat,] %*% t(tTm1 %*% gmu[n_lat,])
      }) %>% Reduce("+",.) / n_latent
      
      # diag(gT) <- diag(gT) * diag(TT) 
      
      gT[!(nonZeroElems == 1)] <- 0
      
      model_mean <- model_mean + rho_t_mu * deltamu
      tTT <- t(TT) + rho_t_T * deltaT
      
      TT <- t(tTT)
      
      # diag(TT) <- exp(diag(TT))  
    }
    
    # adadelta algo
    if(T){
      gmu <- deltaldeltabeta + t(TT %*% t(eps_beta))
      
      gT <- lapply(1:n_latent, function(n_lat){
        - tTm1 %*% eps_beta[n_lat,] %*% t(tTm1 %*% gmu[n_lat,])
      }) %>% Reduce("+",.) / n_latent
      
      # diag(gT) <- diag(gT) * diag(TT) 
      
      gT[!(nonZeroElems == 1)] <- 0
      
      gmu <- apply(gmu, 2, mean)
      
      # accumulate gradient
      egmu <- rho * egmu + (1 - rho) * (gmu)^2
      # compute change
      deltamu <- gmu * sqrt(edeltamu + eps) / sqrt(egmu + eps)
      # accumulate change
      edeltamu <- rho * edeltamu + (1 - rho) * (deltamu)^2
      
      model_mean <- model_mean + eta_mu * deltamu
      
      # accumulate gradient
      egT <- rho * egT + (1 - rho) * (gT)^2
      # compute change
      deltaT <- gT * sqrt(edeltaT + eps) / sqrt(egT + eps)
      # accumulate change
      edeltaT <- rho * edeltaT + (1 - rho) * (deltaT)^2
      
      tTT <- t(TT) + eta_T * deltaT
      
      TT <- t(tTT)
      
      # diag(TT) <- exp(diag(TT))  
    }
    
   
    # save output
    {
      model_mean_output[,i] <- model_mean
      TT_output[,,i] <- TT
      
      Sigma_current <- solve(TT %*% t(TT))
      
      list_lossval <- computeELBO(y, M, sumM, occ, logit_psi, logit_p,
                                  z_flipped, model_mean, Sigma_current)
      # loss_values[i] <- maxGrad
      loss_values1[i] <- list_lossval$loglik
      loss_values2[i] <- - list_lossval$entropy
      loss_values[i] <- loss_values1[i] + loss_values2[i]
      
      model_diag_output[,i] <- as.vector(diag(Sigma_current))
      
      if(verbose){
        # print(paste0("Decrease Mean = ",max(paramDecrease_mean)))
        # print(paste0("Decrease L = ",max(paramDecrease_cov)))
        print(paste0("Loss = ",loss_values[i]))
        # print(paste0("Average Sd = ",mean(diag(Sigma_current))))  
      }
      
      if(i > 2){
        
        # Sigma_current <- L_current %*% t(L_current)
        # Sigma_previous <- L_previous %*% t(L_previous)
        # 
        # diag_current <- diag(Sigma_current)
        # diag_previous <- diag(Sigma_previous)
        # 
        # paramDecrease_mean <- abs( (model_mean_output[,i] - model_mean_output[,i-1]) / model_mean_output[,i-1]) 
        # paramDecrease_cov <- abs( (diag_current - diag_previous) / diag_previous) 
        # paramDecrease_cov <- paramDecrease_cov[!is.na(paramDecrease_cov)]
        
        # model_sigma_output[,i] <- as.vector(Sigma_current)
        
        
      }  
    }
    
  }
  
  list("model_mean" = model_mean,
       "TT" = TT,
       "model_mean_output" = model_mean_output,
       "model_diag_output" = model_diag_output,
       "TT_output" = TT_output,
       "idx_covs_flipped" = idx_covs_flipped,
       "idx_blocks_start" = idx_blocks_start,
       "idx_blocks_end" = idx_blocks_end,
       "loss_values1" = loss_values1,
       "loss_values2" = loss_values2,
       "loss_values" = loss_values)
}


simulateFromModelSparse <- function(model_mean, TT, 
                              idx_covs_flipped, 
                              numSims = 1000){
  
  numCov <- length(model_mean)
  
  eps_beta <- matrix(rnorm(numSims * numCov), numSims, numCov)
  
  tTm1 <- t(solve(TT))

  # create latent variables
  {
    
    z_flipped <- sapply(1:numSims, function(l){
      
      model_mean + tTm1 %*% eps_beta[l,]
      
    })
    
    z <- z_flipped[idx_covs_flipped,,drop=F]
    
    # beta_psi_flipped <- z_flipped[1:ncov_psi,,drop=F]
    # beta_p_flipped <- z_flipped[ncov_psi + 1:ncov_p,,drop=F]
    # 
    # beta_psi <- z[1:ncov_psi,,drop=F]
    # beta_p <- z[ncov_psi + 1:ncov_p,,drop=F]
    
   
    
  }
  
  # if(useSparse){
    # beta_samples <- beta_samples[,idx_covs_flipped]
  # }
  
  t(z)
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
