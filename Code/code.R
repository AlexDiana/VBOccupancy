library(Rcpp)
library(RcppArmadillo)
library(tidyverse)
library(MASS)
library(lubridate)
library(here)
library(splines)

sourceCpp(here("Code","code.cpp"))
source(here("Code","functions.R"))

# CLEAN DATA -----

data0 <- read.csv(here("Data","Ringlet_BNM_1970_2014_processed.csv"))

# clean data

{
  data <- data0 %>%
    rename(Site = Gridref) %>% 
    mutate(SamplUnit = paste(Site, Year, sep = " - ")) %>% 
    arrange(Year, Site)
  
  data_site <- data %>% 
    group_by(SamplUnit) %>% 
    filter(row_number() == 1) %>% 
    ungroup() 
  
  # occupancy covariates
  {
    data_site$time <- factor(data_site$Year)
    
    t_basis <- model.matrix(~ time - 1, data = data_site)
    
    df_time_occ <- unique(as.data.frame(t_basis))
    
    rownames(df_time_occ) <- sort(unique(data$Year))
    
    X_psi <- cbind(1, t_basis)
  }
  
  # detection covariates
  {
    data_week <- scale(data$Week)
    spl_degree_p <- 3
    
    w_basis <- bs(data_week, 
                  degree = spl_degree_p,
                  intercept = F)
    
    colnames(w_basis) <- paste0("Wspline",1:ncol(w_basis))
    
    df_time_det <- 
      bs(sort(unique(data_week)), 
         degree = spl_degree_p,
         Boundary.knots = attr(w_basis, "Boundary.knots"),
         intercept = F) %>% 
      as.matrix
    
    rownames(df_time_det) <- unique(data_week)
    
    X_p <- cbind(1, scale(data$listL), w_basis)
    colnames(X_p)[1] <- "P intercept"
    colnames(X_p)[2] <- "list length"  
  }
  
  y <- data$Occ
  
  M <- data %>% 
    group_by(SamplUnit) %>% 
    summarise(MM = n()) %>% 
    ungroup() %>% 
    dplyr::select(MM)
  
  covNames <- c(
    "Psi_intercept",
    colnames(t_basis), 
    colnames(X_p)
  )
  
  M <- M %>% as.vector %>% unlist
  
  n <- length(M)
  
  sumM <- c(0, cumsum(M)[-n]) %>% as.vector
  
  occ <- sapply(1:n, function(i){
    # print(i)
    if(sum(y[sumM[i] + 1:M[i]]) > 0){
      1
    } else {
      0
    }
  })
  
  ncov_psi <- ncol(X_psi)
  ncov_p <- ncol(X_p)
  
}

# FIT MODEL ------

useDiag <- F # set to true if using a diagonal covariance matrix
useSparse <- F # set to true if using a full covariance matrix
# otherwise, it will set a sparse matrix with blocks identified in the next
# chunk of code

# blocks
{
  # idx_covs <- 1:(ncov_psi + ncov_p)
  idx_covs <- c(1,ncov_psi + 1,2:ncov_psi,ncov_psi + 2:ncov_p)
  idx_covs_flipped <- order(idx_covs)
  
  idx_blocks_start <- c(1,3:length(idx_covs))
  idx_blocks_end <- c(2,3:length(idx_covs))
  # idx_blocks_start <- 1:length(idx_covs)
  # idx_blocks_end <- 1:length(idx_covs)
}

# dim L
{
  numCov <- ncov_psi + ncov_p
  if(useDiag){
    dimL <- numCov 
  } else if(useSparse) {
    dimL <- sum(
      sapply(seq_along(idx_blocks_start), function(i) {
        computeDimL(idx_blocks_end[i] - idx_blocks_start[i] + 1)    
      })
    )
  } else {
    dimL <- numCov * (numCov + 1) / 2
  }  
}

n_latent <- 2

# starting point 
{
  model_mean <- rep(0, numCov)
  model_cov <- rep(0, dimL)
}

# adam
{
  eta <- .025
  beta1 <- .9
  beta2 <- .999
  
  
  epsilon <- 1e-8
  
  v_tm1 <- rep(0, numCov + dimL)
  m_tm1 <- rep(0, numCov + dimL)
}

epochs <- 1000
loss_values <- rep(NA, epochs)
model_mean_output <- matrix(NA, numCov, epochs)
model_cov_output <- matrix(NA, dimL, epochs)
model_sigma_output <- matrix(NA, numCov * numCov, epochs)
model_diag_output <- matrix(NA, numCov, epochs)

for (i in 1:epochs) {
  
  print(paste0("Epoch = ",i))
  
  {
    eps_beta <- matrix(rnorm(n_latent / 2 * numCov), n_latent / 2, numCov)
    eps_beta <- rbind(eps_beta, - eps_beta)
  }
  
  if(useDiag){
    L <- createDiagMat(model_cov)
  } else if(useSparse) {
    L <- createSparseMat(model_cov, idx_blocks_start, idx_blocks_end)
  } else {
    L <- createCholMat(model_cov, numCov)
  }
  
  z_flipped <- sapply(1:n_latent, function(l){
    
    model_mean + L %*% eps_beta[l,]
    
  })
  
  z <- z_flipped[idx_covs_flipped,,drop=F]
  
  beta_psi_flipped <- z_flipped[1:ncov_psi,,drop=F]
  beta_p_flipped <- z_flipped[ncov_psi + 1:ncov_p,,drop=F]
  
  beta_psi <- z[1:ncov_psi,,drop=F]
  beta_p <- z[ncov_psi + 1:ncov_p,,drop=F]
  
  logit_psi <-  X_psi %*% beta_psi
  logit_p <-  X_p %*% beta_p
  
  list_gradient <- computegrad(model_cov, y, M, eps_beta, sumM, occ, logit_psi, logit_p, 
                               beta_psi, beta_p, X_psi, 
                               X_p, idx_covs - 1, idx_blocks_start - 1, 
                               idx_blocks_end - 1, n_latent, useDiag, useSparse)
  deltaldeltamu <- list_gradient$deltaldelteamu
  deltaldelteaL <- list_gradient$deltaldelteaL
  deltaldelteabeta <- list_gradient$deltaldelteabeta
  
  deltaldeltamu <- apply(deltaldeltamu, 2, mean)
  deltaldeltaL <- apply(deltaldelteaL, 2, mean)
  
  # prior
  deltaldeltamu <- deltaldeltamu - model_mean
  
  # entropy 
  diagElements <- extractDiagElement(numCov, idx_blocks_start, idx_blocks_end,
                                     useSparse, useDiag)  
  
  deltaldeltaL[diagElements] <- deltaldeltaL[diagElements] +
    derlogsoftplus(model_cov)[diagElements]
  
  # adam
  {
    g <- c(deltaldeltamu, deltaldeltaL)
    m_t <- beta1 * m_tm1 + (1 - beta1) * g
    v_t <- beta2 * v_tm1 + (1 - beta2) * g^2
    
    v_tm1 <- v_t
    m_tm1 <- m_t
    
    model_mean <- model_mean + eta * (m_t / (sqrt(v_t) + epsilon))[1:numCov]
    model_cov <- model_cov + eta * (m_t / (sqrt(v_t) + epsilon))[numCov + 1:dimL]
    
  }
  
  model_mean_output[,i] <- model_mean
  model_cov_output[,i] <- model_cov
  
  maxGrad <- max(abs(c(deltaldeltamu, deltaldeltaL)))
  loss_values[i] <- maxGrad
  
  if(i > 2){
    
    if(useDiag){
      L_current <- createDiagMat(model_cov)
      L_previous <- createDiagMat(model_cov_output[,i-1])
      
    } else if (useSparse) {
      L_current <- createSparseMat(model_cov, idx_blocks_start, idx_blocks_end)
      L_previous <- createSparseMat(model_cov_output[,i-1], idx_blocks_start, idx_blocks_end)
      
    } else {
      L_current <- createCholMat(model_cov, numCov)
      L_previous <- createCholMat(model_cov_output[,i-1], numCov)
      
    }
    
    Sigma_current <- L_current %*% t(L_current)
    Sigma_previous <- L_previous %*% t(L_previous)
    
    diag_current <- diag(Sigma_current)
    diag_previous <- diag(Sigma_previous)
    
    paramDecrease_mean <- abs( (model_mean_output[,i] - model_mean_output[,i-1]) / model_mean_output[,i-1]) 
    paramDecrease_cov <- abs( (diag_current - diag_previous) / diag_previous) 
    paramDecrease_cov <- paramDecrease_cov[!is.na(paramDecrease_cov)]
    
    model_sigma_output[,i] <- as.vector(Sigma_current)
    model_diag_output[,i] <- as.vector(diag(Sigma_current))
    
    print(paste0("Decrease Mean = ",max(paramDecrease_mean)))
    print(paste0("Decrease L = ",max(paramDecrease_cov)))
    print(paste0("Average Sd = ",mean(diag(Sigma_current))))
  }
  
}

# DIAGNOSTICS ----

subsetEpochs <- 50:epochs
subsetCovs <- 1:(ncov_psi + ncov_p)

model_mean_output[subsetCovs,subsetEpochs] %>% 
  t %>% 
  as.data.frame %>% 
  # 'colnames<-' (1:(ncov_psi + ncov_p)) %>% 
  'colnames<-' (subsetCovs) %>%
  mutate(Epoch = subsetEpochs) %>% 
  pivot_longer(!Epoch, names_to = "Covariate",
               values_to = "Value") %>% 
  ggplot(aes(x = Epoch,
             y = Value,
             color = Covariate)) + geom_line()

subsetCovs <- 1:dimL
model_sigma_output[subsetCovs,subsetEpochs] %>%
  t %>%
  as.data.frame %>%
  # 'colnames<-' (1:dimL) %>%
  'colnames<-' (subsetCovs) %>%
  mutate(Epoch = subsetEpochs) %>%
  pivot_longer(!Epoch, names_to = "Covariate",
               values_to = "Value") %>%
  ggplot(aes(x = Epoch,
             y = Value,
             color = Covariate)) + geom_line()

# OUTPUT -----

beta_samples <- simulateFromModel(model_mean, model_cov, useDiag, useSparse,
                                  idx_covs_flipped)

# yearly occupancy probabilities
{
  b_t_qtl <- generateTrend(beta_samples, df_time_occ)
  
  years <- rownames(df_time_occ)
  
  Y <- length(years)
  
  (plot_yearocc <- ggplot(data = NULL, aes(x = 1:Y,
                                           ymin = b_t_qtl[1,],
                                           ymax = b_t_qtl[2,],
                                           y = b_t_qtl[3,])) + 
      geom_errorbar(size = .7) + geom_point() + ylim(c(0,1)) + theme_bw() + 
      scale_x_continuous(breaks = 1:Y, labels = years,
                         name = "Year", limits = c(0.5,(Y+1))) + 
      ylab("Occupancy probability") + 
      theme(axis.text = element_text(angle = 90, size = 12),
            axis.title = element_text(size = 15)))
  
}

# detection probability pattern

{
  b_t_qtl <- generateTrend_det(beta_samples, df_time_det, year = 2000)
  
  week0 <- as.Date("01/01/2000", format = "%d/%m/%Y")
  
  idxWeeks <- 5 * 0:10
  labelWeeks <- week0 + days(7 * idxWeeks)
  labelWeeks <- format(labelWeeks, "%Y-%d-%b")
  labelWeeks <- gsub(pattern = "2000-", replacement = "", as.character(labelWeeks))
  
  plot_dettrend <- ggplot(data = NULL, aes(x = 1:53,
                                           ymin = b_t_qtl[1,],
                                           ymax = b_t_qtl[2,],
                                           y = b_t_qtl[3,])) + 
    geom_errorbar(size = .7) + 
    geom_point() + 
    ylim(c(0,.2)) +
    theme_bw() + 
    scale_x_continuous(breaks = idxWeeks + 1, 
                       labels = labelWeeks,
                       name = "Day",
                       limits = c(3,53)) + 
    ylab("Detection probability") + 
    theme(axis.text = element_text(angle = 90, size = 12),
          axis.title = element_text(size = 15))
  
}

# DETECTION COVARIATES ------

idx_cov <- which(covNames == "list length")

plotCovariatesQtl(beta_samples, idx_cov)


