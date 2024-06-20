library(Rcpp)
library(RcppArmadillo)
library(tidyverse)
library(MASS)
library(lubridate)
library(here)
library(splines)
library(lubridate)
library(reshape2)

sourceCpp(here("Code","code.cpp"))
source(here("Code","functions.R"))

# CLEAN DATA -----

load("~/VBOccupancy/Data/ringletdata.rda")
load("~/VBOccupancy/Data/maxListLAreaRinglet.rda")

# clean data

{
 
  data$maxListlArea <- maxListlArea
  
  data <- data %>%
    rename(Site = Gridref) %>% 
    mutate(SamplUnit = paste(Site, Year, sep = " - ")) %>% 
    arrange(Year, Site)
  
  data$relativeListLength <- data$listL / data$maxListlArea
  
  meanDataEast <- mean(data$EAST); sdDataEast <- sd(data$EAST)
  meanDataNorth <- mean(data$NORTH); sdDataNorth <- sd(data$NORTH)
  sdBoth <- (sdDataEast + sdDataNorth) / 2
  data$EAST <- (data$EAST - meanDataEast) / sdBoth
  data$NORTH <- (data$NORTH - meanDataNorth) / sdBoth
  
  data$relativeListLength <- (data$relativeListLength -
                                mean(data$relativeListLength)) / sd(data$relativeListLength)
  
  
  data_date <- as.Date(data$Date)
  
  data_jDate <- as.POSIXlt(data_date, format = "%d%b%y")
  data$JulianDate <- data_jDate$yday
  data$JulianDateSq <- (data_jDate$yday)^2
  data$JulianDateCub <- (data_jDate$yday)^3
  
  
  data$JulianDate <- scale(data$JulianDate)
  data$JulianDateSq <- scale(data$JulianDateSq)
  data$JulianDateCub <- scale(data$JulianDateCub)
  
  data_site <- data %>% 
    group_by(SamplUnit) %>% 
    filter(row_number() == 1) %>% 
    ungroup() 
  
  data_site$time <- factor(data_site$Year)
  # t_basis <- model.matrix(~ time - 1 , data = data_site)
  t_basis <- model.matrix(~ time, data = data_site)
  df_time_occ <- unique(as.data.frame(t_basis))
  rownames(df_time_occ) <- sort(unique(data$Year))
  Y <- length(unique(data$Year))
  X_psi <- cbind(1, t_basis) %>% as.matrix
  # X_psi <- t_basis %>% as.matrix
  
  X_p <- cbind(1, data[,c(12,13,14,15)]) %>% as.matrix
  
}
  
{
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
  numCov <- ncov_psi + ncov_p
  
}

# FIT MODEL  ------

epochs <- 10
list_params <- fitModelSparse(y, M, sumM, occ, X_psi, X_p,
                              useDiag = F, useSparse = F,
                              epochs, n_latent = 2, verbose = T)
model_mean <- list_params$model_mean
TT <- list_params$TT
idx_covs_flipped <- list_params$idx_covs_flipped

# OUTPUT -----

beta_samples <- simulateFromModelSparse(model_mean, TT,
                                  idx_covs_flipped)

beta_psi_output <- beta_samples[,1:ncol(X_psi)]
beta_p_output <- beta_samples[,ncol(X_psi) + 1:ncol(X_p)]

# year plot 

{
  
  dataJulianDate <- data_jDate$yday
  dataJulianDateSq <- (data_jDate$yday)^2
  
  data_jDate <- as.POSIXlt(data_date, format = "%d%b%y")
  dataJulianDate <- data_jDate$yday
  dataJulianDateSq <- (data_jDate$yday)^2
  dataJulianDateCub <- (data_jDate$yday)^3
  
  mean_dataJulianDate <- mean(dataJulianDate)
  sd_dataJulianDate <- sd(dataJulianDate)
  mean_dataJulianDateSq <- mean(dataJulianDateSq)
  sd_dataJulianDateSq <- sd(dataJulianDateSq)
  mean_dataJulianDateCub <- mean(dataJulianDateCub)
  sd_dataJulianDateCub <- sd(dataJulianDateCub)
  year <- 2000
  # year0 <- 1970
  
  niterations <- nrow(beta_p_output)
  x <- 1:365
  captureProbWide <- t(sapply(1:niterations, function(i){
    
    b <- beta_p_output[i,c(1,3,4,5)]
    
    b_tilde <- c(b[1] - 
                   b[2] * mean_dataJulianDate / sd_dataJulianDate -
                   b[3] * mean_dataJulianDateSq / sd_dataJulianDateSq -
                   b[4] * mean_dataJulianDateCub / sd_dataJulianDateCub ,
                 b[2] / sd_dataJulianDate,
                 b[3] / sd_dataJulianDateSq,
                 b[4] / sd_dataJulianDateCub)
    
    logistic(b_tilde[1] + b_tilde[2] * x + b_tilde[3] * x * x + b_tilde[4] * x * x * x)
  }))
  
  captureProbLong <- melt(captureProbWide)
  captureProbMean <- apply(captureProbWide, 2, function(x){
    quantile(x, probs = c(0.025,0.5,0.975))
  })
  
  idxdatesOnAxis <- c(91, 121, 152, 182, 213, 244)
  datesOnAxis <- as.Date(idxdatesOnAxis, origin=as.Date(paste0(year,"-01-01")))
  
  x_grid <- 1:365
  
  (jdateplot <- ggplot() + 
      # geom_line(data = captureProbLong, aes(x = Var2, y = value,
      #                                       group = Var1), alpha = .0075) +
      geom_ribbon(data = NULL, 
                  aes(x = 1:365, 
                      ymin=captureProbMean[1,],
                      ymax=captureProbMean[3,]), fill="grey", alpha=0.9) +
      geom_line(data = NULL, aes(x = 1:365, y = captureProbMean[2,]), color = "black") + 
      scale_x_continuous(
        name = "Date",
        breaks = x_grid[idxdatesOnAxis],
        labels = format(datesOnAxis, "%d-%B")) + 
      scale_y_continuous(name = "Detection Probability"
                         # breaks = c(0, 1e-11, 1e-8, 1e-6,  1e-4, 1e-3, 0.01, 0.1, 0.4)
                         # labels = c(0, )
      ) +
      # coord_trans(y = "newscale") + 
      # coord_cartesian(xlim = c(130, 247), 
      coord_cartesian(xlim = c(110, 260),
                      ylim = c(0, 0.35)) +
      # ylim = c(0, 0.32)) +
      ggplot2::theme(
        plot.title = ggplot2::element_text(hjust = 0.5, size = 20),
        axis.title = ggplot2::element_text(size = 20, face = "bold"),
        axis.text = ggplot2::element_text(size = 11, face = "bold", angle = 0),
        panel.grid.major = ggplot2::element_line(colour = "grey", size = 0.15),
        panel.background = ggplot2::element_rect(fill = "white", color = "black")
      ) )
  
  # ggsave(paste0("jdate_plot",year,".png"), jdateplot,
         # width = 8, height = 5)
  
  
}

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


