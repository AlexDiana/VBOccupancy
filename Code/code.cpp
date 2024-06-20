
#include <RcppArmadillo.h>

using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

double logistic(double x){
  return(1 / (1 + exp(-x)));
}

double loglogistic(double x){
  return(- log(1 + exp(-x)));
}

// [[Rcpp::export]]
double loglikocc(NumericVector y,
                 NumericVector M,
                 NumericVector sumM,
                 NumericVector occ,
                 NumericVector logit_psi,
                 NumericVector logit_p) {
  
  double loglik = 0;
  
  int n = occ.size();
  
  for(int i = 0; i < n; i++){
    
    if(occ[i] == 1){
      
      // loglik += R::dbinom(1, 1, logistic(logit_psi[i]), 1);
      loglik += loglogistic(logit_psi[i]);
      
      for(int k = 0; k < M[i]; k++){
        
        // loglik += R::dbinom(y[sumM[i] + k], 1, logistic(logit_p[sumM[i] + k]), 1);
        loglik += R::dbinom(y[sumM[i] + k], 1, logistic(logit_p[sumM[i] + k]), 1);
        
      }
      
    } else {
      
      double p_prod = 1;
      
      for(int k = 0; k < M[i]; k++){
        
        p_prod *= (1 - logistic(logit_p[sumM[i] + k]));
        
      }
      
      double psi = logistic(logit_psi[i]);
      
      loglik += log( psi * p_prod + (1 - psi)  );
      
    }
    
  }
  
  return loglik;
}

double entropy_fun(arma::vec beta,
                   arma::vec mu,
                   arma::mat Sigma){
  
  double logdet = log(arma::det(Sigma));
  
  // double term2 = arma::as_scalar(arma::trans(beta - mu) * arma::inv(Sigma) * (beta - mu));
    
  // return(-.5 * (logdet + term2));
  
  return(- .5 * logdet);
  
}

// [[Rcpp::export]]
List computeELBO(NumericVector y,
                 NumericVector M,
                 NumericVector sumM,
                 NumericVector occ,
                 NumericMatrix logit_psi,
                 NumericMatrix logit_p,
                 arma::mat z,
                 arma::vec mu,
                 arma::mat Sigma){
  
  int n_latent = z.n_cols;
  
  double elbo = 0;
  double loglik = 0;
  double entropy = 0;
  for(int l = 0; l < n_latent; l++){
    
    NumericVector logit_psi_l = logit_psi(_, l);
    NumericVector logit_p_l = logit_p(_, l);
    
    arma::vec z_l = z.col(l);
    
    loglik += (1.0 / n_latent) * loglikocc(y, M, sumM, occ, logit_psi_l, logit_p_l);
    entropy += (1.0 / n_latent) *  entropy_fun(z_l, mu, Sigma);
    
  }
  
  return List::create(
    Named("loglik") = loglik,
    Named("entropy") = entropy
  );
  // return(elbo);
}


// [[Rcpp::export]]
double logliknorm(NumericVector y,
                 double sigma,
                 arma::vec Xbeta) {
  
  double loglik = 0;
  
  int n = y.size();
  
  for(int i = 0; i < n; i++){
    
    loglik += R::dnorm(y[i], Xbeta[i], sigma, 1);
    
  }
  
  return loglik;
}

// [[Rcpp::export]]
List computeELBOsimple(NumericVector y,
                       arma::mat X,
                       double sigma,
                         arma::mat z,
                         arma::vec mu,
                         arma::mat Sigma){
  
  int n_latent = z.n_cols;
  
  double elbo = 0;
  double loglik = 0;
  double entropy = 0;
  for(int l = 0; l < n_latent; l++){
    
    arma::vec Xbeta = X * z.col(l);
    
    
    loglik += (1.0 / n_latent) * logliknorm(y, sigma, Xbeta);
    entropy += (1.0 / n_latent) *  entropy_fun(z.col(l), mu, Sigma);
    
  }
  
  return List::create(
    Named("loglik") = loglik,
    Named("entropy") = entropy
  );
  // return(elbo);
}

arma::vec sigmoid(arma::vec x){
  return(1 / (1 + exp(-x)));
}

// [[Rcpp::export]]
arma::vec dersoftplus(arma::vec x){
  return exp(x) / (1 + exp(x));
}

// [[Rcpp::export]]
arma::vec derlogsoftplus(arma::vec x){
  return exp(x) / ((1 + exp(x)) % log(1 + exp(x)));
  // return (- 1 / (exp(x) + 1));
}

// [[Rcpp::export]]
arma::vec f(arma::rowvec x, arma::vec beta){
  double xbeta = dot(x, beta);
  
  arma::vec x_vec = arma::conv_to<arma::vec>::from(x);
  
  return (x_vec * exp(- xbeta)) / pow((1 + exp(- xbeta)),2);
}

// [[Rcpp::export]]
arma::vec derloglogit(arma::rowvec x, arma::vec beta){
  
  double xbeta = dot(x, beta);
  
  double denom = exp(xbeta) + 1;
  
  arma::vec x_vec = arma::conv_to<arma::vec>::from(x);
  
  return x_vec / denom;
}

// [[Rcpp::export]]
arma::vec derlog1mlogit(arma::rowvec x, arma::vec beta){
  
  double xbeta = dot(x, beta);
  
  double denom = exp(xbeta) + 1;
  
  arma::vec x_vec = arma::conv_to<arma::vec>::from(x);
  
  return - x_vec * exp(xbeta) / denom;
}

arma::vec computeSumDerivs(arma::mat& X_p, arma::vec beta_p, arma::vec& logit_p,
                           int start, int end){
  
  arma::vec sumDerivs = arma::zeros(X_p.n_cols);
  
  for(int i = start; i < end; i++){
    
    sumDerivs += (
      - f(X_p.row(i), beta_p) * (exp(logit_p[i]) + 1)
    );
    
  }
  
  return sumDerivs;
}

// [[Rcpp::export]]
arma::cube create_identity_deltabetadeltaL(int n_latent, int numCov,
                                           arma::vec model_cov, 
                                           arma::mat eps_beta) {
  
  arma::cube deltabetadelta(n_latent, numCov, numCov);
  
  for (int i = 0; i < numCov; ++i) {
    
    arma::vec eps_beta_i = eps_beta.col(i);
    arma::vec der_softplus = dersoftplus(model_cov); 
    
    for (int j = 0; j < n_latent; ++j) {
      // deltabetadelta(j, i, i) = eps_beta_i[j] * der_softplus[i];
      deltabetadelta(j, i, i) = eps_beta_i[j];
    }
  }
  
  return deltabetadelta;
}

// [[Rcpp::export]]
arma::cube create_general_deltabetadeltaL(int n_latent, int numCov,
                                          arma::vec model_cov, 
                                          arma::mat eps_beta) {
  
  int numElemL = model_cov.size();
  
  arma::cube deltabetadelta(n_latent, numCov, numElemL);
  
  arma::vec der_softplus = dersoftplus(model_cov); 
  
  for (int l = 0; l < n_latent; ++l) {
    
    for (int i = 0; i < numCov; ++i) {
      
      int start = i * (i + 1) / 2.0;
      int end = start + (i + 1);
      
      for (int j = start; j < end; j++){
        
        deltabetadelta(l, i, j) = eps_beta(l, j - start);
        
        if(j == (end - 1)){
          
          // deltabetadelta(l, i, j) = deltabetadelta(l, i, j) *
            // der_softplus[i];
          
        }
        
      }
      
    }
  }
  
  // # deltabetadeltaL = torch.zeros((self.n_latent, numCov, dimL))
    // # for i in range(numCov):
    // #     numElems = i + 1
    // #     startIdx = int(i * (i + 1) / 2.0)
    // #     deltabetadeltaL[:, i, range(startIdx, startIdx + numElems)] = torch.tensor(eps_beta[:,range(numElems)],
    // #                                                                             dtype=torch.float).squeeze(2)
    // #     deltabetadeltaL[:, i, startIdx + numElems - 1] = deltabetadeltaL[:, i, startIdx + numElems - 1] * \
    // #                                                   dersoftplus(model.cov[startIdx + numElems - 1])
    
    return deltabetadelta;
}

int computeDimL(int n){
  
  return (n * (n + 1) / 2);
  
}

// [[Rcpp::export]]
arma::cube create_sparse_deltabetadeltaL(int n_latent, int numCov,
                                         arma::vec model_cov, 
                                         arma::mat eps_beta,
                                         IntegerVector idx_start,
                                         IntegerVector idx_end) {
  
  int numElemL = model_cov.size();
  int numBlocks = idx_start.size();
  
  arma::cube deltabetadelta(n_latent, numCov, numElemL);
  
  arma::vec der_softplus = dersoftplus(model_cov); 
  
  for (int l = 0; l < n_latent; ++l) {
    
    int lengthBlocks = 0;
    int numCovs = 0;
    
    for(int k = 0; k < numBlocks; k++){
      
      int numCovBlock = idx_end[k] - idx_start[k] + 1;
      
      for (int i = 0; i < numCovBlock; ++i) {
        
        int start = i * (i + 1) / 2.0;
        int end = start + (i + 1);
        
        for (int j = start; j < end; j++){
          
          deltabetadelta(l, numCovs + i, lengthBlocks + j) = 
            eps_beta(l, numCovs + j - start);
          
          if(j == (end - 1)){
            
            // deltabetadelta(l, numCovs + i, lengthBlocks + j) =
            //   deltabetadelta(l, numCovs + i, lengthBlocks + j) *
            //   der_softplus[numCovs + j];
            
          }
          
        }
        
      }  
      
      numCovs += numCovBlock;
      lengthBlocks += computeDimL(numCovBlock);
      
    }
    
  }
  
  // # deltabetadeltaL = torch.zeros((self.n_latent, numCov, dimL))
    // # for i in range(numCov):
    // #     numElems = i + 1
    // #     startIdx = int(i * (i + 1) / 2.0)
    // #     deltabetadeltaL[:, i, range(startIdx, startIdx + numElems)] = torch.tensor(eps_beta[:,range(numElems)],
    // #                                                                             dtype=torch.float).squeeze(2)
    // #     deltabetadeltaL[:, i, startIdx + numElems - 1] = deltabetadeltaL[:, i, startIdx + numElems - 1] * \
    // #                                                   dersoftplus(model.cov[startIdx + numElems - 1])
    
    return deltabetadelta;
}

arma::vec computedeltaldelteabeta(
    arma::vec y,
    arma::vec occ,
    arma::vec M,
    arma::vec sumM,
    arma::mat X_psi,
    arma::mat X_p,
    arma::vec beta_psi,
    arma::vec beta_p,
    arma::vec logit_psi,
    arma::vec logit_p){
  
  int n = occ.size();
  int ncov_psi = X_psi.n_cols;
  int ncov_p = X_p.n_cols;
  int numCov = beta_psi.size() + beta_p.size();
  
  arma::vec deltaldelteabeta = arma::vec(numCov);
  
  // arma::vec beta_psi_l = beta_psi.col(l);
  // arma::vec beta_p_l = beta_p.col(l);
  // 
  // arma::vec logit_psi_l = logit_psi.col(l);
  // arma::vec logit_p_l = logit_p.col(l);
  
  for(int i = 0; i < n; i++){
    
    if(occ[i] == 1){
      
      deltaldelteabeta.subvec(0, ncov_psi - 1) +=
        derloglogit(X_psi.row(i), beta_psi);
      
      for(int k = 0; k < M[i]; k++){
        
        if(y[sumM[i] + k] == 1){
          
          deltaldelteabeta.subvec(ncov_psi, ncov_psi + ncov_p - 1) +=
            derloglogit(X_p.row(sumM[i] + k), beta_p);
          
        } else {
          
          deltaldelteabeta.subvec(ncov_psi, ncov_psi + ncov_p - 1) +=
            derlog1mlogit(X_p.row(sumM[i] + k), beta_p);
          
        }
        
      }
      
    } else {
      
      double p_prod = 1;
      
      for(int k = 0; k < M[i]; k++){
        
        p_prod *= (1 - logistic(logit_p[sumM[i] + k]));
        
      }
      
      double psi = logistic(logit_psi[i]);
      
      double l_i = psi * p_prod + (1 - psi);
      
      int start = sumM[i];
      int end = sumM[i] + M[i];
      
      arma::vec sumderivs = computeSumDerivs(X_p, beta_p, logit_p, start, end);
      
      deltaldelteabeta.subvec(0, ncov_psi - 1) +=
        f(X_psi.row(i), beta_psi) * (p_prod - 1) / l_i;
      
      deltaldelteabeta.subvec(ncov_psi, ncov_psi + ncov_p - 1) +=
        sumderivs * psi * p_prod / l_i;
      
    }
    
  }
  
  return deltaldelteabeta;
}

arma::vec computedeltaldelteabeta_single(
    arma::vec y,
    arma::mat X,
    double sigma,
    arma::vec beta){
  
  int n = y.size();
  int numCov = X.n_cols;
  
  arma::vec deltaldelteabeta = arma::vec(numCov);
  
  arma::vec Xbeta = X * beta;
  
  for(int i = 0; i < n; i++) {
    
    arma::vec x_vec = arma::conv_to<arma::vec>::from(X.row(i));
    
    deltaldelteabeta +=
    (1 / (sigma*sigma)) * (y[i] - Xbeta[i]) * x_vec;
    
  }
  
  return deltaldelteabeta;
}

arma::vec computedeltaldelteabeta_single_logistic(
    arma::vec y,
    arma::mat X,
    arma::vec beta){
  
  int n = y.size();
  int numCov = X.n_cols;
  
  arma::vec deltaldelteabeta = arma::vec(numCov);
  
  arma::vec Xbeta = X * beta;
  
  for(int i = 0; i < n; i++) {
    
    arma::vec x_vec = arma::conv_to<arma::vec>::from(X.row(i));
    
    if(y[i] == 0){
      deltaldelteabeta +=  derlog1mlogit(X.row(i), beta); 
    } else {
      deltaldelteabeta +=  derloglogit(X.row(i), beta); 
    }
    // deltaldelteabeta +=
      // (1 / (sigma*sigma)) * (y[i] - Xbeta[i]) * x_vec;
    
  }
  
  return deltaldelteabeta;
}
// 
// arma::vec computedeltaldelteabeta_single(
//     arma::vec y,
//     arma::mat X,
//     arma::vec beta,
//     double sigma){
//   
//   int n = y.size();
//   int numCov = X.n_cols;
//   
//   arma::vec deltaldelteabeta = arma::vec(numCov);
//   
//   arma::vec Xbeta = X * beta;
//   
//   for(int i = 0; i < n; i++) {
//     
//     arma::vec x_vec = arma::conv_to<arma::vec>::from(X.row(i));
//     
//     deltaldelteabeta +=
//       (1 / (sigma*sigma)) * (y[i] - Xbeta[i]) * x_vec;
//     
//   }
//   
//   return deltaldelteabeta;
// }


// [[Rcpp::export]]
arma::mat computedeltaldeltabeta(
    NumericVector y,
    NumericVector M,
    arma::mat eps_beta,
    NumericVector sumM,
    NumericVector occ,
    arma::mat logit_psi,
    arma::mat logit_p,
    arma::mat beta_psi,
    arma::mat beta_p,
    arma::mat X_psi,
    arma::mat X_p,
    arma::uvec idx_covs,
    IntegerVector idx_blocks_start,
    IntegerVector idx_blocks_end,
    int n_latent) {
  
  int ncov_psi = X_psi.n_cols;
  int ncov_p = X_p.n_cols;
  
  int numCov = ncov_psi + ncov_p;
  
  int n = occ.size();
  
  arma::mat deltaldelteabeta(n_latent, numCov);
  
  // arma::cube deltabetadeltam(n_latent, numCov, numCov);
  // 
  // for (int i = 0; i < n_latent; ++i) {
  //   deltabetadeltam.row(i) = arma::eye<arma::mat>(numCov, numCov);
  // }
  // 
  // arma::cube deltabetadeltaL;
  // if(useDiag){
  //   deltabetadeltaL = create_identity_deltabetadeltaL(n_latent, numCov,
  //                                                     model_cov,
  //                                                     eps_beta);
  //   // deltabetadeltaL = 
  //   //   arma::zeros(n_latent, numCov, numCov);
  //   // 
  //   // for (int i = 0; i < numCov; ++i) {
  //   //   
  //   //   arma::vec eps_beta_i = eps_beta.col(i);
  //   //   arma::vec der_softplus = dersoftplus(model_cov); 
  //   //   
  //   //   for (int j = 0; j < n_latent; ++j) {
  //   //     deltabetadeltaL(j, i, i) = eps_beta_i[j] * der_softplus[i];
  //   //   }
  //   // }
  //   
  // } else if (useSparse) {
  //   deltabetadeltaL = create_sparse_deltabetadeltaL(n_latent, numCov,
  //                                                   model_cov,
  //                                                   eps_beta,
  //                                                   idx_blocks_start,
  //                                                   idx_blocks_end);
  // }
  // else {
  //   deltabetadeltaL = create_general_deltabetadeltaL(n_latent, numCov,
  //                                                    model_cov,
  //                                                    eps_beta);
  // }
  // 
  // arma::mat deltaldeltamu(n_latent, numCov);
  // arma::mat deltaldeltaL(n_latent, numElemL);
  // arma::mat deltaldelteabeta(n_latent, numCov);
  // // arma::cube deltabetadelteaL(n_latent, numCov, numCov);
  // 
  // arma::cube sumderivscube(n_latent, n, ncov_p);
  
  for(int l = 0; l < n_latent; l++){
    // Rcout << l << std::endl;
    // arma::vec deltaldelteabeta_l = arma::vec(numCov);
    // 
    arma::vec beta_psi_l = beta_psi.col(l);
    arma::vec beta_p_l = beta_p.col(l);
    
    arma::vec logit_psi_l = logit_psi.col(l);
    arma::vec logit_p_l = logit_p.col(l);
    
    arma::vec deltaldelteabeta_l = computedeltaldelteabeta(y, occ, M, sumM,
                                                           X_psi, X_p,
                                                           beta_psi_l, beta_p_l,
                                                           logit_psi_l, logit_p_l);
    
    // flip deltaldelteabeta
    
    deltaldelteabeta_l = deltaldelteabeta_l.elem(idx_covs);
    deltaldelteabeta.row(l) = arma::conv_to<arma::rowvec>::from(deltaldelteabeta_l);
    
  }
  
  return deltaldelteabeta;
  
}

// [[Rcpp::export]]
List computegrad_usingfun(
  arma::vec model_cov, 
  NumericVector y,
  NumericVector M,
  arma::mat eps_beta,
  NumericVector sumM,
  NumericVector occ,
  arma::mat logit_psi,
  arma::mat logit_p,
  arma::mat beta_psi,
  arma::mat beta_p,
  arma::mat X_psi,
  arma::mat X_p,
  arma::uvec idx_covs,
  IntegerVector idx_blocks_start,
  IntegerVector idx_blocks_end,
  int n_latent,
  bool useDiag,
  bool useSparse) {
  
  int ncov_psi = X_psi.n_cols;
  int ncov_p = X_p.n_cols;
  
  int numCov = ncov_psi + ncov_p;
  int numElemL = model_cov.size();
  
  int n = occ.size();
  
  arma::cube deltabetadeltam(n_latent, numCov, numCov);
  
  for (int i = 0; i < n_latent; ++i) {
    deltabetadeltam.row(i) = arma::eye<arma::mat>(numCov, numCov);
  }
  
  arma::cube deltabetadeltaL;
  if(useDiag){
    deltabetadeltaL = create_identity_deltabetadeltaL(n_latent, numCov,
                                                      model_cov,
                                                      eps_beta);
    // deltabetadeltaL = 
    //   arma::zeros(n_latent, numCov, numCov);
    // 
    // for (int i = 0; i < numCov; ++i) {
    //   
    //   arma::vec eps_beta_i = eps_beta.col(i);
    //   arma::vec der_softplus = dersoftplus(model_cov); 
    //   
    //   for (int j = 0; j < n_latent; ++j) {
    //     deltabetadeltaL(j, i, i) = eps_beta_i[j] * der_softplus[i];
    //   }
    // }
    
  } else if (useSparse) {
    deltabetadeltaL = create_sparse_deltabetadeltaL(n_latent, numCov,
                                                    model_cov,
                                                    eps_beta,
                                                    idx_blocks_start,
                                                    idx_blocks_end);
  }
  else {
    deltabetadeltaL = create_general_deltabetadeltaL(n_latent, numCov,
                                                     model_cov,
                                                     eps_beta);
  }
  
  arma::mat deltaldeltamu(n_latent, numCov);
  arma::mat deltaldeltaL(n_latent, numElemL);
  arma::mat deltaldelteabeta(n_latent, numCov);
  // arma::cube deltabetadelteaL(n_latent, numCov, numCov);
  
  arma::cube sumderivscube(n_latent, n, ncov_p);
  
  for(int l = 0; l < n_latent; l++){
    // Rcout << l << std::endl;
    // arma::vec deltaldelteabeta_l = arma::vec(numCov);
    // 
    arma::vec beta_psi_l = beta_psi.col(l);
    arma::vec beta_p_l = beta_p.col(l);

    arma::vec logit_psi_l = logit_psi.col(l);
    arma::vec logit_p_l = logit_p.col(l);
    
    arma::vec deltaldelteabeta_l = computedeltaldelteabeta(y, occ, M, sumM,
                                                           X_psi, X_p,
                                                           beta_psi_l, beta_p_l,
                                                           logit_psi_l, logit_p_l);
    
    // flip deltaldelteabeta
    
    deltaldelteabeta_l = deltaldelteabeta_l.elem(idx_covs);
    deltaldelteabeta.row(l) = arma::conv_to<arma::rowvec>::from(deltaldelteabeta_l);
    
    // 
    arma::mat deltabetadeltam_l = deltabetadeltam.row(l);
    arma::rowvec deltabetadeltamm_l = arma::conv_to<arma::rowvec>::from(
      arma::trans(deltaldelteabeta_l) * deltabetadeltam_l 
    );
    deltaldeltamu.row(l) = deltabetadeltamm_l;
    
    // 
    arma::mat deltabetadeltaL_l = deltabetadeltaL.row(l);
    arma::rowvec deltabetadeltaLL_l = arma::conv_to<arma::rowvec>::from(
      arma::trans(deltaldelteabeta_l) * deltabetadeltaL_l
    );
    
    deltaldeltaL.row(l) = deltabetadeltaLL_l;
    // deltabetadelteaL.subcube(arma::span(l),
                             // arma::span(),
                             // arma::span()) = deltabetadeltaL_l;
    
  }
  
  return List::create(
    Named("deltaldeltamu") = deltaldeltamu,
    Named("deltaldeltaL") = deltaldeltaL,
    Named("deltabetadeltaL") = deltabetadeltaL,
    Named("deltaldelteabeta") = deltaldelteabeta
  );
  
}

// [[Rcpp::export]]
List computegrad_usingfun_simple(
  arma::vec model_cov, 
  NumericVector y,
  arma::mat eps_beta,
  arma::mat beta,
  arma::mat X,
  double sigma,
  arma::uvec idx_covs,
  IntegerVector idx_blocks_start,
  IntegerVector idx_blocks_end,
  int n_latent,
  bool useDiag,
  bool useSparse) {
  
  int ncov = X.n_cols;
  
  int numCov = ncov;
  int numElemL = model_cov.size();
  
  int n = y.size();
  
  arma::cube deltabetadeltam(n_latent, numCov, numCov);
  
  for (int i = 0; i < n_latent; ++i) {
    deltabetadeltam.row(i) = arma::eye<arma::mat>(numCov, numCov);
  }
  
  arma::cube deltabetadeltaL;
  if(useDiag){
    deltabetadeltaL = create_identity_deltabetadeltaL(n_latent, numCov,
                                                      model_cov,
                                                      eps_beta);
    
  } else if (useSparse) {
    deltabetadeltaL = create_sparse_deltabetadeltaL(n_latent, numCov,
                                                    model_cov,
                                                    eps_beta,
                                                    idx_blocks_start,
                                                    idx_blocks_end);
  }
  else {
    deltabetadeltaL = create_general_deltabetadeltaL(n_latent, numCov,
                                                     model_cov,
                                                     eps_beta);
  }
  
  arma::mat deltaldeltamu(n_latent, numCov);
  arma::mat deltaldeltaL(n_latent, numElemL);
  arma::mat deltaldelteabeta(n_latent, numCov);
  // arma::cube deltabetadelteaL(n_latent, numCov, numCov);
  
  for(int l = 0; l < n_latent; l++){

    arma::vec beta_l = beta.col(l);

    arma::vec deltaldelteabeta_l = computedeltaldelteabeta_single(y, X, sigma,
                                                                  beta_l);
    // arma::vec deltaldelteabeta_l = computedeltaldelteabeta_single(y, X,
    //                                                               beta_l);
    
    // flip deltaldelteabeta
    
    deltaldelteabeta_l = deltaldelteabeta_l.elem(idx_covs);
    deltaldelteabeta.row(l) = arma::conv_to<arma::rowvec>::from(deltaldelteabeta_l);
    
    // 
    arma::mat deltabetadeltam_l = deltabetadeltam.row(l);
    arma::rowvec deltabetadeltamm_l = arma::conv_to<arma::rowvec>::from(
      arma::trans(deltaldelteabeta_l) * deltabetadeltam_l 
    );
    deltaldeltamu.row(l) = deltabetadeltamm_l;
    
    // 
    arma::mat deltabetadeltaL_l = deltabetadeltaL.row(l);
    arma::rowvec deltabetadeltaLL_l = arma::conv_to<arma::rowvec>::from(
      arma::trans(deltaldelteabeta_l) * deltabetadeltaL_l
    );
    
    deltaldeltaL.row(l) = deltabetadeltaLL_l;
    // deltabetadelteaL.subcube(arma::span(l),
                             // arma::span(),
                             // arma::span()) = deltabetadeltaL_l;
    
  }
  
  return List::create(
    Named("deltaldeltamu") = deltaldeltamu,
    Named("deltaldeltaL") = deltaldeltaL,
    Named("deltabetadeltaL") = deltabetadeltaL,
    Named("deltaldelteabeta") = deltaldelteabeta
  );
  
}

// [[Rcpp::export]]
List computegrad(
  arma::vec model_cov, 
  NumericVector y,
  NumericVector M,
  arma::mat eps_beta,
  NumericVector sumM,
  NumericVector occ,
  arma::mat logit_psi,
  arma::mat logit_p,
  arma::mat beta_psi,
  arma::mat beta_p,
  arma::mat X_psi,
  arma::mat X_p,
  arma::uvec idx_covs,
  IntegerVector idx_blocks_start,
  IntegerVector idx_blocks_end,
  int n_latent,
  bool useDiag,
  bool useSparse) {
  
  int ncov_psi = X_psi.n_cols;
  int ncov_p = X_p.n_cols;
  
  int numCov = ncov_psi + ncov_p;
  int numElemL = model_cov.size();
  
  int n = occ.size();
  
  arma::cube deltabetadeltam(n_latent, numCov, numCov);
  
  for (int i = 0; i < n_latent; ++i) {
    deltabetadeltam.row(i) = arma::eye<arma::mat>(numCov, numCov);
  }
  
  arma::cube deltabetadeltaL;
  if(useDiag){
    deltabetadeltaL = create_identity_deltabetadeltaL(n_latent, numCov,
                                                      model_cov,
                                                      eps_beta);
    
  } else if (useSparse) {
    deltabetadeltaL = create_sparse_deltabetadeltaL(n_latent, numCov,
                                                    model_cov,
                                                    eps_beta,
                                                    idx_blocks_start,
                                                    idx_blocks_end);
  }
  else {
    deltabetadeltaL = create_general_deltabetadeltaL(n_latent, numCov,
                                                     model_cov,
                                                     eps_beta);
  }
  
  // arma::cube deltabetadeltaL(n_latent, numCov, numCov);
  // 
    // for (int i = 0; i < numCov; ++i) {
      //   
        //   arma::vec eps_beta_i = eps_beta.col(i);
        //   arma::vec der_softplus = dersoftplus(model_cov); 
        //   
          //   for (int j = 0; j < n_latent; ++j) {
            //     deltabetadeltaL(j, i, i) = eps_beta_i[j] * der_softplus[j];
            //   }
        // }
  
  arma::mat deltaldelteamu(n_latent, numCov);
  arma::mat deltaldelteaL(n_latent, numElemL);
  
  arma::mat deltaldelteabeta(n_latent, numCov);
  
  arma::cube sumderivscube(n_latent, n, ncov_p);
  
  for(int l = 0; l < n_latent; l++){
    
    arma::vec deltaldelteabeta_l = arma::vec(numCov);
    
    arma::vec beta_psi_l = beta_psi.col(l);
    arma::vec beta_p_l = beta_p.col(l);
    
    arma::vec logit_psi_l = logit_psi.col(l);
    arma::vec logit_p_l = logit_p.col(l);
    
    for(int i = 0; i < n; i++){
      
      if(occ[i] == 1){
        
        deltaldelteabeta_l.subvec(0, ncov_psi - 1) +=
          derloglogit(X_psi.row(i), beta_psi_l);
        
        for(int k = 0; k < M[i]; k++){
          
          if(y[sumM[i] + k] == 1){
            
            deltaldelteabeta_l.subvec(ncov_psi, ncov_psi + ncov_p - 1) +=
              derloglogit(X_p.row(sumM[i] + k), beta_p_l);
            
          } else {
            
            deltaldelteabeta_l.subvec(ncov_psi, ncov_psi + ncov_p - 1) +=
              derlog1mlogit(X_p.row(sumM[i] + k), beta_p_l);
            
          }
          
        }
        
      } else {
        
        double p_prod = 1;
        
        for(int k = 0; k < M[i]; k++){
          
          p_prod *= (1 - logistic(logit_p_l[sumM[i] + k]));
          
        }
        
        double psi = logistic(logit_psi_l[i]);
        
        double l_i = psi * p_prod + (1 - psi);
        
        int start = sumM[i];
        int end = sumM[i] + M[i];
        
        arma::vec sumderivs = computeSumDerivs(X_p, beta_p_l, logit_p_l, start, end);
        
        sumderivscube.subcube(arma::span(l), arma::span(i), arma::span()) = sumderivs;
        
        deltaldelteabeta_l.subvec(0, ncov_psi - 1) +=
          f(X_psi.row(i), beta_psi_l) * (p_prod - 1) / l_i;
        
        deltaldelteabeta_l.subvec(ncov_psi, ncov_psi + ncov_p - 1) +=
          sumderivs * psi * p_prod / l_i;
        
      }
      
    }
    
    // flip deltaldelteabeta
    
    deltaldelteabeta_l = deltaldelteabeta_l.elem(idx_covs);
    
    arma::mat deltabetadeltam_l = deltabetadeltam.row(l);
    arma::rowvec deltabetadeltamm_l = arma::conv_to<arma::rowvec>::from(
      arma::trans(deltaldelteabeta_l) * deltabetadeltam_l 
    );
    deltaldelteamu.row(l) = deltabetadeltamm_l;
    
    arma::mat deltabetadeltaL_l = deltabetadeltaL.row(l);
    arma::rowvec deltabetadeltaLL_l = arma::conv_to<arma::rowvec>::from(
      arma::trans(deltaldelteabeta_l) * deltabetadeltaL_l
    );
    deltaldelteaL.row(l) = deltabetadeltaLL_l;
    
    deltaldelteabeta.row(l) = arma::conv_to<arma::rowvec>::from(deltaldelteabeta_l);
  }
  
  return List::create(
    Named("deltaldelteamu") = deltaldelteamu,
    Named("deltaldelteaL") = deltaldelteaL,
    Named("deltabetadeltaL") = deltabetadeltaL,
    Named("deltaldelteabeta") = deltaldelteabeta
  );
}

// CREATE GRID

// [[Rcpp::export]]
bool isPointInBandRight(arma::mat X_tilde, arma::vec x_grid, arma::vec y_grid, int i, int j){
  
  for(int k = 0; k < X_tilde.n_rows; k++){
    
    if((X_tilde(k,1) < y_grid[j + 1]) & (X_tilde(k,1) > y_grid[j - 1])){
      if(X_tilde(k,0) < x_grid[i + 1]){
        return(true);
      }
    } 
    
  }
  
  return(false);
}

// [[Rcpp::export]]
bool isPointInBandLeft(arma::mat X_tilde, arma::vec x_grid, arma::vec y_grid, int i, int j) {
  
  for(int k = 0; k < X_tilde.n_rows; k++){
    
    if((X_tilde(k,1) < y_grid[j + 1]) & (X_tilde(k,1) > y_grid[j - 1])){
      if(X_tilde(k,0) > x_grid[i - 1]){
        return(true);
      }
    } 
    
  }
  
  return(false);
}

// [[Rcpp::export]]
bool isPointInBandUp(arma::mat X_tilde, arma::vec x_grid, arma::vec y_grid, int i, int j){
  
  for(int k = 0; k < X_tilde.n_rows; k++){
    
    if((X_tilde(k,0) < x_grid[i + 1]) & (X_tilde(k,0) > x_grid[i - 1])){
      if(X_tilde(k,1) > y_grid[j-1]){
        return(true);
      }
    }
    
  }
  
  return(false);
  
}

// [[Rcpp::export]]
bool isPointInBandDown(arma::mat X_tilde, arma::vec x_grid, arma::vec y_grid, int i, int j){
  
  for(int k = 0; k < X_tilde.n_rows; k++){
    
    if((X_tilde(k,0) < x_grid[i + 1]) & (X_tilde(k,0) > x_grid[i - 1])){
      if(X_tilde(k,1) < y_grid[j+1]){
        return(true);
      }
    }
    
  }
  
  return(false);
  
}

// [[Rcpp::export]]
IntegerVector findClosestPoint(arma::mat XY_sp, arma::mat X_tilde){
  
  IntegerVector closestPoint(XY_sp.n_rows);
  
  for(int k = 0; k < XY_sp.n_rows; k++){
    
    double newDistance = 0;
    double minDistance = exp(50);
    int bestIndex = 0;
    
    for(int i = 0; i < X_tilde.n_rows; i++){
      newDistance = pow(X_tilde(i, 0) - XY_sp(k, 0), 2) + pow(X_tilde(i, 1) - XY_sp(k, 1), 2);
      
      if(newDistance < minDistance){
        minDistance = newDistance;
        bestIndex = i + 1;
      }
    }
    
    closestPoint[k] = bestIndex;
    
  }
  
  return(closestPoint);
}
