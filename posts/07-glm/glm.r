box::use(
    stats[
        model.frame, model.matrix, model.response, 
        binomial, gaussian
    ]
)

glm_custom = function (formula, data, family = gaussian(), max_iter = 100, tol = 1e-7, eps = 1e-8) {
    if (missing(data)) {
        data = environment(formula)
    }
    
    if (is.character(family)) 
        family = get(family, mode = "function", envir = parent.frame())
    if (is.function(family)) 
        family = family()
    if (is.null(family$family)) {
        print(family)
        stop("'family' not recognized")
    }
    
    mf = model.frame(formula, data)
    X = model.matrix(formula, data)
    y = model.response(mf)
    n = nrow(X)
    p = ncol(X)
    
    beta = matrix(0, nrow = p, ncol = 1)
    
    converged = FALSE
    
    for (i in 1:max_iter) {
        # 1. Calculate linear predictor
        eta = X %*% beta
        
        # 2. Calculate fitted values
        mu = family$linkinv(eta)
        
        # 3. Calculate weights
        V = as.vector(family$variance(mu))
        gradient = as.vector(family$mu.eta(eta))
        w_vec = (gradient^2) / V
        
        # 4. Working response
        z = as.vector(eta) + (as.vector(y) - mu) / gradient
        
        # 5. Update coefficients by solving weighted least squares
        W = diag(as.vector(w_vec), n, n)
        beta_new = solve(t(X) %*% W %*% X) %*% t(X) %*% W %*% z
        
        # 6. Check convergence
        if (max(abs(beta_new - beta)) < tol) {
            beta = beta_new
            converged = TRUE
            break
        }
        
        # Final beta
        beta = beta_new
    }
    
    if (!converged) {
        warning("Algorithm did not converge in ", max_iter, " iterations")
    }
    
    list(
        coefficients = as.vector(beta),
        converged = converged,
        iterations = i
    )
}
