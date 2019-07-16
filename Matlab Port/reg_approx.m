% Approximate Regularization Subproblem using LSQR
function s = reg_approx(J, r, delta)
    [~,n] = size(J);
    
    % Parameters
    TOL = 1e-6; % Tolerance
    MAXIT = 2*n; % Max iterations
    
    % Solve perturbed least squares problem
    s = lsqr([J; sqrt(delta)*eye(n)],[-r; zeros(n,1)],TOL,MAXIT); 
    
end