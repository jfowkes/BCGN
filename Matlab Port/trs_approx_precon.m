% Preconditioned Steihaug-Toint Conjugate Gradient
function s = trs_approx_precon(J, gradf, delta)
    [~,n] = size(J);

    % Parameters
    TAU = 1e-5; % Tolerance
    MAXITER = 2*n; % Max iterations

    % Jacobi preconditioner
    Minv = 1./sum(J'.*J',2); % 1/diag(J^TJ)
    
    % Initialize
    s = zeros(n,1);
    g = gradf;
    v = Minv.*g;
    gv = dot(g,v);
    p = -v;

    k = 0;
    while norm(J'*(J*s) + gradf) > TAU*norm(gradf) && k < MAXITER

        % Calculate curvature
        kappa = p'*(J'*(J*p));

        % Check for zero curvature
        if kappa < 1e-30 % Find boundary solution
            sigma = roots([p'*p, 2*s'*p, s'*s-delta^2]); % Find quadratic roots
            if isempty(sigma)
                sigma = zeros(2,1); % failed step: delta too large
            end
            s = s + max(0,max(sigma(1),sigma(2))) * p; % take positive root
            return;
        end
            
        % Calculate step length for s and g
        alpha = gv/kappa;

        % Trust region active: boundary solution
        if norm(s + alpha*p) >= delta
            sigma = roots([p'*p, 2*s'*p, s'*s-delta^2]); % Find quadratic roots
            if isempty(sigma)
                sigma = zeros(2,1); % failed step: delta too large
            end
            s = s + max(0,max(sigma(1),sigma(2))) * p; % take positive root
            return;
        end

        % Take step for s and g
        s = s + alpha * p;
        g = g + alpha * J'*(J*p);
        v = Minv.*g;

        % Calculate step length for p
        gv_new = dot(g,v);
        beta = gv_new/gv;
        gv = gv_new;

        % Take step for p
        p = -v + beta * p;

        % Update iteration count
        k = k + 1;
        
    end
    
    % Trust region inactive: interior solution (or failed to converge)
   
end