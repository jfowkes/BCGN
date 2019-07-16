% Regularization Update (same as TR update with grow/shrink swapped)
function [x, sigma] = reg_update(f, x, s, Delta_m, sigma)

    % Regularization parameters
    ETA1 = 0.25;
    ETA2 = 0.75;
    GAMMA1 = 2.;
    GAMMA2 = 0.5;
    SIGMA_MIN = 1e-150;
    SIGMA_MAX = 1e150;

    % Evaluate sufficient decrease
    rho = (f(x) - f(x+s))/Delta_m;

    % Accept trial point
    if rho >= ETA1
        x = x + s;
    end

    % Update regularization parameter
    if rho < ETA1
        sigma = sigma*GAMMA1;
        sigma = min(sigma,SIGMA_MAX);
    elseif rho >= ETA2
        sigma = sigma*GAMMA2;
        sigma = max(sigma,SIGMA_MIN);
    end    
      
end