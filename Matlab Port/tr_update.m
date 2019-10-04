% Trust Region Update
function [x, delta] = tr_update(f, x, s, Delta_m, delta)

    % Trust Region parameters
    ETA1 = 0.1;
    ETA2 = 0.75;
    GAMMA1 = 0.5;
    GAMMA2 = 2.;
    DELTA_MIN = 1e-15;
    DELTA_MAX = 1e3;

    % Evaluate sufficient decrease
    rho = (f(x) - f(x+s))/Delta_m;

    % Accept trial point
    if rho >= ETA1
        x = x + s;
    end

    % Update trust region radius
    if rho < ETA1
        delta = delta*GAMMA1;
        delta = max(delta,DELTA_MIN);
    elseif rho >= ETA2
        delta = delta*GAMMA2;
        delta = min(delta,DELTA_MAX);
    end    
      
end