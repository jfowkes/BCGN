% Regularization Subproblem
function [s, delta] = reg(J, gradf, delta)
    [~,n] = size(J);

    % Parameters
    SIGMA_MIN = 1e-8;

    % J'J singular: limit sigma to sigma_min
    if rank(J) ~= n
        delta = max(delta,SIGMA_MIN);
    end
        
    % Solve *perturbed* normal equations to find search direction
    [~,R] = qr([J; sqrt(delta)*eye(n)],0); % economic QR
    t = R'\-gradf;
    s = R\t;

end