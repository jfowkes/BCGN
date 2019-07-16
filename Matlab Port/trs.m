% Trust Region Subproblem
function s = trs(J, gradf, delta)
    [~,n] = size(J);

    % Trust Region subproblem parameters
    LEPS = 1e-8;
    KE = 0.01;

    % J'J full rank
    if rank(J) == n

        % Set lambda (for newton iteration)
        lambda = 0;

        % Solve normal equations to find search direction
        [~,R] = qr(J,0); % economic QR
        t = R'\-gradf;
        s = R\t;
        ns = norm(s);

        % Trust region inactive: interior solution
        if ns < delta
            return;
        end
        % Else trust region active

    % J'J singular: lamda_1 = 0
    else

        % Set lambda for newton iteration
        lambda = LEPS;

        % Solve *perturbed* normal equations to find search direction
        [~,R] = qr([J; sqrt(lambda)*eye(n)],0); % economic QR
        t = R'\-gradf;
        s = R\t;
        ns = norm(s);

        % Hard case: find eigenvector of zero eigenvalue
        if ns < delta
            u = R\zeros(n,1); % since Q.T*zeros(m+p)=zeros(p)
            alpha = roots([u'*u, 2*s'*u, s'*s-delta^2]); % Find quadratic roots
            if isempty(alpha)
                alpha = zeros(2,1); % failed step: delta too large
            end
            s = s + alpha(1)*u; % FIXME: choosing alpha at random?
            return; 
        end
        % Else trust region active

    end
        
    % Trust region active: newton iteration
    while abs(ns - delta) > KE * delta

        % Solve R'w = s and calculate new lamda
        w = R'\s;
        nw = norm(w);
        lambda = lambda + (ns - delta)/delta * (ns/nw).^2;

        % Solve *perturbed* normal equations to find search direction
        [~,R] = qr([J; sqrt(lambda)*eye(n)],0); % economic QR
        t = R'\-gradf;
        s = R\t;
        ns = norm(s);
        
    end
end