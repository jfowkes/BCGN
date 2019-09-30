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

    % J'J singular: lambda_1 = 0
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
            [u,~] = eigs(R'*R,1,'smallestabs'); % since R'R = J'J + lambda*I
            alpha = roots([u'*u, 2*s'*u, s'*s-delta^2]); % Find quadratic roots
            s = modelmin(s+alpha(1)*u, s+alpha(2)*u); % Find step that makes trs model smallest
            return; 
        end
        % Else trust region active

    end
        
    % Trust region active: newton iteration
    while abs(ns - delta) > KE * delta

        % Solve R'w = s and calculate new lambda
        w = R'\s;
        nw = norm(w);
        lambda = lambda + (ns - delta)/delta * (ns/nw).^2;

        % Solve *perturbed* normal equations to find search direction
        [~,R] = qr([J; sqrt(lambda)*eye(n)],0); % economic QR
        t = R'\-gradf;
        s = R\t;
        ns = norm(s);
        
    end
    
    % Hard case: find step that makes trs model smallest
    function sh = modelmin(s1, s2)
        Js1 = J*s1;
        Js2 = J*s2;
        qs1 = gradf'*s1 + 0.5*(Js1'*Js1);
        qs2 = gradf'*s2 + 0.5*(Js2'*Js2);
        if qs1 < qs2
            sh = s1;
        else
            sh = s2;
        end
        return;
    end
        
end