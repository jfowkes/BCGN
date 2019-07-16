% Random Block-Coordinate Gauss-Newton
function x = BCGN(res, jac, x0, fxopt, it_max, ftol, p, algorithm)
    [n,~] = size(x0);

    % Full function and gradient
    f = @(x) 0.5*res(x)'*res(x);
    grad = @(x) jac(x)'*res(x);

    % Plotting
    plot_data = NaN(2,it_max+1);
    plot_data(1,1) = f(x0)-fxopt;
    plot_data(2,1) = norm(grad(x0));
    
    k = 1;
    x = x0;
    while k < it_max && abs(f(x) - fxopt) > ftol

        % Randomly select column indices
        S = randperm(n,p);
        
        % Assemble reduced matrices
        J = jac(x);
        J_S = J(:,S);
        r = res(x);
        gradf_S = J_S'*r;

        % Set initial trust region radius
        if k == 1
            delta = norm(gradf_S)/10;
        end
            
        % Solve subproblem
        if strcmp(algorithm,'tr')
            s_S = trs(J_S, gradf_S, delta);
        elseif strcmp(algorithm,'tr_approx')
            s_S = trs_approx(J_S, gradf_S, delta);
        elseif strcmp(algorithm,'tr_approx_precon')
            s_S = trs_approx_precon(J_S, gradf_S, delta);    
        elseif strcmp(algorithm,'reg')
            [s_S, delta] = reg(J_S, gradf_S, delta);
        elseif strcmp(algorithm,'reg_approx')
            s_S = reg_approx(J_S, r, delta);
        else
            error('Incorrect algorithm name.');
        end
        
        % Decrease
        Js_S = J_S*s_S;
        Delta_m = -gradf_S'*s_S -0.5*(Js_S'*Js_S);

        % Project step to R^n
        s = zeros(n,1);
        s(S) = s_S;
        
        % Update parameter and take step
        if startsWith(algorithm,'tr')
            [x, delta] = tr_update(f, x, s, Delta_m, delta);
        elseif startsWith(algorithm,'reg')
            [x, delta] = reg_update(f, x, s, Delta_m, delta); % same as tr_update with grow/shrink swapped
        end
        k = k + 1;
        
        % Plotting
        plot_data(1,k) = f(x)-fxopt;
        plot_data(2,k) = norm(grad(x));

    end

    % Plotting
    figure()
    subplot(1,2,1);
    X = 1:it_max+1;
    semilogy(X,plot_data(1,:),'LineWidth',2)
    xlabel('Iterations')
    ylabel('Norm Residual')
    grid on;
    subplot(1,2,2);
    semilogy(X,plot_data(2,:),'LineWidth',2)
    xlabel('Iterations')
    ylabel('Norm Gradient')
    grid on;
   
end