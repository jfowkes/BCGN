% Gauss-Newton
function x = GN(res, jac, x0, fxopt, it_max, ftol, algorithm)

    % Full function and gradient
    f = @(x) 0.5*res(x)'*res(x);
    grad = @(x) jac(x)'*res(x);

    % Plotting
    plot_data = NaN(2,it_max+1);
    plot_data(1,1) = f(x0)-fxopt;
    plot_data(2,1) = norm(grad(x0));
    
    % Set initial trust region radius
    delta = norm(grad(x0))/10;
    
    k = 1;
    x = x0;
    while k < it_max && abs(f(x) - fxopt) > ftol

        % Assemble matrices
        J = jac(x);
        r = res(x);
        gradf = J'*r;

        % Solve subproblem
        if strcmp(algorithm,'tr')
            s = trs(J, gradf, delta);
        elseif strcmp(algorithm,'tr_approx')
            s = trs_approx(J, gradf, delta);
        elseif strcmp(algorithm,'tr_approx_precon')
            s = trs_approx_precon(J, gradf, delta);    
        elseif strcmp(algorithm,'reg')
            [s, delta] = reg(J, gradf, delta);
        elseif strcmp(algorithm,'reg_approx')
            s = reg_approx(J, r, delta);
        else
            error('Incorrect algorithm name.');
        end
        
        % Decrease
        Js = J*s;
        Delta_m = -gradf'*s -0.5*(Js'*Js);

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