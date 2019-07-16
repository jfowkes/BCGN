% Adaptive Random Block-Coordinate Gauss-Newton
function x = ABCGN(res, jac, x0, fxopt, it_max, ftol, p, algorithm)
    [n,~] = size(x0);

    % Adaptive parameters
    KAPPA = 0.7; % adaptive stopping rule
    STEP = 2; % adaptive step size
    
    % Full function and gradient
    f = @(x) 0.5*res(x)'*res(x);
    grad = @(x) jac(x)'*res(x);

    % Plotting
    plot_data = NaN(3,it_max+1);
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
        
        % Loop tolerance
        Js_S = J_S*s_S;
        Delta_m = -gradf_S'*s_S -0.5*(Js_S'*Js_S);
        stopping_rule = -Delta_m + (1-KAPPA)/2*power(norm(r),2) > 0;
        
        % Iteratively refine block size
        [~,p_in] = size(S);
        while KAPPA ~= 1 && p_in ~= n && stopping_rule
            
            % Increase block size
            step = min(STEP,n-p_in);
            rem_inds = setdiff(1:n,S); 
            SA = randsample(rem_inds,step);
            S = [S,SA];
            
            % Assemble reduced matrices
            J_S = J(:,S);
            gradf_S = J_S'*r;
            
            % Set initial trust region radius
            if k == 1
                delta = norm(gradf_S)/10;
            end

            % Update block size
            p_in = p_in + step;
            
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
            
            % Loop tolerance
            Js_S = J_S*s_S;
            Delta_m = -gradf_S'*s_S -0.5*(Js_S'*Js_S);
            stopping_rule = -Delta_m + (1-KAPPA)/2*power(norm(r),2) > 0;
            
        end

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
        plot_data(3,k) = p_in;

    end

    % Plotting
    figure()
    subplot(2,2,1);
    X = 1:it_max+1;
    semilogy(X,plot_data(1,:),'LineWidth',2)
    xlabel('Iterations')
    ylabel('Norm Residual')
    grid on;
    subplot(2,2,2);
    semilogy(X,plot_data(2,:),'LineWidth',2)
    xlabel('Iterations')
    ylabel('Norm Gradient')
    grid on;
    subplot(2,2,[3,4]);
    semilogy(X,plot_data(3,:),'LineWidth',2)
    xlabel('Iterations')
    ylabel('Block Size')
    grid on;
   
end