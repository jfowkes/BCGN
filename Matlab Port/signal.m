function signal()

% Functions
SIGMA = 35;
psi = @(s,t) exp(-SIGMA*(s-t).^2);
dpsidt = @(s,t) 2*SIGMA*(s-t)*psi(s,t);

% Data
%k = 10;
%m = 50;
k = 5;
m = 15;
rng(0); % set seed
alpha = 0 + (5-0).*rand(k,1) % unknown
t = 0.2 + (0.8-0.2).*rand(k,1) % unknown
s = linspace(0,1,m); % observed
y = zeros(m,1); % observed
for i=1:m
    y(i) = sum(alpha'*psi(s(i),t)); %+ 0.01*randn(1);
end

% Plot psi(s)
ss = linspace(0,1,1000);
yy = zeros(1,1000);
for i=1:1000
    yy(i) = sum(alpha'*psi(ss(i),t));
end
hold on;
plot(ss,yy,'-','LineWidth',2)
plot(s,y,'.','MarkerSize',15)
xlabel('s')
ylabel('y(s)')
legend('y(s)','y observed','Location','southeast')
grid on;
hold off;
 
% Residual r_i
function r = res(x)
    tx = x(1:k);
    alphax = x(k+1:end);
    r = zeros(m,1);
    for i=1:m
        r(i) = sum(alphax'*psi(s(i),tx)) - y(i);
    end
end
 
% Jacobian dr_i/d_xj
function J = jac(x)
    tx = x(1:k);
    alphax = x(k+1:end);
    J = zeros(m,2*k);
    for i=1:m
        for j=1:k % dr_i/d_tj
            J(i,j) = dpsidt(s(i),tx(j))*alphax(j);
        end
        for j=1:k % dr_i/d_alphaj
            J(i,k+j) = psi(s(i),tx(j));
        end
    end
end

% GN Inputs
x0 = 0.5*ones(2*k,1);
%x0 = np.random.rand(2*k)
fxopt = 0;
IT_MAX = 500;
FTOL = 1e-15;

% Run Gauss-Newton
xopt = ABCGN(@res,@jac,x0,fxopt,IT_MAX,FTOL,2,'tr');
disp('t*:')
disp(xopt(1:k))
disp('alpha*:')
disp(xopt(k+1:end))
 
% Plot t*
figure()
hold on;
plot(xopt(1:k),xopt(k+1:end),'r.','MarkerSize',15)
plot(t,alpha,'k.','MarkerSize',15)
xlabel('t')
ylabel('alpha')
legend('t* (optimization)','t (actual)','Location','northwest')
grid on;
hold off;

end
