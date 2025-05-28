%% PROXIMAL POINT, April 23 2024
% Solve min_x f(x), specifically for f(x) = ||x||_1

rng(1);
n   = 3;
x0  = 20*randn(n,1);
xStar = zeros(n,1); % this minimizes ||x||_1
style = 4;
switch style
    case 1
        V = 3; str='Scaled identity (and pos def)';
    case 2
        V = diag(rand(n,1)); str = 'Diagonal (and pos def)';
    case 3
        % Our theory does NOT cover this case!
        V = randn(n); V = .1*V*V' + .1*eye(n);
        V = tril(V); str = 'Lower triangular (and pos def)';
    case 4
        V = randn(n); V = .1*V*V' + .1*eye(n);
        V = tril(V); 
        V = inv(tril(max(.01,inv(V)))); % make inv(V) >= 0
        str = 'Lower triangular, and inverse non-negative (and pos def)';
end

e = eig(V); 
if isreal(e), disp('Eigenvalues are real'); else, disp('Eigenvalues are NOT REAL!!'); end
fprintf('min eig(V) is %g\n', min(e) )

prox_l1    = @(y,t) sign(y).*max(0,abs(y)-t); % prox of t*||x||_1. Note the convention
maxIts  = 300;
printEvery = 1;
errFcn  = @(x) norm(x-xStar);
errHist = zeros(maxIts,1);
tol     = 1e-10;
x       = x0;
for k = 1:maxIts
    %{
    We want to do x = (I + inv(V) df )^{-1} x_old
    and we do this via
        x = (V + df )^{-1} V x_old
    %}
    x = triangularProx(V*x,prox_l1,V);
    errHist(k) = errFcn(x);
    if ~mod(k,printEvery)
        fprintf('Iter %4d, error is %.2e\n', k, errHist(k) )
    end
    if errHist(k)<tol
        errHist=errHist(1:k);
        disp('Reached tolerance, quitting.')
        break;
    end
end
figure(1); clf;
semilogy(errHist,'linewidth',2); ylabel('Error'); xlabel('Iterations');
set(gca,'fontsize',16)
title(str)




%% FORWARD BACKWARD
% Solve min_x f(x) + g(x) with f smooth, g prox
%   Let's do f(x) = .5||Ax-b||^2 and g(x) = ||x||_1
%{
We want ||inv(V) H || < 1 or 2, where H = \nabla^2 f is the Hessian
%}

rng(1);
n   = 10;
m   = n; % for now
A   = randn(m,n); SPECIAL_STRUCTURE = false;
lambda = 2; % adjust this so xStar isn't all zeros or anything!
if true
    invA = tril(rand(n));
%     invA = invA - diag(diag(invA)) + eye(n); lambda = .5; % make it have 1 on diag. Works well
    A    = inv(invA');  % so... invA*H = A
    SPECIAL_STRUCTURE = true;
end
b   = randn(m,1);
f     = @(x) norm(A*x-b)^2/2;
gradf = @(x) A'*(A*x-b);
H     = A'*A; % Hessian
L     = norm(A)^2; % == max(eig(H))
cvx_begin
    variable xStar(n)
    cvx_quiet true
    cvx_precision best
    minimize sum_square( A*xStar - b )/2 + lambda*norm(xStar,1)
cvx_end

% x0  = 20*randn(n,1);
% 
fStar   = sum_square( A*xStar - b )/2 + lambda*norm(xStar,1);
errFcn  = @(x) norm(x-xStar)/norm(xStar);
objFcn  = @(x) norm(A*x-b)^2/2 + lambda*norm(x,1);

%%

style = 5;
switch style
    case 1
        V = L; str='Scaled identity (and pos def)';
    case 2
        V = L*eye(n) + diag(rand(n,1)); str = 'Diagonal (and pos def)';
        % or...
%         V = diag(diag(H)); V = V*norm(V\H)/.95; % Makes it worse
    case 3
        % Our theory does NOT cover this case!
        V = randn(n); V = .1*V*V' + L*eye(n);
        V = tril(V); str = 'Lower triangular (and pos def)';
    case 4
        V = randn(n); V = .1*V*V' + L*eye(n);
        V = tril(V); 
        V = inv(tril(max(.01,inv(V)))); % make inv(V) >= 0
        str = 'Lower triangular, and inverse non-negative (and pos def)';

    case 5
        % Trying to find a USEFUL preconditioner

        % do some QR stuff
        [Q,R] = qr(H); % Q*R = H; let's make R positive
        D     = diag(sign(diag(R)));
        % Q*D*D*R - H
        R = D*R; % and Q <-- Q*D is still orthogonal
%         cond(R'\H) % this is 1 !!
        V = R'; % perturbing by identity ruins things...
        
%         V = diag(diag(V));
        V = inv(tril(max(0,inv(V)))); % this makes things worse...

%         V = V + .001*eye(n);
        V = V*norm(V\H)/.95;

        str = 'QR stuff...';


        if SPECIAL_STRUCTURE
            % We're cheating here... we created "A" to have a nice
            % structure that we know, so now we're exploting that.
            %   (i.e., if we couldn't get an advantage in this case, it'd
            %   be unreasonable to expect an advantage in the fair case).

            % This DOES help!! but it stalls (due to numerical issues?)
            % But cond(V) itself is reasonable, so that doesn't make sense.
            V = inv(invA); % so inv(V) is all non-negative
%             V = V*norm(V\H)/.99;
            V = V*norm(V\H)/.02;
            % AHHH. What we really need is for V\H to be firmly
            % non-expansive, not just non-expansive
            % i.e., we want 2*V\H - I to be non-expansive

            str = 'Using special structure';
        end
end

e = eig(V); 
if isreal(e), disp('Eigenvalues are real'); else, disp('Eigenvalues are NOT REAL!!'); end
fprintf('min eig(V) is %g\n', min(e) )

contractionFactor = norm( V\H );
condVH            = cond(V\H);
fprintf('Contraction factor is %g (should be < 1 ), condition # is %.1e (%.1e for plain Hessian)\n', contractionFactor,condVH, cond(H) )
contractionFactor2 = norm( 2*V\H  - eye(n) );
if contractionFactor2 > 1, fid = 2; ans='NO!'; else, fid = 1; ans='Yes.'; end
fprintf(fid,'... but are we firmly non-expansive? %s Want 2V\\H-I to be nonexpansive; its norm is: %g\n', ans, contractionFactor2);


prox_l1    = @(y,t) sign(y).*max(0,abs(y)-t); % prox of t*||x||_1. Note the convention
prox_g     = @(y,t) prox_l1(y,lambda*t);
maxIts  = 1.5e5;
printEvery = round(maxIts/20);
errHist = zeros(maxIts,1);
objHist = zeros(maxIts,1);
tol     = 1e-10;
x       = x0;
for k = 1:maxIts
    %{
    We want to do x = (I + inv(V) dg )^{-1} y
        where y = (I - inv(V) nabla f ) x_old
    and we do this via
        x = (V + df )^{-1} V x_old
    %}

%     y = x - V\gradf(x);
%     x = triangularProx(V*y,prox_g,V);

    % Let's combine the above two steps to be more stable...
    x = triangularProx(V*x - gradf(x),prox_g,V);

    errHist(k) = errFcn(x);
    objHist(k) = objFcn(x);
    if ~mod(k,printEvery)
        fprintf('Iter %4d, error is %.2e\n', k, errHist(k) )
    end
    if errHist(k)<tol
        errHist=errHist(1:k);
        objHist=objHist(1:k);
        disp('Reached tolerance, quitting.')
        break;
    elseif errHist(k) > 1e10
        errHist=errHist(1:k);
        objHist=objHist(1:k);
        disp('Divergence, quitting.')
        break;
    end
end

sstr = sprintf('%s, cocoercivity %.2f', str, contractionFactor2 );

figure(1); %clf;
subplot(2,1,1);
semilogy(errHist,'linewidth',2,'DisplayName',sstr); 
hold all
ylabel('Error'); xlabel('Iterations');
set(gca,'fontsize',16)
% title(str)
legend()

subplot(2,1,2);
semilogy(objHist - fStar,'linewidth',2,'DisplayName',sstr); 
hold all
ylabel('Objective value error'); xlabel('Iterations');
set(gca,'fontsize',16)
% title(str)
legend()
title(sprintf("Lasso problem, %d x %d", m, n ))
%%
export_fig BasicTest_specialStructure_Apr25_2024 -pdf -transparent