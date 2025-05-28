function [x,sg] = triangularProx(y,prox,V,returnCertificate)
% y = triangularProx(x,prox,V)
%   solves
%   x = ( V + df )^{-1}y, i.e., 
%   y = ( V + df )(x)
%   for a lower-triangular V, using forward substitution-like ideas
%
% V should be a positive definite lower triangular matrix
% y should be a vector
% prox should be a function prox(y,t) which returns
%   the 1D (or separable) solution to ( I + t*df )^{-1} y
%   e.g., if f(x)=|x| then
%   prox = @(y,t) sign(y).*max(0, abs(y) - t )

if nargin < 4 || isempty(returnCertificate), returnCertificate = false; end

x = zeros(size(y));
[n,p] = size(y);
if p>1, error('y should be a column vector'); end
if n > 1 && numel(V)==1
    V = V*eye(n);
end

if any(diag(V)<0)
    error('Found negative entries on diagonal, this may cause issues');
end

if returnCertificate
    % code is a bit slower
    % The idea is to return sg = df(x) that we can use as a certificate
    % ("sg" stands for "subgradient)
    % So if x = prox_{tg}(y) then y-x \in t*subdiff(x)
    sg = zeros(size(x));
    
    x(1)  = prox( y(1), 1 );
    sg(1) = y(1) - x(1);
    x(1)  = x(1)/V(1,1);

    for i = 2:n
        yy = y(i) - V(i,1:(i-1))*x(1:(i-1));
        x(i)  = prox( yy , 1 ); 
        sg(i) = yy - x(i);
        x(i)  = x(i)/V(i,i);
    end

else

    % for the (1,1) block, we solve x(1) = ( v11 + df )^{-1} y
    %   which is equivalent to      x(1) = ( 1 + 1/v11 df )^{-1} y/v11

    % from stephen
    % x(1) = prox( y(1), V(1,1) ); % OLD convention, where prox(y,t) returned (I + 1/t df)^{-1} y
    % x(1) = prox( y(1)/V(1,1), 1/V(1,1) ); % NEW convention, new "t" is old "1/t"
    % x(1) = prox( y(1), 1 ) / V(1,1); % same thing, maybe more stable?
    
    % i think we want this
    x(1) = prox( y(1) / V(1,1), 1 / V(1,1) ); % same thing, maybe more stable?

    for i = 2:n
        % from stephen
        %     x(i) = prox( y(i) - V(i,1:(i-1))*x(1:(i-1)), V(i,i) ); % OLD convention
        %     x(i) = prox( ( y(i) - V(i,1:(i-1))*x(1:(i-1)) )/V(i,i), 1/V(i,i) ); % NEW
        % x(i) = prox( y(i) - V(i,1:(i-1))*x(1:(i-1)) , 1 )/V(i,i); % NEW (but more stable??)

        % from alex
        % i think we want this
        x(i) = prox( (y(i) - V(i,1:(i-1))*x(1:(i-1)))/ V(i, i) , 1 / V(i, i) ); % NEW (but more stable??)
    end
end