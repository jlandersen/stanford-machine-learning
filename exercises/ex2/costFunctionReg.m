function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


h = @(x) sigmoid(theta'*x');

res = 0;
reg = 0;
for i = 1:m
	res = res + (y(i)*log(h(X(i,:))) + (1 - y(i))*log(1-h(X(i,:))));
end

for j = 2:size(theta)
	reg = reg + theta(j)^2;
end

J = (-((1/m)*res)) + ((lambda/(2*m))*reg);

for j = 1:size(grad)
    res = 0;
	reguralized = 0;
    for i = 1:m
        xTransposed = X(i, :)';
        res = res + (h(X(i,:)) - y(i))*xTransposed(j);
    end

    if (j > 1) 
  		regularized = theta(j);
    	grad(j) = ((1/m)*res) + ((lambda/m)*regularized);
    else
    	grad(j) = ((1/m)*res);
    endif
end


% =============================================================

end
