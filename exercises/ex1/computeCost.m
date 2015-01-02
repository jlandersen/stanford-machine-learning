function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y


m = length(y); % number of training examples

J = 0;

res = 0;
theta = theta';

for i = 1:m
	xTransposed = X(i, :)';
	h = theta*xTransposed;
	res = res + (h - y(i))^2;
end

J = (1/(2*m)) *res;

grad = 

% =========================================================================

end
