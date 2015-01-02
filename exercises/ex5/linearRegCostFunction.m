function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

allhtheta = X*theta;

J = (1/(2*m))*sum((allhtheta - y).^2) + (lambda/(2*m))*sum(theta(2:end).^2);

grad = (1/m)*X'*(allhtheta - y);

temp = theta; 

temp = temp .* (lambda/m);
temp(1) = 0;   
grad = grad + temp;

grad = grad(:);

end
