function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

allhtheta = sigmoid(X*theta);

J = (1/m)*sum(-y.*log(allhtheta) - (1-y).*log(1-(allhtheta))) + (lambda/(2*m))*sum(theta(2:end).^2);

grad = (1/m)*X'*(allhtheta - y);

temp = theta; 

temp = temp .* (lambda/m);
temp(1) = 0;   
grad = grad + temp;

grad = grad(:);

end
