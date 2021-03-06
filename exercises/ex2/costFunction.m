function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

h = @(x) sigmoid(theta'*x');

res = 0;
for i = 1:m
	res = res + -y(i)*log(h(X(i,:))) - (1 - y(i))*log(1-h(X(i,:)));
end
J = (1/m)*res;

for j = 1:size(grad)
    res = 0;
        for i = 1:m
            xTransposed = X(i, :)';
            res = res + (h(X(i,:)) - y(i))*xTransposed(j);
        end
    grad(j) = (1/m) * res;
end

end
