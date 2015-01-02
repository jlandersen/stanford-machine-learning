function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
num_theta = length(theta);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    thetaTransposed = theta';

    for j = 1:num_theta
        res = 0;
            for i = 1:m
                xTransposed = X(i, :)';
                h = thetaTransposed*xTransposed;
                res = res + (h - y(i))*xTransposed(j);
            end
        theta(j) = theta(j) - alpha * (1/m) * res;
    end

 
    J_history(iter) = computeCost(X, y, theta);
end

end
