function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m,1) X]; % Add the bias column
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


costsum = 0;

for i = 1:m
	a1 = X(i, :)';
	z2 = Theta1 * a1;
	a2 = sigmoid(z2);
	a2 = [1 ; a2];

	z3 = Theta2*a2;
	a3 = sigmoid(z3);
	for k = 1:num_labels
		yvector = zeros(num_labels,1);
		yvector(y(i)) = 1;
		costsum = costsum + (-yvector(k)*log(a3(k))-(1-yvector(k))*log(1-(a3(k))));
	endfor
endfor

J = (1/m)*costsum;

kmax = size(Theta1, 2);
jmax = size(Theta1, 1);
theta1regularizedsum = 0;
for j = 1:jmax
	for k = 2:kmax
		theta1regularizedsum = theta1regularizedsum + Theta1(j,k)^2;
	endfor
endfor

kmax = size(Theta2, 2);
jmax = size(Theta2, 1);
theta2regularizedsum = 0;
for j = 1:jmax
	for k = 2:kmax
		theta2regularizedsum = theta2regularizedsum + Theta2(j,k)^2;
	endfor
endfor

J = J + (lambda/(2*m))*(theta1regularizedsum + theta2regularizedsum);

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%


accumdelta = zeros(size(Theta1));
accumdelta2 = zeros(size(Theta2));
y_matrix = eye(num_labels)(y,:);

for t = 1:m
	a1 = X(t, :)';

	z2 = Theta1 * a1;
	a2 = sigmoid(z2);
	a2 = [1 ; a2];

	z3 = Theta2*a2;
	a3 = sigmoid(z3);

	delta3 = (a3 - y_matrix(t,:)');
	theta2_unbiased = Theta2(:,2:end);

	delta2 = (theta2_unbiased'*delta3) .* sigmoidGradient(z2);

	accumdelta2 = accumdelta2 + delta3*a2';
	accumdelta = accumdelta + delta2*a1';
endfor

Theta1_grad = (1/m) .* accumdelta;
Theta2_grad = (1/m) .* accumdelta2;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Reg_Theta1 = (lambda/m)*Theta1;
Reg_Theta1(:,1) = 0;
Theta1_grad = Theta1_grad + Reg_Theta1;


Reg_Theta2 = (lambda/m)*Theta2;
Reg_Theta2(:,1) = 0;

Theta2_grad = Theta2_grad + Reg_Theta2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
