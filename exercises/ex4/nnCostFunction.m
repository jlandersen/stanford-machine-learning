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

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
X = [ones(m,1) X]; % Add the bias column
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

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



Reg_Theta1 = (lambda/m)*Theta1;
Reg_Theta1(:,1) = 0;
Theta1_grad = Theta1_grad + Reg_Theta1;


Reg_Theta2 = (lambda/m)*Theta2;
Reg_Theta2(:,1) = 0;

Theta2_grad = Theta2_grad + Reg_Theta2;

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
