function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

for i = 1:size(R,1)
	for j = 1:size(R,2)
		if R(i,j) != 1
			continue
		endif
		J = J + (Theta(j,:)*X(i,:)' - Y(i,j)).^2;
	endfor
endfor


regUser = 0;
for j = 1:num_users
	for k = 1:num_features
		regUser = regUser + Theta(j,k)^2;
	endfor
endfor
regUser = (lambda/2)*regUser;

regMovie = 0;
for i = 1:num_movies
	for k = 1:num_features
		regMovie = regMovie + X(i,k)^2;
	endfor
endfor
regMovie = (lambda/2)*regMovie;

J = (1/2)*J + regUser + regMovie;

for i = 1:size(R,1)
	res = 0;
	for j = 1:size(R,2)
		if R(i,j) != 1
			continue
		endif

		res = res + ((Theta(j,:)*X(i,:)' - Y(i,j))*Theta(j,:));
	endfor

	X_grad(i,:) = res + lambda*X(i,:);
endfor


for j = 1:size(R,2)
	res = 0;
	for i = 1:size(R,1)
		if R(i,j) != 1
			continue
		endif

		res = res + ((Theta(j,:)*X(i,:)' - Y(i,j))*X(i,:));
	endfor

	Theta_grad(j,:) = res + lambda*Theta(j,:);
endfor




% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end