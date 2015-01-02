function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

Ureduce = U(:,1:K);

for i = 1:size(X,1)
	x = X(i, :)';
	projection_k = x'*Ureduce;
	Z(i, :) = projection_k;
endfor


end
