function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

m = size(X, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

for i = 1:m
	x = X(i, :)';
	c = 0;
	closestdistance = Inf;

	for j = 1:K
		centroid = centroids(j,:)';
		distance = sqrt(sum((x - centroid).^2));
		if distance < closestdistance
			closestdistance = distance;
			c = j;
		endif
	endfor

	idx(i) = c;

endfor

end

