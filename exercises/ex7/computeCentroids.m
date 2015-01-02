function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


combined = [ X idx ];
length = size(combined)(2);
actuallength = length - 1;

for i = 1:K
	kassigned = combined(combined(:,length) == i, 1:actuallength);
	kassignedcount = size(kassigned)(1);
	if (kassignedcount == 0)
		continue
	endif
	centroids(i,:) = (1/kassignedcount).*sum(kassigned);
endfor


end

