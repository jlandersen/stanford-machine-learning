function x = emailFeatures(word_indices)
%EMAILFEATURES takes in a word_indices vector and produces a feature vector
%from the word indices
%   x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
%   produces a feature vector from the word indices. 

n = 1899;

x = zeros(n, 1);

for i = 1:length(word_indices)
	index = word_indices(i);
	x(index) = 1;
endfor	

end
