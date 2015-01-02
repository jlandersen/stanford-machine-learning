function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = zeros(size(z));

s = @(x) 1/(1+e^-x);

for i = 1:size(z,1)
	for j = 1:size(z,2)
		g(i,j) = s(z(i,j));
	end
end

end
