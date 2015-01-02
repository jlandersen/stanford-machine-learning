function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

C = 1;
sigma = 0.3;

Ctest = [ 0.01 ; 0.03; 0.1 ; 0.3; 1; 3; ];
sigmatest = [ 0.01 ; 0.03; 0.1 ; 0.3; 1; 3; ];
cn = size(Ctest)(1)
sigman = size(sigmatest)(1)

bestfit = [ -1, -1 ]
highesterr = inf

run = 0

for i = 1:cn
	for j = 1:sigman
		Cval = Ctest(i)
		sigmaval = sigmatest(j)

		model = svmTrain(X, y, Cval, @(x1, x2) gaussianKernel(x1, x2, sigmaval)); 
		predictions = svmPredict(model, Xval);
		err = mean(double(predictions ~= yval));
		if err < highesterr
			bestfit = [ i, j ];
			highesterr = err;
		endif
		run = run + 1
	endfor
endfor

C = Ctest(bestfit(1));
sigma = sigmatest(bestfit(2));

fprintf('Best fit\n');
C
sigma

end
