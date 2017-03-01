function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
available_List = [0.01,0.03,0.1,0.3,1,3,10,30];
try_times = size(available_List,2);
error_M = ones(try_times);

for i=1:try_times
    for j=1:try_times
        C = available_List(i);
        sigma = available_List(j);
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
        prediction = svmPredict(model, Xval);
        
        error_M(i,j) = sum((prediction - yval).^2);
    end
end

[min_col, min_j] = min(error_M);

[min_val, min_i] = min(min_col);

sigma = available_List(min_i);
C = available_List(min_j(min_i));

% =========================================================================

end
