function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h = X * theta;
error = h - y;
theta_need_penalty = theta(2:end);
J = (sum(error .* error) + sum(lambda * (theta_need_penalty.*theta_need_penalty))) / (2 * m);

theta_need_penalty = [0;theta_need_penalty];
for j=1:n
    grad(j) = (sum((h-y).*X(:,j)) + lambda * theta_need_penalty(j))/m;    
end


% =========================================================================

grad = grad(:);

end
