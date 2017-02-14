function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n = length(theta)

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    newtheta = theta;
    newtheta(1) = theta(1) - alpha * sum((X * theta - y).*X(:,1)) / m;
    newtheta(2) = theta(2) - alpha * sum((X * theta - y).*X(:,2)) / m;
    
    %newtheta = theta;
    %for feature_i = 1:n
    %  newtheta(feature_i) = theta(feature_i) - alpha * sum((X * theta - y).* X(:,feature_i)) / m;
    %end

    %if(theta == newtheta)
    %  printf("finished!, %d, %d", iter, theta)
    %  break
    %end
    
    theta = newtheta;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
      

end

end
