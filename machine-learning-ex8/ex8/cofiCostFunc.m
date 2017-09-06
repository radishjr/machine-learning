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

#X = X(1:num_movies, 1:num_features);
#Theta = Theta(1:num_users, 1:num_features);
#Y = Y(1:num_movies, 1:num_users);
#R = R(1:num_movies, 1:num_users);

%Silly way to get the job done
%for j = 1:num_users
%  for i = 1:num_movies
%    if R(i, j) == 1
%      grad = (X(i, :) * Theta(j, :)' - Y(i,j));
%      for k = 1:num_features
%        X_grad(i, k) += grad * Theta(j,k);
%      end
%      J += grad^2/2;
%    end
%    if j == 1
%      Theta_lambda = lambda * (sum(Theta(i) ^ 2)) / 2;
%      J += Theta_lambda;
%     end
%  end
%  X_lambda = lambda * (sum(X_grad(j) ^ 2)) / 2;
%  J += X_lambda;
%end
%
%Theta_grad = zeros(size(Theta));
%for j = 1:num_users
%  for i = 1:num_movies
%    grad = 0;
%    if R(i, j) == 1      
%      grad = X(i, :) * Theta(j, :)' - Y(i,j);
%    end
%    for k = 1:num_features  
%      Theta_grad(j, k) += grad * X(i,k);
%    end
%  end
%  for k = 1:num_features  
%    Theta_grad(j, k) += lambda * Theta(j,k);
%  end
%end
%
%
%for i = 1:num_movies
%  for j = 1:num_users
%    grad = 0;
%    if R(i, j) == 1      
%      grad = X(i, :) * Theta(j, :)' - Y(i,j);
%    end
%    for k = 1:num_features  
%      X_grad(i, k) += grad * Theta(j,k);
%    end
%  end
%  for k = 1:num_features  
%    X_grad(i, k) += lambda * X(i,k);
%  end
%end

J = sum(sum((((X * Theta') - Y).^2).* R)) / 2 + lambda / 2 * (sum(sum(Theta.^2)) + sum(sum(X.^2)));
Theta_grad = ((X * Theta' - Y) .* R)'*X + lambda * Theta;
X_grad = ((X * Theta' - Y) .* R)*Theta + lambda * X;


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
