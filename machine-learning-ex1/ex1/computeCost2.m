sfunction [J, gradient] = computeCost2(theta)
  data = load('ex1data1.txt');
  X = data(:, 1); y = data(:, 2);
  X
  y
  m = length(y); % number of training examples
  n = length(theta);

  J = 0;

  for i=1:m
    hx = theta(1)* X(i,1) + theta(2) * X(i,2) ; % equals X * theta
    
    J = J + (hx - y(i)) ^ 2;
  end
  J = (J / (2 * m));

  gradient = theta;

  for feature_i = 1:n
    gradient(feature_i) = sum((X * theta - y).* X(:,feature_i)) / m;
  end

  gradient
end