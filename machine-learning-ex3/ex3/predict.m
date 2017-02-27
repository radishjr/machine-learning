function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%







a1 = [ones(m, 1) X];

z2 = a1 * Theta1';
a2 = zeros(m, size(Theta1)(1));
[col1, row1] = size(a2);
for i = 1:col1
  for j = 1:row1
    a2(i, j) = 1/(1 + e^(-z2(i,j)));
  endfor
endfor
a2 = [ones(m, 1) a2];

z3 = a2 * Theta2';
a3 = zeros(m, size(Theta2)(1));
[col2, row2] = size(a3);
for i = 1:col2
  for j = 1:row2
    a3(i, j) = 1/(1 + e^(-z3(i,j)));
  endfor
endfor

[maxV indices] = max(a3, [], 2);

p = indices;


% =========================================================================


end
