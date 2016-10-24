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

% We add the one value to the X array (bias unit)
X = [ones(m, 1) X];

% The second layer for the Neural Network
z2 =  X * Theta1';
lenZ2 = size(z2, 1); % we get the size of the z2 just to add ones (bias unit) to the third layer
a2 = [ones(lenZ2, 1) sigmoid(z2)];

% The third layer for the Neural Network
z3 = a2 * Theta2';

%The output values 
a3 = sigmoid(z3); % This value represents Ho

%Get the maximum value for each row for the predictions

[value index] = max(a3, [], 2);

p = index;
% =========================================================================


end
