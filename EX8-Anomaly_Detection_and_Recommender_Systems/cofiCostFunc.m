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

%num_users = 4; num_movies = 5; num_features = 3;
%X = X(1:num_movies, 1:num_features);
%Theta = Theta(1:num_users, 1:num_features);
%Y = Y(1:num_movies, 1:num_users);
%R = R(1:num_movies, 1:num_users);


%X = [5 x 3], Theta = [4 x 3], Y = [5 x 4], R = [5 x 4];

% We multiply by R, because we just need the cost of all 1's values
sqrt_error = ((X * Theta' - Y) .* R);
sqrt_root_error = sqrt_error .^2;
grad_error = sqrt_error * Theta;

J = (1/2) * sum(sum(sqrt_root_error)); % [value]
X_grad = grad_error; % [5 x 3]
Theta_grad = sqrt_error' * X; % [4 x 3]

% Adding regularization

J = J + (lambda/2) * sum(sum((Theta .^2))) + (lambda/2) * sum(sum(X .^2));
X_grad = X_grad + lambda * X;
Theta_grad = Theta_grad + lambda * Theta;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
