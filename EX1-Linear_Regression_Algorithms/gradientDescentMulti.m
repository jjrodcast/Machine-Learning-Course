function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

% We need to storage this auxiliar variable to iterate through all the features
% here we storage the length of thetha values
len = length(theta);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    % We use temporary matrix with the information of X
    X_temp = X;
    
    % Iterate through all features and calculate the hypothesis for each one
    for i = 1:len 
         X_temp(:,i) = X(:,i) .* theta(i); 
    end
    
    % Once we have the hypothesis, we sum the hypothesis for each training register
    V_hypothesis = sum(X_temp, 2);

    % Finally, once again we iterate over the thetha length to calculate the new values for each one
    % here we apply gradient descent step for each thetha
    for j = 1:len
        theta(j)  = theta(j) - alpha * (1/m) * sum( (V_hypothesis - y) .* X(:,j) );
    end

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
