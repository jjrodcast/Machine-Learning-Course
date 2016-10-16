function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

% This is a first method we can use to calculate featureNormalize
% just making operation one by one on matrix

% First storing mean in mu variable
mu = mean(X);

% Second storing standard deviation in sigma variable
sigma = std(X);

%Third storing calculus of feature normalization
X_norm = (X .- mu) ./ sigma;



% Alternative method that could be used is the Next
% using for loop for each operation

%values = [1, size(X, 2)];
%for i = values
%    i_norm = (X(:,i) - mu(i)) / sigma(i);
%    X_norm(:,i) = i_norm; 
%end
% ============================================================

end
