function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

grad = zeros(size(theta));

ma = X * theta - y;
J = ma' * ma / (2*m);

J = J + (theta(2:end)' * theta(2:end)) * lambda / (2 * m);

grad = X' * ma / m;	%n+1,m * m,1 = n+1,1%
grad(2:end,:) = grad(2:end,:) + theta(2:end,:) * lambda / m;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
