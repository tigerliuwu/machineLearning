function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
stheta = length(theta);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

S = sigmoid(X * theta); % (m,1)
T = y .* log(S) + ( 1 - y) .* log(1 - S);
J = sum(T) * (-1/m);
theta2 = theta(2:stheta,1);
J = J + theta2' * theta2 * lambda/ (2 * m);


T = S - y;	% (m,n+1) * (n+1,1) = (m,1)
% X = (m,n+1)
grad = T' * X * 1/m;
grad = grad';

grad(2:stheta,1) = grad(2:stheta,1) .+ theta(2:stheta,1) * lambda/m;




% =============================================================

end
