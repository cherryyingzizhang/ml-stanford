function J = computeCost(x, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(x, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

for i = 1:m
    J = J + ((theta(1) + theta(2)*x(i,2)) - y(i,1))^2;
end
J = J * 1/(2*m);

% =========================================================================

end
