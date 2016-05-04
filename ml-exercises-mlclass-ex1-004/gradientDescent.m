function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

theta2Value = 0;  
theta1Value = 0;
    
for i = 1:m
    theta2Value = theta2Value + X(i,2)*((theta(1) + theta(2)*X(i,2)) - y(i,1));
    theta1Value = theta1Value + ((theta(1) + theta(2)*X(i,2)) - y(i,1));
end
    
theta1Value = theta(1) - alpha*theta1Value/m;
theta2Value = theta(2) - theta2Value * alpha/m;
theta(1) = theta1Value;
theta(2) = theta2Value;

% theta = theta - alpha/m*X.'*(X*theta-y);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
