function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

% oneVector = ones(1,m);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

  theta1Value = 0;
  theta2Value = 0;
  theta3Value = 0;
  
  for i = 1:m
      hypothesis = theta(1) + theta(2)*X(i,2) + theta(3)*X(i,3);
      theta3Value = theta3Value + X(i,3)*(hypothesis - y(i,1));
      theta2Value = theta2Value + X(i,2)*(hypothesis - y(i,1));
      theta1Value = theta1Value + (hypothesis - y(i,1));
  end
  
  theta1Value = theta(1) - alpha*theta1Value/m;
  theta2Value = theta(2) - theta2Value * alpha/m;
  theta3Value = theta(3) - theta3Value * alpha/m;
  theta(1) = theta1Value;
  theta(2) = theta2Value;
  theta(3) = theta3Value;

%       theta1Value = theta(1,1) - alpha/m * (  oneVector * ( (X*theta-y).*X(:,1) )  );
%       theta2Value = theta(2,1) - alpha/m * (  oneVector * ( (X*theta-y).*X(:,2) )  );
%       theta3Value = theta(3,1) - alpha/m * (  oneVector * ( (X*theta-y).*X(:,3) )  );
%   
%       theta(1) = theta1Value;
%       
%       theta(2) = theta2Value;
%       
%       theta(3) = theta3Value;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
