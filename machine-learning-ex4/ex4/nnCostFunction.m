function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% PLEASE READ THIS NOTE: All of the for-loops in this file could be instead
% be combined into one for-loop. The reason why I am not optimizing this
% code is because the assignment tells you to implement each step
% sequentially, and I was just trying to follow those steps, regardless
% of whether or not they are optimal.

%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));           
%For our example, Theta1 is 25 x 401, Theta2 is 10 x 26

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Add the column of ones to the X matrix (bias unit for each experiment)
X_modified = [ones(m,1), X]; % the modified matrix is now mx401, m = # experiments

for i = 1:m
    X_i = X_modified(i,:); %ith row vector of the X matrix... 1x401
    
    % 25x1 vector = 25x401 * transpose(1x401)
    z = Theta1 * X_i';
    % use the sigmoid function to get the hidden layer activations
    hidden_layer_activations = sigmoid(z);
    % add the bias unit to the hidden layer (26x1):
    hidden_layer_activations = [1; hidden_layer_activations];
    
    % (10x1 vector) = 10x26 Theta2 * 26x1 vector
    z_output = Theta2 * hidden_layer_activations;
    % h(x) vector of outputs
    h_output = sigmoid(z_output);
    
    % To update the cost function for the ith experiment, we need
    % to re-code the y value for the ith experiment into size(h_output)
    % number of vectors e.g. if the ith experiment's true y value was 5,
    % then all vectors other than the fifth vector are zero vectors, and
    % the fifth vector is [0; 0; 0; 0; 1; ... 0]
    for j = 1:size(h_output,1)
        y_vector = zeros(size(h_output,1),1);
        y_vector(y(i),1) = 1;
        
        J = J + -y_vector(j,1) * log(h_output(j)) - (1- y_vector(j,1)) * log (1-h_output(j));
    end
end

%The formula for the cost function has a 1/m term at the very end of all
%the summations.
J = J / m; 

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

for t = 1:m
    % Part 2 - Step 1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %a1 (first layer activations)
    X_i = X_modified(t,:); %t-th row vector of the X matrix w/ bias unit (1x401)
    
    % 25x1 vector = 25x401 * transpose(1x401)
    z = Theta1 * X_i'; % actually z2 since z1 does not exist.
    % use the sigmoid function to get the hidden layer activations
    hidden_layer_activations = sigmoid(z); % a2
    
    % add the bias unit to the hidden layer (26x1):
    hidden_layer_activations = [1; hidden_layer_activations];
    % (10x1 vector) = 10x26 Theta2 * 26x1 vector
    z_output = Theta2 * hidden_layer_activations; %z3
    % h(x) vector of outputs
    h_output = sigmoid(z_output); %a3
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Part 2 - Step 2
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    y_vector = 1:size(h_output,1); %creates a row vector the same size as h_output
    y_vector = (y_vector == y(t,1))'; %transpose to make it a column vector
    
    delta_output = (h_output - y_vector);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Part 2 - Step 3
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Remember that z1 does not exist... after a1 is 
    % calculating z2.
    theta2Transpose = Theta2';
    delta_hidden_layer = theta2Transpose(2:end,:) * delta_output .* sigmoidGradient(z);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    % Part 2 - Step 4
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Theta1_grad = Theta1_grad + delta_hidden_layer * X_i;
    Theta2_grad = Theta2_grad + delta_output * hidden_layer_activations';
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

% Part 2 - Step 5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 3.1
% Calculating the regularization term(s) of the cost function and adding it to
% the cost function. Note that we do not consider the bias unit column in
% either theta matrices, which is why some forloops start at index 2
% instead of 1

J_regularization_term_theta1 = 0;
for i = 1:hidden_layer_size
    for j = 2:input_layer_size+1
        J_regularization_term_theta1 = J_regularization_term_theta1 + Theta1(i,j)^2;
    end
end

J_regularization_term_theta2 = 0;
for i = 1:num_labels % output layer size
    for j = 2:hidden_layer_size+1
        J_regularization_term_theta2 = J_regularization_term_theta2 + Theta2(i,j)^2;
    end
end

% Add these regularization terms into the total cost function
J = J + lambda/(2*m) * (J_regularization_term_theta1 + J_regularization_term_theta2);


% Part 3.2
% Adding regularization for the gradients
% Note that Theta1 and Theta1_grad is 25 x 401, 
% and Theta2 and Theta2_grad is 10 x 26
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda./m .* Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda./m .* Theta2(:, 2:end);






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
