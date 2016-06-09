function [error_train, error_val] = learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the training and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
% ---------------------- Solution ----------------------

for i = 1:m
    theta = trainLinearReg(X(1:i, :), y(1:i, :), lambda);
    
    %0 for lambda because when plotting learning curve, error_train
    %and error_val are not defined with the regularization term
    [J_train, ~] = linearRegCostFunction(X(1:i, :), y(1:i, :), theta, 0);
    error_train(i) = J_train;
    [J_val, ~] = linearRegCostFunction(Xval, yval, theta, 0);
    error_val(i) = J_val;
end

%Reason why we do not have regularization term when doing learning
%curve/assessing error after getting the hypothesis theta values:

%Regularization is used to artificially increase the error of the model 
%in order to get a certain type of desired training parameters (theta). 
%For linear regression, regularization keeps the model from getting too complex
%(overfitting) by artificially creating extra error for every non-zero parameter 
%(theta). The higher the theta, the higher the artificial error.
%But, this is only necessary when you are training your thetas / parameters.
%When it comes time to assess how well your parameters are doing, there is
%no reason to artificially add error to the result. The artificial error term
%is the lambda term.
%In short, lambda is used to produce certain types of thetas. Once the thetas
%are created, and you want to say how the model does in the real world, using
%lambda adds unnecessary error.



% -------------------------------------------------------------

% =========================================================================

end
