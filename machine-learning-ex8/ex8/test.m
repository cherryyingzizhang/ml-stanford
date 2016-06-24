%Test File

% Load dataset which has X.
X = [1, 2;
     1, 2;
     1, 2];

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);
mu
sigma2
% Compute the mean of the data and the variances
% In particular, mu(i) should contain the mean of
% the data for the i-th feature and sigma2(i)
% should contain variance of the i-th feature.
for i = 1:n
    mu(i,1) = sum(X(:,i))/m;
    mu(i,1)
    sigma2(i,1) = sum((X(:,i)-mu(i,1)).^2)/m;
    sigma2(i,1)
end