function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
theta_0 = theta(1,1);
theta_1 = theta(2,1);

h = theta_0*X(:,1) + theta_1*X(:,2);

diff = h - y;

leastSqrSum = sum((diff).^2);

J = (leastSqrSum/(2*m));



% =========================================================================

end
