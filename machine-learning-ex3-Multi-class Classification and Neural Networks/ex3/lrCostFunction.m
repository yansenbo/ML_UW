function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

z_theta = X*theta;
h_theta = sigmoid(z_theta);
Reg_J = lambda/2/m*(theta(2,:)'*theta(2,:));
J = 1/m*(-y'*log(h_theta)-(1-y)'*log(1-h_theta))+Reg_J;
%J = 1/m*(-y'*log(sigmoid(X*theta))-(1-y)'*log(1-sigmoid(X*theta))) + lambda/2/m*(theta(2,:)'*theta(2,0));

% grad = grad + 1/m*X'*(h_theta-y);
Reg_grad = lambda/m*theta;
Reg_grad(1) = 0;
% grad = grad + Reg_grad;
grad = 1/m*X'*(sigmoid(X*theta)-y)+Reg_grad;

% temp1=-(log(sigmoid(X*theta)))'*y;
% temp2=(log(1-sigmoid(X*theta)))'*(1-y);
% 
% tt=theta(2:size(theta)).^2;
% J=(temp1-temp2)/m+lambda*sum(tt)/2/m;
% 
% %
% jf=0;
% for i=1:m
%         tmp=sigmoid(X(i,:)*theta);
%     jf=jf+(-y(i)*log(tmp)-(1-y(i))*log(1-tmp));
% end
% jf=jf/m;
% tmp=0;
% for j=2:size(theta)
%     tmp=tmp+theta(j)*theta(j);
% end
% jf=jf+tmp*lambda/2/m;
% 
% %
% grad=X'*(sigmoid(X*theta)-y)/m;
% tmp=theta;
% tmp(1)=0;
% grad=grad+lambda*tmp/m;







% =============================================================

grad = grad(:);

end
