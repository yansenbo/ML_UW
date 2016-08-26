function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% X = [ones(m,1) X];
% h_x = sigmoid(sigmoid(X*Theta1)*Theta2);
% [M,p] = max(h_x,[],2);  

X=[ones(m, 1) X]; %add one vector [1,1,..1]'
lx=size(X,1);
lt=size(Theta1,1);
tmp=zeros(lx,lt);
tmp=sigmoid(X*Theta1');
ot=ones(lx,1);
temp=[ot tmp];
tp=temp*Theta2';
otp=sigmoid(tp);
[M,p]=max(otp, [], 2); %M-max elements && p-index/position of elements


% =========================================================================


end
