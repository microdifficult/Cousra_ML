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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

J=(1/m).*((-y)'*log(sigmoid(X*theta))-(ones(size(y))-y)'*log(1-sigmoid(X*theta)))+lambda./(2*m)*(theta(2:end)'*theta(2:end));


grad=(1/m).*((sigmoid(X*theta)-y)'*X)'+lambda/m.*theta;
grad(1)=(1/m).*((sigmoid(X*theta)-y)'*X(:,1))';   %先全算然后在用grad（1）覆盖

% =============================================================

end
%注释：1）正则项计算的是每个维度的参数theta（j）^2的加和，J最后算出来的是一个数
%      2）自己设计的算法思想是先利用公式把所有的梯度算出来，因为第一个参数不需要正则，因此再另算覆盖到第一个梯度上
%      3）对于梯度的公式是先乘再加和，公式含义是所有的残差与每一个维度的X的叉乘
%      4）grad：梯度算法很简洁巧妙，先计算每个维度的梯度，在利用矩阵加法的特性，每一个都加上theta，达到满足方程的目的。
%      5)很多细节：比如1-y（i）用(ones(size(y))-y)，theta（j）^2=theta*theta',关键活用矩阵叉乘