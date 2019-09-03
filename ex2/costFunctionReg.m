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
grad(1)=(1/m).*((sigmoid(X*theta)-y)'*X(:,1))';   %��ȫ��Ȼ������grad��1������

% =============================================================

end
%ע�ͣ�1��������������ÿ��ά�ȵĲ���theta��j��^2�ļӺͣ�J������������һ����
%      2���Լ���Ƶ��㷨˼���������ù�ʽ�����е��ݶ����������Ϊ��һ����������Ҫ������������㸲�ǵ���һ���ݶ���
%      3�������ݶȵĹ�ʽ���ȳ��ټӺͣ���ʽ���������еĲв���ÿһ��ά�ȵ�X�Ĳ��
%      4��grad���ݶ��㷨�ܼ������ȼ���ÿ��ά�ȵ��ݶȣ������þ���ӷ������ԣ�ÿһ��������theta���ﵽ���㷽�̵�Ŀ�ġ�
%      5)�ܶ�ϸ�ڣ�����1-y��i����(ones(size(y))-y)��theta��j��^2=theta*theta',�ؼ����þ�����