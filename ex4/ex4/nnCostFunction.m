function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
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
%
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



yy=zeros(m,num_labels);
% %注释：1 label化y写法1，双重循环
% for k=1:num_labels
%     for i=1:m
%     if y(i)==k
%         yy(i,k)=1;
%     end
%     end
% end


%注释：2 label化y写法1，逻辑值付给所在列，骚气的写法
for k=1:num_labels
    yy(:,k)=(y==k);
end

% Part 1:计算J
a1=[ones(size(X,1),1),X];
a2=[ones(size(a1,1),1),sigmoid(a1*Theta1')];
%注释：最常犯错误，先激发之后再加bias；是【ones，sigmoid】不是sigmoid（【ones，。。】）
h=sigmoid(a2*Theta2');
for k=1:num_labels
j=1/m.*((-yy(:,k)'*log(h(:,k)))-(ones(m,1)-yy(:,k))'*log(ones(m,1)-h(:,k)));
J=J+j;
end


%Part 3:计算regularization
%注释：3 yy和output_L2都是5000*10的矩阵，处理是对应列进行J的累加运算

aa=0;
for k=2:(input_layer_size+1)
  a=Theta1(:,k)'*Theta1(:,k);
  aa=aa+a;
end

bb=0;
for k=2:(hidden_layer_size+1)%注释：又容易出错！！！是+1不是原有个数
  b=Theta2(:,k)'*Theta2(:,k);
  bb=bb+b;
end

r = (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));
%注释：易出错，矩阵自身运算用点乘配合sum更快速，不一定非要用叉乘
J=J+lambda/(2*m)*(aa+bb);
%J=J+r;



%Part 3:计算backpropagation

a1=[ones(size(X,1),1),X];%5000*401
z2=a1*Theta1';%5000*25
a2=[ones(size(a1,1),1),sigmoid(z2)];%5000*26
z3=a2*Theta2';%5000*10
a3=sigmoid(z3);





delta3=a3-yy;%5000*10
delta2=delta3*Theta2;%5000*26   %公式是g不是a
delta2=delta2(:,2:end).*sigmoidGradient(z2);%5000*25




%注释：至今仍是难点：在于对于每一样点都会生成一个梯度矩阵，并将梯度矩阵相加
for i=1:m
Theta1_grad=(Theta1_grad+delta2(i,:)'*a1(i,:)); %25*401
Theta2_grad=(Theta2_grad+delta3(i,:)'*a2(i,:));%10*26
end
Theta1_grad=Theta1_grad/m;
Theta2_grad=Theta2_grad/m;
%注释：错误在于m应该是加和后在除以m，而不是在循环里除m





m1 = size(Theta1,1);
m2 = size(Theta2,1);
Thet1 = [zeros(m1,1) Theta1(:,2:end)];
Thet2 = [zeros(m2,1) Theta2(:,2:end)];
 
Theta1_grad = Theta1_grad +lambda/m*Thet1; %25x401
Theta2_grad = Theta2_grad +lambda/m*Thet2; %10x26  


%注释：不加bias所以要把第一列去掉，相当于在每一层把累加项再加上一个lambda/m*Theta

% Delta1 = zeros(size(Theta1)); %25x401
% Delta2 = zeros(size(Theta2)); %10x26
% for i=1:m;
%     a1 = X(i,:); %a1 is 1x400
%     a1 =[1 a1]; %1x401
%     z2 = a1*Theta1';%1x25
%     a2 = sigmoid(z2);
%     m1 = size(a2,1);
%     a2 = [ones(m1,1) a2]; %1x26
%  
%     z3 = a2*Theta2'; %1x10
%     a3 = sigmoid(z3); %1x10
%     
%     delta_3 = a3 - yy(i,:); %1x10
%     temp = delta_3*Theta2; %1x26
%     delta_2 = temp(2:end).*sigmoidGradient(z2); %1x25
%     
%     Delta1 = Delta1 + delta_2'*a1;%25*401
%     Delta2 = Delta2 + delta_3'*a2;%10*26
% end
%  


% 


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
