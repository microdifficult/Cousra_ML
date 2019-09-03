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
% %ע�ͣ�1 label��yд��1��˫��ѭ��
% for k=1:num_labels
%     for i=1:m
%     if y(i)==k
%         yy(i,k)=1;
%     end
%     end
% end


%ע�ͣ�2 label��yд��1���߼�ֵ���������У�ɧ����д��
for k=1:num_labels
    yy(:,k)=(y==k);
end

% Part 1:����J
a1=[ones(size(X,1),1),X];
a2=[ones(size(a1,1),1),sigmoid(a1*Theta1')];
%ע�ͣ���������ȼ���֮���ټ�bias���ǡ�ones��sigmoid������sigmoid����ones����������
h=sigmoid(a2*Theta2');
for k=1:num_labels
j=1/m.*((-yy(:,k)'*log(h(:,k)))-(ones(m,1)-yy(:,k))'*log(ones(m,1)-h(:,k)));
J=J+j;
end


%Part 3:����regularization
%ע�ͣ�3 yy��output_L2����5000*10�ľ��󣬴����Ƕ�Ӧ�н���J���ۼ�����

aa=0;
for k=2:(input_layer_size+1)
  a=Theta1(:,k)'*Theta1(:,k);
  aa=aa+a;
end

bb=0;
for k=2:(hidden_layer_size+1)%ע�ͣ������׳���������+1����ԭ�и���
  b=Theta2(:,k)'*Theta2(:,k);
  bb=bb+b;
end

r = (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));
%ע�ͣ��׳����������������õ�����sum�����٣���һ����Ҫ�ò��
J=J+lambda/(2*m)*(aa+bb);
%J=J+r;



%Part 3:����backpropagation

a1=[ones(size(X,1),1),X];%5000*401
z2=a1*Theta1';%5000*25
a2=[ones(size(a1,1),1),sigmoid(z2)];%5000*26
z3=a2*Theta2';%5000*10
a3=sigmoid(z3);





delta3=a3-yy;%5000*10
delta2=delta3*Theta2;%5000*26   %��ʽ��g����a
delta2=delta2(:,2:end).*sigmoidGradient(z2);%5000*25




%ע�ͣ����������ѵ㣺���ڶ���ÿһ���㶼������һ���ݶȾ��󣬲����ݶȾ������
for i=1:m
Theta1_grad=(Theta1_grad+delta2(i,:)'*a1(i,:)); %25*401
Theta2_grad=(Theta2_grad+delta3(i,:)'*a2(i,:));%10*26
end
Theta1_grad=Theta1_grad/m;
Theta2_grad=Theta2_grad/m;
%ע�ͣ���������mӦ���ǼӺͺ��ڳ���m����������ѭ�����m





m1 = size(Theta1,1);
m2 = size(Theta2,1);
Thet1 = [zeros(m1,1) Theta1(:,2:end)];
Thet2 = [zeros(m2,1) Theta2(:,2:end)];
 
Theta1_grad = Theta1_grad +lambda/m*Thet1; %25x401
Theta2_grad = Theta2_grad +lambda/m*Thet2; %10x26  


%ע�ͣ�����bias����Ҫ�ѵ�һ��ȥ�����൱����ÿһ����ۼ����ټ���һ��lambda/m*Theta

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
