% This code implements a state space neural network with ordered variance
% (SSNNO) on a continulous stirred tank (CSTR) example.
% The training data is in input-output form
% Input: Jacket temperature
% Output: Reactor temperature

%% Defining training and testing dataset
clear all;close all
global Xtr Utr n m p l NT 
% Consider the CSTR plant data [time u x1 x2]
load data -ascii
u = data(:,2)';     % the jacket temperature is the input
x = data(:,[3 4])'; % reactor conc, reactor temp
y = data(:,4)';     % the reactor temparature is the output
randn('state',1)
rand('state',5)
NT=500;n=2;m=1; p=1; T = 0.1;  C=[0 1];                                

Utr=u(:,1:500); 
Xtr=x(:,1:500);                                           % xtrain contains the state sequence (x(1),...x(NT)) 
Ytr=y(:,1:500)+0.05*randn(1,500);                         % ytrain contains the output sequence (y(1),...y(NT)) 
Uts=u(:,501:900);
% Use the below Uts for step response
%Uts=-0.1*ones(1,400);
Xts=x(:,501:900);
Yts=y(:,501:900)+0.05*randn(1,400);

%% State space model identification using SSNNO
alpha=0.3;  beta=0.1;  gamma=0.1;
epsilon = 0.0001;
l=3;      % l denotes the number of states in the trained SSNNO model. 
A=rand(l,3*l+m+p+2); % A contain the weights, biases and the initial state: A=[Af1 Bf1 Af2 Ag1 Ag2 x0] 
Q=diag([0.01,0.02,0.05]);
for i=1:2
   % Training the SSNNO
    fun = @(A)alpha*trace((Ytr-A(:,3*l+m+2:3*l+m+p+1)'*tanh(A(:,2*l+m+2:3*l+m+1)*fNN(A,Utr)))'*(Ytr-A(:,3*l+m+2:3*l+m+p+1)'*tanh(A(:,2*l+m+2:3*l+m+1)*fNN(A,Utr))))+beta*trace((fNN(A,Utr)-mean(fNN(A,Utr)')')'*Q*(fNN(A,Utr)-mean(fNN(A,Utr)')'))+gamma*trace(A(:,2*l+m+2:3*l+m+p+1)'*A(:,2*l+m+2:3*l+m+p+1));
    options = optimoptions('fminunc','MaxIterations',1e6,'MaxFunctionEvaluations',1e6,'OptimalityTolerance',1e-5);
    [A,fval,flag]=fminunc(fun,A,options);  
    J1=alpha*trace((Ytr-A(:,3*l+m+2:3*l+m+p+1)'*tanh(A(:,2*l+m+2:3*l+m+1)*fNN(A,Utr)))'*(Ytr-A(:,3*l+m+2:3*l+m+p+1)'*tanh(A(:,2*l+m+2:3*l+m+1)*fNN(A,Utr))));
    J2=beta*trace((fNN(A,Utr)-mean(fNN(A,Utr)')')'*Q*(fNN(A,Utr)-mean(fNN(A,Utr)')'));
    J3=gamma*trace(A(:,2*l+m+2:3*l+m+p+1)'*A(:,2*l+m+2:3*l+m+p+1));      
end
Xpr=fNN(A,Utr);
Vx=[var(Xpr(1,:)); var(Xpr(2,:)); var(Xpr(3,:))];
for s0=l:-1:1
   if Vx(s0)>epsilon
       s=s0;      % Number of significant variables
       break;
   end    
end 
Vx0=Vx;
J2a=beta*(NT-1)*(Q(1,1)*Vx(1)+Q(2,2)*Vx(2)+Q(3,3)*Vx(3));    

%% Performance evaluation on training and testing data
% Predicting the training output with SSNN
Xtrp=zeros(l,NT+1);  Xtrp(:,1)=A(:,3*l+m+p+2); Ytrp=zeros(p,NT); 
bf1s=A(:,l+m+1)+A(:,s+1:l)*mean(Xpr(s+1:l,:)')';    %  The residual state variables are replaced with its mean value and incorporated in the bias term
bg1s=A(:,2*l+m+2+s:3*l+m+1)*mean(Xpr(s+1:l,:)')';   %  The residual state variables are replaced with its mean value and incorporated in the bias term
for k=1:NT
Xtrp(:,k+1)=A(:,l+m+2:2*l+m+1)*tanh(A(:,1:l)*Xtrp(:,k)+A(:,l+1:l+m)*Utr(:,k)+A(:,l+m+1));
Ytrp(:,k)=A(:,3*l+m+2:3*l+m+p+1)'*tanh(A(:,2*l+m+2:3*l+m+1)*Xtrp(:,k));
end  

% Step response of CSTR with ODE
B = 22.0;Da = 0.082;beta = 3.0;
T=1;Nt=25;
xk=[0;0];
yp=zeros(p,Nt+1);

for k=1:1:Nt+1
t(k)=k*T;
td(k)=(k-1)*T;
tk=t(k);
time=[tk tk+T];
uk=Uts(k);
[tt,xx]=ode45('ode_cstr',time,xk,[],B,Da,beta,uk);
xk=xx(length(xx),:)'; 
yp(k+1)=xk(2)+0.05*randn(1);
end

%Estimating the initial state
Xts10=randn(l,1);
options1 = optimoptions('fsolve','MaxIterations',1e7,'MaxFunctionEvaluations',1e6)
fun0=@(Xts1)[Yts(:,1)-A(:,3*l+m+2:3*l+m+p+1)'*tanh(A(:,2*l+m+2:3*l+m+1)*Xts1); Yts(:,2)-A(:,3*l+m+2:3*l+m+p+1)'*tanh(A(:,2*l+m+2:3*l+m+1)*(A(:,l+m+2:2*l+m+1)*tanh(A(:,1:l)*Xts1+A(:,l+1:l+m)*Uts(:,1)+A(:,l+m+1))))];
% Use the below fun0 for step response
%fun0=@(Xts1)[yp(:,1)-A(:,3*l+m+2:3*l+m+p+1)'*tanh(A(:,2*l+m+2:3*l+m+1)*Xts1); yp(:,2)-A(:,3*l+m+2:3*l+m+p+1)'*tanh(A(:,2*l+m+2:3*l+m+1)*(A(:,l+m+2:2*l+m+1)*tanh(A(:,1:l)*Xts1+A(:,l+1:l+m)*Uts(:,1)+A(:,l+m+1))))]; 
[Xts1,fval0,flag0] = fsolve(fun0,Xts10,options1);


% Predicting the testing output with SSNN    -
xtestp=zeros(l,400);  utestp=Uts;   xtestp(:,1)=Xts1;  ytest=Yts;
for k=1:400
xtestp(:,k+1)=A(:,l+m+2:2*l+m+1)*tanh(A(:,1:l)*xtestp(:,k)+A(:,l+1:l+m)*utestp(:,k)+A(:,l+m+1));
ytestp(:,k)=A(:,3*l+m+2:3*l+m+p+1)'*tanh(A(:,2*l+m+2:3*l+m+1)*xtestp(:,k));
end 
msetr0=mse(Ytr,Ytrp);
msets0=mse(ytest,ytestp);

%% Plotting Results
figure(1)
plot(Ytr,'r', LineWidth=3)
hold on
plot(Ytrp,'g', LineWidth=2)
xlabel('$k$','Interpreter','latex');ylabel('$y_{tr}, \hat{y}_{tr}$','Interpreter','latex');
legend('$y_{tr}$', '$\hat{y}_{tr}$','Interpreter','latex')
grid on

figure(2)
plot(ytest,'r', LineWidth=3)
hold on
plot(ytestp,'g', LineWidth=2)
xlabel('$k$','Interpreter','latex');ylabel('$y_{ts}, \hat{y}_{ts}$','Interpreter','latex');
legend('$y_{ts}$', '$\hat{y}_{ts}$','Interpreter','latex')
grid on

% For plotting step response
% figure(2)
% plot(td,yp(:,1:26),'r', LineWidth=3)
% hold on
% plot(td,ytestp(:,1:26),'g', LineWidth=2)
% xlabel('$k$','Interpreter','latex');ylabel('$y_{k}, \hat{y}_{o_k}$','Interpreter','latex');
% legend('$y_{k}$', '$\hat{y}_{o_k}$','Interpreter','latex')
% grid on

%% Function computing the predicted state sequence using the state subnetwork fNN
function xpred=fNN(A,u)    
global NT N n m Ts l p
xpred=zeros(l,NT); xpred(:,1)=A(:,3*l+m+p+2);
for i=1:NT-1
    xpred(:,i+1)=A(:,l+m+2:2*l+m+1)*tanh(A(:,1:l)*xpred(:,i)+A(:,l+1:l+m)*u(:,i)+A(:,l+m+1));
end
end
