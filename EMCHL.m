function [BX_opt, BY_opt,Wx, Wy] = EMCHL(dataset, trainLabel, param)
bit = param.bit;
maxIter = param.maxIter;
sampleColumn = param.num_samples; 
lambda = param.lambda;
XTrain = dataset.XDatabase;  
YTrain = dataset.YDatabase;  

numTrain = size(trainLabel, 1);

U = ones(numTrain, bit);
U(randn(numTrain, bit) <= 0) = -1;

V = ones(numTrain, bit);
V(randn(numTrain, bit) <= 0) = -1;

BX_opt = U;
BY_opt = V;

Wx = ones(size(XTrain,2),size(BX_opt,2));
Wy = ones(size(YTrain,2),size(BY_opt,2));

SX = ones(numTrain, bit);
SY = ones(numTrain, bit)';
F = zeros(numTrain, bit);
V = zeros(numTrain, bit);
objectivex = 0;

for epoch = 1:maxIter
    % sample Sc
    Sc = randperm(numTrain, sampleColumn);
    
    % update BX
    SX = single(trainLabel * trainLabel(Sc, :)' > 0); % 18000x8 logical
    U = updateColumnU(U, V, SX, Sc, bit, lambda, sampleColumn,F,SY);
    
    % update BY
    SY = single(trainLabel(Sc, :) * trainLabel' > 0);
    V = updateColumnV(V, U, SY, Sc, bit, lambda, sampleColumn,F,SX);
    
    % update F
    F=SX.*SY'+ U.*V + V;
    if bit >= 16
     F(F>0)=1;
     F(F<=0)=-1;
    end

   
 end

    BX_opt = U;
    BY_opt = V;
  
    Wx =(2.*XTrain' * XTrain+ param.gamma * eye(size(XTrain, 2))) \ ...
    (XTrain' * BX_opt+0.16*XTrain'* (SX+BY_opt)); % 500x8; 0.12
    Wy =(2.*YTrain' * YTrain + param.gamma * eye(size(YTrain, 2))) \ ...
    (YTrain' * BY_opt+0.16*YTrain'* (SY'+BX_opt));  % 1000x8; 1.0

   save x8.mat x

end

function U = updateColumnU(U, V, S, Sc, bit, lambda, sampleColumn,f,SY)
m = sampleColumn;
n = size(U, 1);

if bit < 16
      S(S == 0) = -0.8;
      S(S == 1) = 1.2;
end

for k = 1: bit
    TX = lambda * U * V(Sc, :)' / bit;
    AX =1 ./(1 + exp(-TX));
    Vjk = V(Sc, k)';
    p = lambda * ((S - AX) .* repmat(Vjk, n, 1)) * ones(m, 1) / bit + m * lambda^2 * U(:, k) / (4 * bit^2);

    S1 = single(S);
    S1(S1==0)= -0.8;
    U_opt=ones(n, 1);
    
    if bit < 16
       FV=0.1*f(:,k).* V(:,k);  %  iapr 0.1  flick 10
       p=V(:,k)+S1(:,k)+FV+p;
    else
       FV=f(:,k).* V(:,k);
       p = 0.001*(V(:,k)+S1(:,k)+FV)+p;
    end
    
    U_opt(p <=0) = -1;
    U_opt(p > 0) = 1;
    U(:, k) = U_opt;
end
end

function V = updateColumnV(V, U, S, Sc, bit, lambda, sampleColumn,f,SX)
m = sampleColumn;
n = size(U, 1);
if bit < 16
      S(S == 0) = -0.8;
      S(S == 1) = 1.2;
end

for k = 1: bit
    TX = lambda * U(Sc, :) * V' / bit;
    AX = 1 ./(1 + exp(-TX));
    Ujk = U(Sc, k)';
    p = lambda * ((S' - AX') .* repmat(Ujk, n, 1)) * ones(m, 1)  / bit + m * lambda^2 * V(:, k) / (4 * bit^2);
    S = S';
    S1 = single(S);
    S1(S1==0)=-0.8;
    V_opt = ones(n, 1);
    
    if bit < 16
      FU=0.1*f(:,k).*U(:,k);  %  iapr 0.1  flick 0.1
      p = lambda *(U(:,k)+S1(:,k))/bit+FU+p;
    else 
      FU=f(:,k).*U(:,k);
      p = 0.001*(U(:,k)+S1(:,k)+FU)+p;
    end
    
    V_opt(p <= 0) = -1;
    V_opt(p > 0) = 1;
    V(:, k) = V_opt;
    S = S';
end
end
