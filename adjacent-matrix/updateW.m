function L = updateW(options, f, X)
%UPDATEW.m用于更新权重矩阵
%f   上次迭代得到的分类器的预测结果
%X　数据矩阵，无label
%options   结构体
%　　　　options.NN 近邻个数
%　　　　options.GraphDistanceFunction  计算样本间距离时使用的函数, 'eucliden'
%　　　　options.GraphWeightParam  计算权重矩阵时涉及的超参数，
%                                                            即高斯核函数的窗宽,取0表示使用最大值
%L　Laplacian Matrix,以稀疏矩阵形式存储

n=size(X,1);
K = options.NN;
gamma_I = options.gamma_I;
gamma_X = options.gamma_X;
XDis = feval(options.GraphDistanceFunction, X, X);
fDis = feval(options.GraphDistanceFunction, f, f);
Dis = gamma_X*XDis+gamma_I*fDis;
[~,I] = sort(Dis, 2);
Ind = I(: , 2:( K+1));
 
%计算邻接矩阵
S = zeros(n, n);
for i=1:n
    ii = Ind(i, :);
    S(i, ii) = Dis(i, ii);
end

%计算权重矩阵
if  options.GraphWeightParam == 0
    S1 =S(:);
    t = max(S1);
else 
    t = options.GraphWeightParam;
end

 W = exp((-S.^2)/(2*t));
 W = W.*(S>0);
 
 %对称化
 W = W+((W~=W').*W');
 
 %计算Laplacian Matrix
 d = sum(W,2);
 D = diag(d);
 if options.LaplacianNormalize==1
     D(D~=0)=sqrt(1./D(D~=0));
     L = eye(n) - D*W*D;
 else
     L = D-W;
 end
 L = sparse(L);
