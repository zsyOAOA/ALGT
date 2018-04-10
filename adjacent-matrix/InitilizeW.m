function  [L, W] = InitilizeW(options,X)
% Input:
%   options: structure
%       options.NN: number of neighbour sample
%       options.GraphDistanceFunction: distance measurement
%       options.GraphWeightParam: window width of Gauss kernel
%       options.LaplacianNormalize: 0 or 1. Normalize Laplacian Matrix.
%   X:  N x d matrix
% Output:
%   L: sparse Laplacian matrix
%   W: weighted matrix

n=size(X,1);
K = options.NN;
XDis = feval(options.GraphDistanceFunction, X, X);
[~,I] = sort(XDis, 2);
Ind = I(: , 2:( K+1));

% adjacent matrix
S = zeros(n, n);
for i=1:n
    ii = Ind(i, :);
    S(i, ii) = XDis(i, ii);
end

% weighted matrix
if  options.GraphWeightParam == 0
    S1 =S(:);
    t = mean(S1(S1>0));
else
    t = options.GraphWeightParam;
end
W = exp((-S.^2)/(2*t*t));
W = W.*(S>0);

% symmetrization
W = W+((W~=W').*W');

% Laplacina matrix
d = sum(W,2);
D = diag(d);
if options.LaplacianNormalize==1
    D(D~=0)=sqrt(1./D(D~=0));
    L = eye(n) - D*W*D;
else
    L = D-W;
end
L = sparse(L);


