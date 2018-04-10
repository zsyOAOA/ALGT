function [L,W_full] = laplacian(options,X,F)
% laplacian computes the graph Laplacian.
% usage:
%      [L,options] = laplacian(options,X)
% Input:
%      options: a structure with the following fields
%               options.NN: number of nearest neighbors to use
%               options.GraphDistanceFunction: 'euclidean' | 'cosine' |
%               'hamming_distance'
%               options.GraphWeights: 'distance' | 'binary' | 'heat'
%               options.GraphWeightParam: width for 'heat' kernel
%                                         (if set to 0, it uses the mean 
%                                          edge length distance among
%                                          neighbors
%               options.LaplacianNormalize: 0 | 1
%               options.LaplacianDegree: degree of the iterated Laplacian
%      X: N-by-D data matrix (N examples, D dimensions)
% Output:
%      L: sparse symmetric N-by-N Laplacian matrix
%      W: full weighted matrix
%

W = adjacency(options,X,F);
W_full=full(W);
D = sum(W,2);

if options.LaplacianNormalize == 0
    L = spdiags(D,0,speye(size(W,1)))-W; % L = D-W
else
    D(D~=0)=sqrt(1./D(D~=0));
    D=spdiags(D,0,speye(size(W,1)));
    W=D*W*D;
    L=speye(size(W,1))-W; % L = I-D^-1/2*W*D^-1/2
end

if options.LaplacianDegree>1
    L=mpower(L,options.LaplacianDegree);
end
