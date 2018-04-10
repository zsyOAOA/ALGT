function [classifier,W_cell] = ALGTSVM1(options, data)
% Implementation of ALGTSVM algorithm.
% If you use this code, please cite the following paper:
% @article{yue2017semi,
%  title={Semi-supervised learning through adaptive Laplacian graph trimming},
%  author={Yue, Zongsheng and Meng, Deyu and He, Juan and Zhang, Gemeng},
%  journal={Image and Vision Computing},
%  volume={60},
%  pages={38--47},
%  year={2017},
%  publisher={Elsevier}
% }
% Input:
%	options: structure
%	    fields:
%       options.gamma_A: regularization parameter
%       options.gamma_I: regularization parameter
%       options.gamma_X: regularization parameter
%       optional fields:
%       options.Hinge: 0 or 1. If 0, least square loss; if 1, hinge loss. Default 1.
%       options.UseBias: True of false; Classifier with or without bias term.
%       options.Kernal: Default 'rbf'.
%       options.KernelParam: Default 1.
%       More detail about other fields introduction can be found in lapsvmp.m;
%   data: structure
%       fields:
%       data.X: N x d data matrix, normalized
%       data.Y: N x 1 vector. 1, -1 or 0
%       data.label: Ground Truth

Y = data.label;
X = data.X;
data.K = calckernel(options, X, X);
[data.L, W] = InitilizeW(options, X);
classifier0 = lapsvmp(options, data);

% maximal iteration
maxiter = 100;

ERR = zeros(maxiter, 1);

f0 = data.K(:, classifier0.svs)*classifier0.alpha+classifier0.b;
out0 = sign(f0);
err0 = (length(Y)-nnz(out0==Y))/length(Y);

iter = 1;
ERR(iter)= err0;
f = f0;
W_cell = cell(1, 1);
while (iter<=maxiter)
    W_cell{iter,1} = W;
%     data.L = updateW(options, f, data.X);
    [L,W] = laplacian(options,X,f);
    data.L = L;
    classifier = lapsvmp(options, data);
    f = data.K(:, classifier.svs)*classifier.alpha+classifier.b;
    out = sign(f);
    err = (length(data.Y)-nnz(out==Y))/length(data.Y);
    iter = iter+1;
    ERR(iter) = err;
    if iter>=3
        if ERR(iter) == ERR(iter-1)
            break;
        end
    end
end
ERR = ERR(1:iter);
classifier.ERR = ERR;



