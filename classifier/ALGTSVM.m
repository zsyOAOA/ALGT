function [classifier,classifier0] = ALGTSVM(options, data)
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
%
% Note: This code is based on the implementation of LapLacian SVM, which is
% avaliable at:
%   http://www.dii.unisi.it/~melacci/lapsvmp/index.html
% And the relative paper is:
% @article{belkin2006manifold,
% title={Manifold regularization: A geometric framework for learning from labeled and unlabeled examples},
% author={Belkin, Mikhail and Niyogi, Partha and Sindhwani, Vikas},
% journal={Journal of machine learning research},
% volume={7},
% number={Nov},
% pages={2399--2434},
% year={2006}
%}
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
% Output:
%   classifier: classifier result of ALGMSVM
%   classifier0: classifier result of LapSVM

%Y = data.label;
X = data.X;
data.K = calckernel(options, X, X);
[data.L,~] = InitilizeW(options, X);
classifier0 = lapsvmp(options, data);

% maximal iteration
maxiter = 100;

f0 = data.K(:, classifier0.svs)*classifier0.alpha+classifier0.b;
%out0 = sign(f0);
%err0 = (length(Y)-nnz(out0==Y))/length(Y);

iter = 1;
f = f0;
while (iter<=maxiter)
    if options.display && mod(iter, 10) == 0
        fprintf('Iter = %d...\n', iter);
    end
    [L,~] = laplacian(options,X,f);
    data.L = L;
    classifier = lapsvmp(options, data);
    f = data.K(:, classifier.svs)*classifier.alpha+classifier.b;
    iter = iter+1;
end


