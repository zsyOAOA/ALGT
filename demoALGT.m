clear;
clc;

% setting default paths

% setting random seed
seed = 2001;
rng('default'); rng(seed);

% generating a random dataset
X=[randn(500,5); 2*randn(500,5)+2];
Y=[ones(500,1); -ones(500,1)];

% generating default options
options=make_options('gamma_I',1,'gamma_A',1e-5,'KernelParam',0.35);
options.UseBias=1;
options.UseHinge=1;
options.LaplacianNormalize=0;

% creating the 'data' structure
data.X=X;
data.Y=zeros(size(Y));
data.Y(1:50)=1; % 50 labeled points of class +1
data.Y(501:550)=-1; % 50 labeled points of class -1

% Computing Gram matrix and Laplacian
data.K=calckernel(options,X,X);
%data.L=laplacian(options,X,X);

% training the classifier
for K = 3:2:15
    fprintf('Train Classifier with K = %d\n', K);
    options.NN = K;
    [classifier,classifier0] = ALGTSVM(options, data);

    % computing error rate
    out0=sign(data.K(:,classifier0.svs)*classifier0.alpha+classifier0.b);
    er0=100*(length(data.Y)-nnz(out0==Y))/length(data.Y);
    fprintf('\tLapSVM Error rate=%.1f\n',er0);

    out1=sign(data.K(:,classifier.svs)*classifier.alpha+classifier.b);
    er=100*(length(data.Y)-nnz(out1==Y))/length(data.Y);
    fprintf('\tALGMSVM Error rate=%.1f\n',er);
end

