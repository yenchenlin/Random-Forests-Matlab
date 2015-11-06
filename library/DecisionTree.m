classdef DecisionTree < handle
   properties
       options;
       classes;
       leafdist;
       weakModels;
   end

   methods
       function decisionTreeInstance = DecisionTree(X, y, opts)  % constructor
          % Train a random tree
          % X is NxD, each D-dimensional row is a data point.
          % y is discrete Nx1 vector of labels
          
          d = opts.depth; % max depth of the tree
          
          u = unique(y);
          
          [N, D]= size(X);
          nd = 2^d - 1; % number of nodes in a tree
          numInternals = (nd+1)/2 - 1; % number of internal nodes in a tree
          numLeafs= (nd+1)/2; % number of leaf nodes in a tree
          
          decisionTreeInstance.weakModels = cell(1, numInternals); % each internal nodes correspond to a weak model

          relPredictResult = zeros(N, nd); % (type: double) predict result of relevant data at each node
          decisionTreeInstance.leafdist = zeros(numLeafs, length(u)); % leaf distribution
          
          for n = 1:numInternals
              % get relevant data at this node
              if n == 1 % relevant data at root equals all X and y
                  reld = ones(N, 1) == 1;
                  Xrel = X;
                  yrel = y;
              else
                  reld = relPredictResult(:, n) == 1;
                  Xrel = X(reld, :);
                  yrel = y(reld);
              end
              
              % train weak models
              % decisionTreeInstance.weakModels{n} = weakTrain(Xrel, yrel, opts);
              decisionTreeInstance.weakModels{n} = model.classifier.WeakClassifier.train(Xrel, yrel, opts);
              
              % split data to child nodes
              yhat= decisionTreeInstance.weakModels{n}.predict(Xrel);
              
              relPredictResult(reld, 2*n)= yhat;
              relPredictResult(reld, 2*n + 1)= 1 - yhat; % since yhat is in {0,1} and double
          end
          
          % Go over leaf nodes and assign class statistics
          for n= (nd+1)/2 : nd
              reld= relPredictResult(:, n);
              hc = histc(y(reld==1), u); % histogram count
              hc = hc + 1; % Dirichlet prior
              decisionTreeInstance.leafdist(n - (nd+1)/2 + 1, :)= hc / sum(hc);
          end
          
          % assign properties
          decisionTreeInstance.options = opts;
          decisionTreeInstance.classes = u;
       end
       
       function [Yhard, Ysoft] = predict(instance, X)
           d = instance.options.depth;
           
           [N, D]= size(X);
           nd= 2^d - 1;
           numInternals = (nd+1)/2 - 1;
           
           u = instance.classes;
           
           Yhard = zeros(N, 1);
           Ysoft = zeros(N, length(u));
           
           relPredictResult = zeros(N, nd); % (type: double) predict result of relevant data at each node

           % Propagate data down the tree using weak classifiers at each node
           for n = 1:numInternals
    
               % get relevant data at this node
               if n==1 
                   reld = ones(N, 1) == 1;
                   Xrel= X;
               else
                   reld = relPredictResult(:, n) == 1;
                   Xrel = X(reld, :);
               end
               if size(Xrel, 1) == 0, continue; end % empty branch, ah well
    
               yhat= instance.weakModels{n}.predict(Xrel);
    
               relPredictResult(reld, 2*n)= yhat;
               relPredictResult(reld, 2*n+1)= 1 - yhat; % since yhat is in {0,1} and double
           end
           
           % Go over leafs and assign class probabilities
           for n = (nd+1)/2 : nd
               ff = find(relPredictResult(:, n)==1); % indexes of data whose y == 1 at this leaf
    
               hc = instance.leafdist(n - (nd+1)/2 + 1, :); % histogram count
               vm = max(hc);
               miopt = find(hc == vm); % class(es) with highest histogram count
               mi = miopt(randi(length(miopt), 1)); % choose a class arbitrarily if ties exist
               Yhard(ff) = u(mi);
               Ysoft(ff, :)= repmat(hc, length(ff), 1);
           end

       end
   end

   methods (Static)
      function decisionTreeInstance = train(X, y, opts)
          decisionTreeInstance = model.classifier.DecisionTree(X, y, opts);
      end
   end
end
