classdef ParametricClassifier < handle
   properties
       options;
       treeModels;
   end

   methods
       function parametricClassifierInstance = ParametricClassifier(X, y) % constructor
          % Train a random forest.
          % X is NxD, each D-dimensional row is a data point.
          % For convenience, we assume X is zero mean unit variance.
          % If it isn't, preprocess data with
          %
          % X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X) + 1e-10);
          %
          % If this condition isn't satisfied, some weak learners won't work out
          % of the box in current implementation.
          %
          % Y is discrete Nx1 vector of labels
          
          % preprocess X to make it become zero mean unit variance.
          X = bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X) + 1e-10);
          
          rand('state', 0);
          randn('state', 0);
          
          opts = struct;
          opts.depth = 9;
          opts.numTrees = 100;
          opts.numSplits = 15;
          opts.verbose = true;
          opts.classifierID = 2;
          
          numOfTrees = 100;
          parametricClassifierInstance.treeModels = cell(1, numOfTrees);
          verbose = true;
          
          for i=1:numOfTrees
              %parametricClassifierInstance.treeModels{i} = treeTrain(X, y, parametricClassifierInstance.options);
              parametricClassifierInstance.treeModels{i} = model.classifier.DecisionTree.train(X, y, opts);
              if verbose
                  oneTenNumOfTrees = floor(numOfTrees/10);
                  if mod(i, oneTenNumOfTrees)==0 || i==1 || i==numOfTrees
                      fprintf('Training tree %d/%d...\n', i, numOfTrees);
                  end
              end
          end
          
          % assign properties
          parametricClassifierInstance.options = opts;
       end
       
       function predictedLabel = predict(instance, X)
           % preprocess X to make it become zero mean unit variance.
           X = bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X) + 1e-10);
           
           numTrees = length(instance.treeModels);
           u = instance.treeModels{1}.classes; % Assume we have at least one tree!
           Ysoft = zeros(size(X,1), length(u));
           for i = 1:numTrees
               [~, ysoft] = instance.treeModels{i}.predict(X);
               Ysoft= Ysoft + ysoft;
           end
           Ysoft = Ysoft/numTrees;
           [~, ix]= max(Ysoft, [], 2);
           predictedLabel = u(ix);
       end
   end

   methods (Static)
      function parametricClassifierInstance = train(X, y)
          parametricClassifierInstance = model.classifier.ParametricClassifier(X, y);
      end
   end
end
