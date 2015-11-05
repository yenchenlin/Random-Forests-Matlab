classdef WeakClassifier < handle
   properties
       options;
       weakModel;
       classifierID; % classifier to be used in the end
       numSplits;
   end

   methods
       function weakClassifierInstance = WeakClassifier(X, y, opts) % constructor
          u= unique(y);
          [N, D]= size(X);
          
          if N == 0
              % edge case. No data reached this leaf. Don't do anything.
              weakClassifierInstance.classifierID = 0;
              return
          end
          
          bestgain = -100;
          % Go over all applicable classifiers and generate candidate weak models
          for classf = opts.classifierID % iterate over each kind of classifier
              modelCandidate= struct;
              maxgain= -1;
              
              if classf == 1
                  % Decision stump

                  % proceed to pick optimal splitting value t, based on Information Gain  
                  for q = 1:opts.numSplits

                      if mod(q-1,5) == 0 % use 5 random splits when training each weak learner
                          r = randi(D); % pick a particular feature
                          col = X(:, r);
                          tmin = min(col); % min value of that particular feature among data
                          tmax = max(col); % max value of that particular feature among data
                      end

                      t = rand(1)*(tmax-tmin)+tmin; % pick a random value between tmin and tmax as threshold
                      dec = col < t;
                      Igain = evalDecision(y, dec, u);

                      if Igain > maxgain % random split 5 times, choose max information gain
                          maxgain = Igain;
                          modelCandidate.r = r; % feature to split
                          modelCandidate.t = t; % threshold of feature to split (r)
                      end 
                  end

              elseif classf == 2
                  % Linear classifier using 2 dimensions

                  % Repeat some number of times: 
                  % pick two dimensions, pick 3 random parameters, and see what happens
                  for q = 1:opts.numSplits
                      % pick 2 particular features
                      r1 = randi(D);
                      r2 = randi(D);
                      w = randn(3, 1);

                      dec = [X(:, [r1 r2]), ones(N, 1)]*w < 0;
                      Igain = evalDecision(y, dec, u);

                      if Igain>maxgain
                          maxgain = Igain;
                          modelCandidate.r1 = r1;
                          modelCandidate.r2 = r2;
                          modelCandidate.w = w;
                      end
                  end

              elseif classf == 3
                  % Conic section weak learner in 2D (not too good presently, what is the
                  % best way to randomly suggest good parameters?
                  % Pick random parameters and see what happens
                  
                  for q= 1:opts.numSplits

                      if mod(q-1,5)==0
                          r1= randi(D);
                          r2= randi(D);
                          w= randn(6, 1);
                          phi= [X(:, r1).*X(:, r2), X(:,r1).^2, X(:,r2).^2, X(:, r1), X(:, r2), ones(N, 1)];
                          mv= phi*w;
                      end

                      t1= randn(1);
                      t2= randn(1);
                      if rand(1)<0.5, t1=-inf; end
                      dec= mv<t2 & mv>t1;
                      Igain = evalDecision(y, dec, u);

                      if Igain>maxgain
                          maxgain = Igain;
                          modelCandidate.r1= r1;
                          modelCandidate.r2= r2;
                          modelCandidate.w= w;
                          modelCandidate.t1= t1;
                          modelCandidate.t2= t2;
                      end
                  end
                  
              elseif classf==4
                  % RBF weak learner: Picks an example and bases decision on distance
                  % threshold

                  % Pick random parameters and see what happens
                  for q= 1:opts.numSplits

                      % this is expensive, lets only recompute every once in a while...
                      if mod(q-1,5)==0
                          x= X(randi(size(X, 1)), :);
                          dsts= pdist2(X, x);
                          maxdsts= max(dsts);
                          mindsts= min(dsts);
                      end

                      t = rand(1)*(maxdsts - mindsts)+ mindsts;
                      dec = dsts < t;
                      Igain = evalDecision(Y, dec, u);

                      if Igain>maxgain
                          maxgain = Igain;
                          modelCandidate.x= x;
                          modelCandidate.t= t;
                      end
                  end

              else
                  fprintf('Error in weak train! Classifier with ID = %d does not exist.\n', classf);
              end

              % see if this particular classifier has the best information gain so
              % far, and if so, save it as the best choice for this node
              if maxgain >= bestgain
                  bestgain = maxgain;
                  weakClassifierInstance.weakModel = modelCandidate;
                  weakClassifierInstance.classifierID= classf;
              end
          end
          
          % assign properties
          weakClassifierInstance.options = opts;
          weakClassifierInstance.numSplits = opts.numSplits;
       end
       
       function predictedLabel = predict(instance, X)
           % X is NxD. 
           [N, D]= size(X);
           if instance.classifierID == 1
               % Decision stump classifier
               predictedLabel = double(X(:, instance.weakModel.r) < instance.weakModel.t);

           elseif instance.classifierID == 2
               % 2-D linear clussifier stump
               predictedLabel = double([X(:, [instance.weakModel.r1, instance.weakModel.r2]), ones(N, 1)]*instance.weakModel.w < 0);
           
           elseif instance.classifierID == 3
               % 2-D conic section learner
               r1= instance.weakModel.r1;
               r2= instance.weakModel.r2;
               phi= [X(:, r1).*X(:, r2), X(:,r1).^2, X(:,r2).^2, X(:, r1), X(:, r2), ones(N, 1)];
               mv= phi*instance.weakModel.w;
               predictedLabel = double(mv<instance.weakModel.t2 & mv>instance.weakModel.t1);
           
           elseif instance.classifierID == 4
               % RBF, distance based learner
               predictedLabel= double(pdist2(X, instance.weakModel.x) < instance.weakModel.t);

           elseif instance.classifierID== 0
               %no classifier was fit because there was no training data that reached
               %that leaf. Not much we can do, guess randomly.
               predictedLabel= double(rand(N, 1) < 0.5);
           else
               fprintf('Error in weak test! Classifier with ID = %d does not exist.\n', classifierID);
           end
       end
   end

   methods (Static)
      function weakClassifierInstance = train(X, y, opts)
          weakClassifierInstance = model.classifier.WeakClassifier(X, y, opts);
      end
   end
end

function Igain= evalDecision(Y, dec, u)
% gives Information Gain provided a boolean decision array for what goes
% left or right. u is unique vector of class labels at this node

    YL= Y(dec);
    YR= Y(~dec);
    H= classEntropy(Y, u);
    HL= classEntropy(YL, u);
    HR= classEntropy(YR, u);
    Igain= H - length(YL)/length(Y)*HL - length(YR)/length(Y)*HR;

end

% Helper function for class entropy used with Decision Stump
function H= classEntropy(y, u)

    cdist= histc(y, u) + 1;
    cdist= cdist/sum(cdist);
    cdist= cdist .* log(cdist);
    H= -sum(cdist);
    
end
