% Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
%
% SPDX-License-Identifier: AGPL-3.0-or-later

function [output, attOut, normalOut] = subsamplePointCloudGranual( coords, attIn, normalIn, density_steps, density_idx, weightGraph )
% Perform subsample from finer to coarser
% density_steps, sorted from smaller to bigger numbers

  output = [];
  attOut = [];
  normalOut = [];

  szOrg = size( coords, 1 );

  if (szOrg == 0)
    return;
  end

  if (weightGraph < 0.001)
    approach = 'uniform';
  else
    approach = 'graph';
  end

  density_steps = [0, density_steps];
  density_idx = density_idx + 1;
  density_steps = floor( density_steps * szOrg ); % # of points now

  if ( strcmp( approach, 'graph') )
    score = mxGraphFilter( coords );
    score = score / sum(score);
    score = weightGraph * score + (1-weightGraph) * ones(szOrg, 1) / szOrg;
  end

  s = RandStream('mlfg6331_64');% For reproducibility

  idxPool = 1:szOrg;
  idxSampled = [];
  for i = 2:density_idx
    szPool = size( idxPool, 2 );

    if density_steps(i) - density_steps(i-1) == szPool
      idx = 1:szPool;
    else
      % samples to be dropped
      if (strcmp( approach, 'uniform' ))
        idx = datasample(s, 1:szPool, density_steps(i) - density_steps(i-1), 'Replace', false );
      elseif (strcmp( approach, 'graph') )
        idx = datasample(s, 1:szPool, density_steps(i) - density_steps(i-1), 'Replace', false, 'Weights', score );
      else
        fprintf( 'The approach %s is not supported\n', approach );
        idx = [];
      end
    end

    idxSampled = [idxSampled, idxPool(idx)];% index of all elements to be sampled in the end
    idxPool( idx ) = [];% index of elements remained to be further sampled in future iterations. 4 5 6 7 8 9 10

    if (strcmp( approach, 'graph') )
      score(idx) = [];
    end
  end

  idxSampled = sort( idxSampled );
  output = coords(idxSampled, :);
  if (~isempty( attIn ) )
    attOut = attIn(idxSampled, :);
  end
  if (~isempty( normalIn ) )
    normalOut = normalIn(idxSampled, :);
  end

end
