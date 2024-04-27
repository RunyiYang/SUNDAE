% Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
%
% SPDX-License-Identifier: AGPL-3.0-or-later

  addpath('./graphScore');

  % subsampling ratio set in *ascending* order
  density_steps = [0.05, 0.1, 0.125, 0.2, 0.3, 0.5];

  % weightGraph range: [0.0, 1.0].
  % 0.0: -> equal importance for each point.
  % 1.0: -> scores are 100% from graph filtering
  weightGraph = 0.8;

  % load a point cloud
  currentPC = pcread('ply/cubic.ply');
  Vcurr = double(currentPC.Location);

  % show the input point cloud
  figure;
  scatter3(Vcurr(:, 1), Vcurr(:, 2), Vcurr(:, 3), 5, ones(size(Vcurr,1),1), '.');
  title('org');

  Ccurr = [];
  Vnormal = [];

  for i = 1:size(density_steps,2)
    densityIdx = i;
    [Vcurr1, ~, ~] = subsamplePointCloudGranual( Vcurr, Ccurr, Vnormal, density_steps, densityIdx, weightGraph );

    figure;
    scatter3(Vcurr1(:, 1), Vcurr1(:, 2), Vcurr1(:, 3), 5, ones(size(Vcurr1,1),1), '.');
    title0 = sprintf('subsampling ratio: %.1f%%', density_steps(densityIdx)* 100);
    title( title0 );
  end
