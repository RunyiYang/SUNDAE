// Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>
#include <vector>
#include <math.h>
#include "mex.h"
#include "pccProcessing.hpp"
#include "graphFilter.hpp"

using namespace pcc_processing;
using namespace graphFiltering;

/* The gateway function. */
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {

  /* Check for proper number of arguments */
  if(nrhs != 1) {
    mexErrMsgIdAndTxt("MATLAB:mxGraphFilter:nargin",
                      "MXGRAPHFILTER requires one input argument.");
  }
  if(nlhs != 1) {
    mexErrMsgIdAndTxt("MATLAB:mxGraphFilter:nargout",
                      "MXGRAPHFILTER requires one output argument.");
  }

  /* Check if the input is of proper type */
  if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) ) {
    mexErrMsgIdAndTxt("MATLAB:mxGraphFilter:typeargin",
                      "First argument has to be double list.");
  }
  int size = mxGetM(prhs[0]);
  int dim = mxGetN(prhs[0]);
  if( dim != 3 ) {
    mexErrMsgIdAndTxt("MATLAB:mxGraphFilter:typeargin",
                      "First argument has to be 3-dim list.");
  }

  /* Acquire pointers to the input data */
  double* pCoords = mxGetPr(prhs[0]); // Org block

  PccPointCloud inCloud;

  inCloud.loadBlock( pCoords, mxGetM(prhs[0]) );

  /* output */
  /* prepare for output */
  plhs[0] = mxCreateDoubleMatrix(size, 1, mxREAL);
  double *pScore = mxGetPr(plhs[0]);

  /* compute */
  computeScore(inCloud, pScore);
}
