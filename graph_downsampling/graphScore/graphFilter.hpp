/*
 * @Author: jike5 2719825969@qq.com
 * @Date: 2023-11-02 16:58:57
 * @LastEditors: jike5 2719825969@qq.com
 * @LastEditTime: 2023-11-03 00:09:55
 * @FilePath: /3dgs-downsample-backbone/FRPC/graphScore/graphFilter.hpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
// Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef GRAPHFILTER_HPP
#define GRAPHFILTER_HPP

#include "pccProcessing.hpp"
#include "nanoflann/nanoflann.hpp"

using namespace nanoflann;
#include "nanoflann/KDTreeVectorOfVectorsAdaptor.h"

using namespace pcc_processing;

namespace graphFiltering {

  class commandPar
  {
  public:
    float radius;             //! radius to construct the graph
    int max_nn;               //! max_nn for radius search

    commandPar()
      {
        max_nn = 30;
        radius = 0.0;
      }
  };


  int computeScore( PccPointCloud &cloud, double* pScore );

}

#endif
