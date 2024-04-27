// Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <Eigen/Sparse>
#include <random>
#include <mutex>
#include <chrono>

#include "graphFilter.hpp"

using namespace std;
using namespace pcc_processing;
using namespace Eigen;
using namespace graphFiltering;

/*
   ***********************************************
   Implementation of local functions
   ***********************************************
 */

/**!
 * \function
 *   Compute the minimum and maximum NN distances, find out the
 *   intrinsic resolutions
 * \parameters
 *   @param cloudA: point cloud
 *   @param minDist: output
 *   @param maxDist: output
 * \note
 *   PointT typename of point used in point cloud
 * \author
 *   Dong Tian, MERL
 */
void
findNNdistances(PccPointCloud &cloudA, double &minDist, double &maxDist)
{
  typedef vector< vector<double> > my_vector_of_vectors_t;
  typedef KDTreeVectorOfVectorsAdaptor< my_vector_of_vectors_t, double >  my_kd_tree_t;

  maxDist =  numeric_limits<double>::min();
  minDist =  numeric_limits<double>::max();
  double distTmp = 0;
  mutex myMutex;

  my_kd_tree_t mat_index(3, cloudA.xyz.p, 10); // dim, cloud, max leaf

#pragma omp parallel for
  for (long i = 0; i < cloudA.size; ++i)
  {
    // cout << "*** " << i << endl;
    // do a knn search
    const size_t num_results = 3;
    vector<size_t> indices(num_results);
    vector<double> sqrDist(num_results);

    KNNResultSet<double> resultSet(num_results);

    resultSet.init( &indices[0], &sqrDist[0] );
    mat_index.index->findNeighbors( resultSet, &cloudA.xyz.p[i][0], SearchParams(10) );

    if (indices[0] != i || sqrDist[1] <= 0.0000000001)
    {
      // Print some warnings
      // cerr << "Error! nFound = " << nFound << ", i, iFound = " << i << ", " << indices[0] << ", " << indices[1] << endl;
      // cerr << "       Distances = " << sqrDist[0] << ", " << sqrDist[1] << endl;
      // cerr << "  Some points are repeated!" << endl;
    }

    else
    {
      // Use the second one. assume the first one is the current point
      myMutex.lock();
      distTmp = sqrt( sqrDist[1] );
      if (distTmp > maxDist)
        maxDist = distTmp;
      if (distTmp < minDist)
        minDist = distTmp;
      myMutex.unlock();
    }
  }
}

/**!
 * function to compute graph adjacency matrix
 *   @param cloud: point cloud input
 *   @param A: graph adjacency matrix
 * \note
 *   PointT typename of point used in point cloud
 * \author
 *   Dong Tian, MERL
 */
void buildGraphEigen(PccPointCloud &cloud, commandPar &cPar, SparseMatrix<double> &A)
{
  bool bVer = false;

  int ptSize = cloud.size;
// construct a kd-tree index:

  typedef vector< vector<double> > my_vector_of_vectors_t;
  typedef KDTreeVectorOfVectorsAdaptor< my_vector_of_vectors_t, double >  my_kd_tree_t;
  my_kd_tree_t mat_index(3, cloud.xyz.p, 10); // dim, cloud, max leaf

  // typedef KDTreeSingleIndexAdaptor< L2_Simple_Adaptor<double, PointCloud<double> >, PointCloud<double>, 3> my_kd_tree_t;
  // my_kd_tree_t mat_index(3, cloud.xyz.p, KDTreeSingleIndexAdaptorParams(10)); // dim, cloud, max leaf

  mutex myMutex;
  // search::KdTree<PointT> tree;
  // tree.setInputCloud(cloud.makeShared());
  double var = 0.0;
  double mu = 0.0;
  size_t nnz = 0;
  int testIdxX = -1;
  int testIdxY = -1;

  clock_t t1 = clock();

  // data structure to store graph edges
  typedef struct {
    int x;                      // graph matrix, row
    int y;                      // graph matrix, column
    float val;                  // graph matrix, distance/weight
  } Elem;

  Elem **myList = (Elem **) malloc( sizeof(Elem*) * ptSize);
  int  *myCount = (int*) malloc(sizeof(int) *  ptSize );

  memset( (char*)myCount, 0, sizeof(int)*ptSize );
  for (size_t i = 0; i < ptSize; i++)
  {
    myList[i] = (Elem *) malloc (sizeof(Elem) * cPar.max_nn );
    memset( (char*)myList[i], 0, sizeof(myList[i]) );
  }

#pragma omp parallel for
  for (size_t i = 0; i < ptSize; i++)
  {
    // std::vector<int> indices;
    // std::vector<float> sqr_distances;

    // int nFound = tree.radiusSearch( cloud.points[i], cPar.radius, indices, sqr_distances, cPar.max_nn );

    const size_t num_results = 3;
    KNNResultSet<double> resultSet(num_results);
    std::vector<std::pair<size_t, double> > ret_matches;
    nanoflann::SearchParams params;
    size_t nMatches = mat_index.index->radiusSearch( &cloud.xyz.p[i][0], cPar.radius, ret_matches, params );

    myMutex.lock();
    if ( nMatches > 0 )
    {
      for (size_t j = 0; j < nMatches; j++)
      {
        // indices[j] => ret_matches[j].first: Index of the matched point
        // sqr_distances[j] = ret_matches[j].second: Distance to the matched point
        if (i != ret_matches[j].first && ret_matches[j].second != 0.0)
        {
          int t, s;         // make sure t < s

          // sqr_distances[j] = sqrt( sqr_distances[j] );
          if (i < ret_matches[j].first )
          {
            t = i;      // @DT: Target. like i.  smaller
            s = ret_matches[j].first;
          }
          else
          {
            t = ret_matches[j].first; // @DT: Target. like i.  smaller
            s = i;
          }

          // @DT: Check if (t, s) exist in t-th row. This is needed for multithreading purpose
          bool bNew = true;
          for (size_t k = 0; k < myCount[t] && bNew; k++)
            if ( myList[t][k].x == t && myList[t][k].y == s )
              bNew = false;

          if (bNew && myCount[t] < cPar.max_nn)
          {                 // A new graph edge
            myList[t][ myCount[t] ].x = t;
            myList[t][ myCount[t] ].y = s;
            myList[t][ myCount[t] ].val = ret_matches[j].second;
            myCount[t] ++;
            if (testIdxX < 0) // purely for debugging purpose
            {
              testIdxY = s;
              testIdxX = t;
            }
            mu += ret_matches[j].second;
            nnz++;
          }
          else if (bNew && myCount[t] == cPar.max_nn)
          {                 // A new potential graph edge and max_nn already reached
            // Find the weakest graph edge among existing edges to the current node
            float maxDist = 0.0;
            int   maxIdx = -1;
            for (size_t k = 0; k < myCount[t]; k++)
              if ( myList[t][k].val > maxDist )
              {
                maxDist = myList[t][k].val;
                maxIdx  = k;
              }

            if (maxDist > ret_matches[j].second)
            {               // The new graph edge replace the weakest graph edge
              myList[t][ maxIdx ].x = t;
              myList[t][ maxIdx ].y = s;
              myList[t][ maxIdx ].val = ret_matches[j].second;
              mu -= maxDist;
              mu += ret_matches[j].second;
            }
          }
        }
      }
    }
    myMutex.unlock();
  }

  // average distance
  mu = mu / nnz;

  vector< Triplet<float> > tripletList;
  tripletList.reserve( nnz*2 );

  for (size_t i = 0; i < ptSize; i++)
    for (size_t j = 0; j < myCount[i]; j++)
      var += (myList[i][j].val - mu) * (myList[i][j].val - mu);

  // variance distance
  var = var / nnz;

  for (size_t i = 0; i < ptSize; i++)
  {
    for (size_t j = 0; j < myCount[i]; j++)
    {
      // cout << "x, y, val: " << myList[i][j].x << ", " <<  myList[i][j].y << ", " <<  myList[i][j].val << endl;
      tripletList.push_back( Triplet<float> ( myList[i][j].x, myList[i][j].y, exp( - (myList[i][j].val * myList[i][j].val) / (2*var) ) ) );
    }
    free( myList[i] );
  }

  free( myList ); myList = NULL;
  free( myCount ); myCount = NULL;

  A.reserve( nnz*2 );
  A.setFromTriplets( tripletList.begin(), tripletList.end() );
  A.makeCompressed();
  tripletList.clear();

  if (bVer)
  {
    cout << "Verifying: ... " << endl;
    cout << "nnz: " << nnz << endl;
    cout << "mu: " << mu << endl;
    cout << "A.coeff(" << testIdxX << ", " << testIdxY << "): " << A.coeff(testIdxX, testIdxY) << endl;
    cout << "A.coeff(" << testIdxY << ", " << testIdxX << "): " << A.coeff(testIdxY, testIdxX) << endl;
    cout << "A.coeff(183, 652): " << A.coeff(183,652) << endl;
    cout << "A.coeff(183, 652): " << A.coeff(652,183) << endl;
  }

  // Until here, lower-left part of A is set

  SparseMatrix<double> At = SparseMatrix<double>(A.transpose());
  A = A + At;

  // Now, A is the adjacent matrix
  if (bVer)
  {
    cout << "Testting: ... " << endl;
    cout << "A.coeff(" << testIdxX << ", " << testIdxY << "): " << A.coeff(testIdxX, testIdxY) << endl;
    cout << "A.coeff(" << testIdxY << ", " << testIdxX << "): " << A.coeff(testIdxY, testIdxX) << endl;
    cout << "A.coeff(183, 652): " << A.coeff(183,652) << endl;
    cout << "A.coeff(183, 652): " << A.coeff(652,183) << endl;
  }

  // Until here, A is the adjacency matrix

  clock_t t2 = clock();
  if (bVer)
    cout << "A takes " << (t2-t1)/CLOCKS_PER_SEC << " seconds." << endl;

  SparseMatrix<double> D( ptSize, ptSize );
  D.reserve( ptSize );

  // @DT: Seems no reduce function for sparseMat? Implement myself
  // reduce( A, DV, 1, CV_REDUCE_SUM );
  // for (size_t i = 0; i < cloud.points.size(); i++)
  //   D[i] = 0.0;

  float *dVec = (float*) malloc(sizeof(float) *  ptSize );
  memset( (void*) dVec, 0, sizeof(float)*ptSize);
#pragma omp parallel for
  for (size_t i = 0; i < A.outerSize(); i++)
  {
    for (SparseMatrix<double>::InnerIterator it(A, i);  it; ++it)
      dVec[i] += it.value();
  }
  for (size_t i = 0; i < A.outerSize(); i++)
  {
    D.coeffRef(i, i) = dVec[i];
  }
  free(dVec); dVec = NULL;

  if (bVer)
  {
    cout << "A.outerSize() = " << A.outerSize() << endl;
    cout << "ptSize        = " << ptSize << endl;
    cout << "D(0,0) = " << D.coeff(0,0) << endl;
    cout << "D(ptSize-1, ptSize-1) = " << D.coeff(ptSize-1, ptSize-1) << endl;
  }
  clock_t t3 = clock();
  if (bVer)
    cout << "D takes " << (t3-t2)/CLOCKS_PER_SEC << " seconds." << endl;

  SparseMatrix<double> L( ptSize, ptSize );
  L = D - A;

  if (bVer)
  {
    cout << "L(0,0) = " << L.coeff(0,0) << endl;
    cout << "L(ptSize-1, ptSize-1) = " << L.coeff(ptSize-1, ptSize-1) << endl;
    cout << "L.coeff(" << testIdxX << ", " << testIdxY << "): " << L.coeff(testIdxX, testIdxY) << endl;
    cout << "L.coeff(" << testIdxY << ", " << testIdxX << "): " << L.coeff(testIdxY, testIdxX) << endl;
    cout << "L.coeff(183, 652): " << L.coeff(183,652) << endl;
    cout << "L.coeff(183, 652): " << L.coeff(652,183) << endl;
  }
  clock_t t4 = clock();
  if (bVer)
    cout << "L takes " << (t4-t3)/CLOCKS_PER_SEC << " seconds." << endl;

  // For output purpose
  A = L;

  if (bVer)
    cout << endl;
}

/**!
 * function to perform graph filter on point cloud
 *   @param cloud: point cloud input
 *   @param A: graph adjacency matrix
 * \note
 *   PointT typename of point used in point cloud
 * \author
 *   Dong Tian, MERL
 */
void graphFilter(PccPointCloud &cloud, SparseMatrix<double> &A, VectorXd &score)
{
  // clock_t t1;

  size_t ptSize = cloud.size;
  MatrixXd xyz( ptSize, 3 );

//#pragma omp parallel for
  for (size_t i = 0; i < ptSize; i++)
  {
    xyz.coeffRef(i,0) = cloud.xyz.p[i][0];
    xyz.coeffRef(i,1) = cloud.xyz.p[i][1];
    xyz.coeffRef(i,2) = cloud.xyz.p[i][2];
  }

  xyz = A * xyz;
  score = xyz.rowwise().norm();

  // cout << "score.sum() = " << score.sum() << endl;
  // score = score / score.sum();

  // clock_t t2 = clock();
  // cout << "Graph filtering takes " << (t2-t1)/CLOCKS_PER_SEC << " seconds." << endl;
  // cout << score << endl;
}

/**!
 * function to compute importance score using a graph filter
 *   @param cloud: point cloud input
 * \author
 *   Dong Tian, MERL
 */
int graphFiltering::computeScore(PccPointCloud &cloud, double *pScore)
{
  commandPar cPar;
  VectorXd score;
  
  auto start = std::chrono::system_clock::now();
  
  double minDist = 0.0;
  double maxDist = 0.0;
  findNNdistances(cloud, minDist, maxDist);
  
  auto end = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "Time: findNNdistances cost " <<  duration << " ms" << std::endl;

  if (minDist == maxDist)
    maxDist = 5*minDist;

  cPar.radius = min( minDist * 10, maxDist * 2 );
  cPar.max_nn = 100;

  int ptSize = cloud.size;

  SparseMatrix<double> A( ptSize, ptSize );

  start = std::chrono::system_clock::now();
  buildGraphEigen( cloud, cPar, A );
  end = std::chrono::system_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "Time: buildGraphEigen cost " <<  duration << " ms" << std::endl;
  
  // compute score
  start = std::chrono::system_clock::now();
  graphFilter( cloud, A, score );
  end = std::chrono::system_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "Time: graphFilter cost " <<  duration << " ms" << std::endl;

  for (int i = 0; i < ptSize; i++)
    pScore[i] = score[i];

  return 0;
}
