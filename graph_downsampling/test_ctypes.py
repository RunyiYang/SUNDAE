import numpy as np
import open3d as o3d
import argparse
from utils.graphsample import read_ply, write_ply
import time
import ctypes
from numba import njit, prange

'''
graphfilter.computeScore API:

Args:
    points: np.array (N,m), m >=3, 
        make sure xyz coordinates is the first three elements
'''
def getSampledIndex(points,
                    DLLPath="graph_downsampling/graphScore/filtergraphv2.so",
                    downSampleRate=0.1,
                    weightGraph=0.6,
                    strictSample=True):
    '''
    Args:
        points: np.array (N,m)
        downSampleRate: float, if downSampleRate is below 0.001, use uniform strategy
        weightGraph: float
    '''
    if points is None:
        raise ValueError("points must not be None")
    if points.shape[0] == 0:
        raise ValueError("points must not be None")
    
    szOrg = points.shape[0]
    if downSampleRate == 1:
        return np.arange(points.shape[0])
    else:
        approach = 'uniform' if downSampleRate < 0.001 else 'graph'
        if approach == 'graph':
            points_copy = np.ascontiguousarray(np.copy(points[:,:3].T)).astype(np.float64)
            points_T_ptr = points_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            ptsSize = points_copy.shape[1]
            # 加载共享库
            lib = ctypes.CDLL(DLLPath) # TODO: try and catch
            # import pdb; pdb.set_trace()
            # 定义C++函数的参数和返回值类型
            lib.ctypes_computeScore.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
            lib.ctypes_computeScore.restype = ctypes.POINTER(ctypes.c_double * ptsSize)
            lib.freepScores.argtypes = [ctypes.POINTER(ctypes.c_double * ptsSize)]

            # import pdb; pdb.set_trace()
            t0 = time.time()
            # 调用C++函数
            result = lib.ctypes_computeScore(points_T_ptr, ptsSize)
            # 将返回的Python对象转换为NumPy数组
            # scores = np.ctypeslib.as_array(result, shape=(points.shape[0],))
            scores = np.frombuffer(result.contents)
            # scores = graphfilter.computeScore(points[:, :3])
            print('graphfilter.computeScore cost: ', time.time() - t0, 'seconds')
            # import pdb; pdb.set_trace()
            score = scores / np.sum(scores)
            # high and low frequency number
            # import pdb; pdb.set_trace()
            if strictSample:
                highFrequencyNum = int(szOrg * downSampleRate * weightGraph)
                lowFrequencyNum = int(szOrg * downSampleRate * (1 - weightGraph))
                print('high frequency number:', highFrequencyNum)
                print('low frequency number:', lowFrequencyNum)

                indexHigh = np.argpartition(score, -highFrequencyNum)[-highFrequencyNum:szOrg]
                indexLow = np.argpartition(score, lowFrequencyNum-1)[0:lowFrequencyNum]
                idx = np.concatenate((indexHigh, indexLow))
            else:
                score = weightGraph * score + (1 - weightGraph) * np.ones(szOrg) / szOrg
                idx = np.random.choice(szOrg, 
                                    int(szOrg * downSampleRate), 
                                    replace=False, 
                                    p=score)

            lib.freepScores(result)
        else:
            idx = np.random.choice(szOrg, int(szOrg * downSampleRate), replace=False)
        return idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_ply", help="path to ply file")
    parser.add_argument("-r", "--sample_rate", default=0.1, type=float, help="number of samples")
    parser.add_argument("-s", "--save", default="", help="path to save output ply file")
    parser.add_argument("-m", "--mode", default="FPRC", help="sparse or dense")
    parser.add_argument("-d", "--dll_path", default="graphScore/filtergraphv2.so", help="C++ .so file path")
    args = parser.parse_args()

    t1 = time.time()
    pcd_origin, pc = read_ply(args.path_ply)
    print('Time taken to read ply: ', time.time() - t1, 'seconds')

    num_pts = pc.shape[0]
    sample_rate = args.sample_rate
    print('Processing {} points with sample rate of {}'.format(num_pts, sample_rate))

    t1 = time.time()
    if args.mode == "FPRC":
        idxs = getSampledIndex(pc, args.dll_path, sample_rate, 1.0)
        print('Time taken to sample: ', time.time() - t1, 'seconds')
        sampled_pc = pcd_origin[idxs]
    elif args.mode == "random":
        sampled_pc = pcd_origin[np.random.choice(num_pts, int(np.floor(sample_rate*num_pts)), replace=False)]
        print('Time taken to sample: ', time.time() - t1, 'seconds')
        
    if args.save != "":
        write_ply(args.save, sampled_pc)
        print('Write to {} successfully!'.format(args.save))

            
