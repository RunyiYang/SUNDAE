import numpy as np
import graphfilter
import open3d as o3d
import argparse
from utils import read_ply, write_ply
import time
import ctypes
from numba import njit, prange

# 定义并行化函数
@njit(parallel=True)
def compute_score_wrapper(cloud):
    lib = ctypes.CDLL(
        '/DATA_EDS2/zhuzx/Work/3dgs-downsample-backbone/FRPC/graphScore/graphfilter.so')
    # 定义C++函数的参数和返回值类型
    lib.ctypes_computeScore.argtypes = [ctypes.py_object]
    lib.ctypes_computeScore.restype = ctypes.POINTER(ctypes.c_double)
    result = np.empty(cloud.shape[0], dtype=np.float64)
    for i in prange(cloud.shape[0]):
        result[i] = lib.ctypes_computeScore(cloud[i])
        # lib.free(result)
    return result

def testNumbaJit(points=None, 
                 downSampleRate=0.1,
                 weightGraph=0.8):
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
            t0 = time.time()
            # 调用numba进行加速
            result = compute_score_wrapper(points[None, :, :3])
            # 将返回的Python对象转换为NumPy数组
            scores = np.ctypeslib.as_array(result[0], shape=(points.shape[0],))
            # scores = graphfilter.computeScore(points[:, :3])
            print('graphfilter.computeScore cost: ', time.time() - t0, 'seconds')
            score = scores / np.sum(scores)
            score = weightGraph * score + (1 - weightGraph) * np.ones(szOrg) / szOrg
            idx = np.random.choice(szOrg, 
                                   int(szOrg * downSampleRate), 
                                   replace=False, 
                                   p=score)
        else:
            idx = np.random.choice(szOrg, int(szOrg * downSampleRate), replace=False)
        return idx

'''
graphfilter.computeScore API:

Args:
    points: np.array (N,m), m >=3, 
        make sure xyz coordinates is the first three elements
'''
def getSampledIndex(points=None, 
                    downSampleRate=0.1,
                    weightGraph=0.8):
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
            # 加载共享库
            lib = ctypes.CDLL(
                '/DATA_EDS2/zhuzx/Work/3dgs-downsample-backbone/FRPC/graphScore/graphfilter.so')
            # import pdb; pdb.set_trace()
            # 定义C++函数的参数和返回值类型
            lib.ctypes_computeScore.argtypes = [ctypes.py_object]
            lib.ctypes_computeScore.restype = ctypes.POINTER(ctypes.c_double)
            
            t0 = time.time()
            # 调用C++函数
            # points = points.astype(np.float64)
            result = lib.ctypes_computeScore(points[:, :3])
            # 将返回的Python对象转换为NumPy数组
            scores = np.ctypeslib.as_array(result, shape=(points.shape[0],))
            # scores = graphfilter.computeScore(points[:, :3])
            print('graphfilter.computeScore cost: ', time.time() - t0, 'seconds')
            import pdb; pdb.set_trace()
            score = scores / np.sum(scores)
            score = weightGraph * score + (1 - weightGraph) * np.ones(szOrg) / szOrg
            idx = np.random.choice(szOrg, 
                                   int(szOrg * downSampleRate), 
                                   replace=False, 
                                   p=score)
            lib.free(result)
        else:
            idx = np.random.choice(szOrg, int(szOrg * downSampleRate), replace=False)
        return idx

def subsamplePointCloudGranual(coords, attIn, normalIn, density_steps, density_idx, weightGraph):
    output = []
    attOut = []
    normalOut = []
    
    szOrg = coords.shape[0]
    
    if szOrg == 0:
        return output, attOut, normalOut
    
    approach = 'uniform' if weightGraph < 0.001 else 'graph'
    
    density_steps = np.array([0] + density_steps)
    density_idx += 1
    density_steps = np.floor(density_steps * szOrg).astype(int)
    
    if approach == 'graph':
        score = graphfilter.computeScore(coords)
        score = score / np.sum(score)
        print(score)
        score = weightGraph * score + (1 - weightGraph) * np.ones(szOrg) / szOrg
        
    idxPool = np.arange(szOrg)
    idxSampled = []
    
    for i in range(1, density_idx+1):
        szPool = idxPool.size
        
        if density_steps[i] - density_steps[i - 1] == szPool:
            idx = np.arange(szPool)
        else:
            if approach == 'uniform':
                idx = np.random.choice(szPool, density_steps[i] - density_steps[i-1], replace=False)
            elif approach == 'graph':
                idx = np.random.choice(szPool, density_steps[i] - density_steps[i-1], replace=False, p=score)
            else:
                print(f'The approach {approach} is not supported')
                idx = []
        
        idxSampled.extend(idxPool[idx])
        idxPool = np.delete(idxPool, idx)
        
        if approach == 'graph':
            score = np.delete(score, idx)
            score = score / np.sum(score)
    
    idxSampled = np.sort(idxSampled)
    output = coords[idxSampled, :]
    if isinstance(attIn, list):
        if len(attIn) != 0:
            attOut = attIn[idxSampled, :]
    else:
        attOut = None

    if isinstance(normalIn, list):
        if len(normalIn) != 0:
            normalOut = normalIn[idxSampled, :]
    else:
        normalOut = None

    return output, attOut, normalOut

def visual(pc, sampled_pc):
    # set differnet colors for sampled points
    pc = np.concatenate((pc, np.zeros((pc.shape[0], 1))), axis=1)
    sampled_pc = np.concatenate((sampled_pc, np.ones((sampled_pc.shape[0], 1))), axis=1)
    pc = np.concatenate((pc, sampled_pc), axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:])
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_ply", help="path to ply file")
    parser.add_argument("-r", "--sample_rate", default=0.1, type=float, help="number of samples")
    parser.add_argument("-s", "--save", default="", help="path to save output ply file")
    parser.add_argument("-m", "--mode", default="FPRC", help="sparse or dense")
    args = parser.parse_args()

    t1 = time.time()
    pcd_origin, pc = read_ply(args.path_ply)
    print('Time taken to read ply: ', time.time() - t1, 'seconds')

    num_pts = pc.shape[0]
    sample_rate = args.sample_rate
    print('Processing {} points with sample rate of {}'.format(num_pts, sample_rate))

    t1 = time.time()
    if args.mode == "FPRC":
        idxs = getSampledIndex(pc, sample_rate, 1.0)
        print('Time taken to sample: ', time.time() - t1, 'seconds')
        sampled_pc = pcd_origin[idxs]
    elif args.mode == "random":
        sampled_pc = pcd_origin[np.random.choice(num_pts, int(np.floor(sample_rate*num_pts)), replace=False)]
        print('Time taken to sample: ', time.time() - t1, 'seconds')
        
    if args.save != "":
        write_ply(args.save, sampled_pc)
        print('Write to {} successfully!'.format(args.save))
    else:
        visual(pc, sampled_pc.cpu().numpy())
            
