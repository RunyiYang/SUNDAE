import numpy as np
import torch 
from plyfile import PlyData, PlyElement
import open3d as o3d

def graphFilter(points, adj_mat, is_sparse):
    if is_sparse:
        xyz = torch.sparse.mm(adj_mat, points)
    else:
        xyz = torch.mm(adj_mat, points)
        
    score = torch.norm(xyz, dim=-1)
    return score

def data_sample(k, replacement, score):
    # sample k points with replacement from score
    # return the indices of the sampled points
    # score: (N, 1)
    # return: (k, 1)
    score = score.squeeze()
    idx = torch.multinomial(score, k, replacement=replacement)
    return idx

def write_ply(save_path,points,text=True):
    """
    save_path : path to save: '/yy/XX.ply'
    pt: point_cloud: size (N,3)
    """
    # points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    # vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(points, 'vertex')
    PlyData([el]).write(save_path)
    
def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    x = pc['x']
    y = pc['y']
    z = pc['z']
    points = np.stack((x,y,z), axis=1)
    return pc, points

