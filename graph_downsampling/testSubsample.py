import numpy as np
import open3d as o3d
import py3Dmol
from subsamplePointCloudGranual import subsamplePointCloudGranual 
from IPython.display import display
from plyfile import PlyData, PlyElement

# Load the point cloud using open3d
pcd = o3d.io.read_point_cloud('FRPC/ply/cubic.ply')
Vcurr = np.asarray(pcd.points) # 14408x3

# subsampling ratio set in ascending order
density_steps = [0.05, 0.1, 0.125, 0.2, 0.3, 0.5]
weightGraph = 0.8

# Function to visualize point cloud using py3Dmol
def visualize_point_cloud(points):
    view = py3Dmol.view(width=800, height=400)
    xyz_str = ''.join(['{} {} {}\n'.format(pt[0], pt[1], pt[2]) for pt in points])
    view.addModel(xyz_str, format='xyz')
    view.setStyle({'sphere': {'radius': 0.3}})
    view.zoomTo()
    return view

def write_ply(points, filename):
    # 创建一个三维点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 保存为.PLY文件
    o3d.io.write_point_cloud(filename, pcd)

# Visualize the original point cloud
view = visualize_point_cloud(Vcurr)
display(view)

Ccurr = []
Vnormal = []

for density in density_steps:
    Vcurr1, _, _ = subsamplePointCloudGranual(Vcurr, Ccurr, Vnormal, density_steps, density_steps.index(density), weightGraph)
    write_ply(Vcurr1, 'FRPC/ply/cubic_'+str(density_steps.index(density))+'.ply')
    # Visualize the subsampled point cloud
    view = visualize_point_cloud(Vcurr1)
    display(view)
