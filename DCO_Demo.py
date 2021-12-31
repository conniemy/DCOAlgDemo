
from typing import Dict, Any

import cv2
from cv2 import DISOpticalFlow
import numpy as np
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import scipy
import scipy.sparse
import scipy.sparse.linalg
import GetSparseDepth
import os
import time
import copy
import sys


# # README
# 
# #### 需要的模组:
# - OpenCV 4.0
# - opencv-contrib (`pip install opencv-contrib-python`)
# - NumPy
# - pyquaternion (`pip install pyquaternion`)
#
# #### 运行输入说明:
# - 基于三个变量进行读取
#   - **input_frames**: 输入视频帧路径
#   - **input_recon**: 稠密化所需参数表所在文件夹. 参数文件使用COLMAP格式(https://colmap.github.io/format.html).目录需要包含三个文件: `points2D.txt`, `images.txt`, and `cameras.txt`.
#   - **output_folder**: 数据输出目录
#



input_frames = "densify_data/frames/"
input_colmap = "densify_data/reconstruction/"
output_folder = "output_frames/depthframes/"
output_folder_mixed = "output_frames/ocluframes/"
input_video = "input_video/"
output_video = "output_video/"
ori_frame_folder = "sparse_data/frames/"
ori_params_folder = "sparse_data/calibration/"

dump_debug_images = False

# 算法参数
tau_high = 0.04     #Canny算法高阈值
tau_low = 0.01      #Canny算法低阈值
tau_flow = 0.2      #Canny算法深度滤镜阈值0.3
k_I = 5             #作用于深度轮廓提取的高斯滤波窗口大小
k_T = 7             #（时间轴向）中值滤波窗口大小
k_F = 31            #作用于光流梯度强度方框滤波的窗口大小
lambda_d = 1        #二次优化——深度信息置信权重
lambda_t = 0.01     #二次优化——前一帧深度信息置信权重
lambda_s = 1        #二次优化——平滑权重

num_solver_iterations = 500
scaling_factor = 1.0
input_video_name = "ori_video.mp4"
input_video_params_name = "videoParams.txt"
input_video_cali_params_name = "camera.txt"
depth_video_name = "depth.mp4"
mix_video_name = "mixed.avi"

fx = 493.2261388164407
fy = 492.3052568731896
cx = 615.217412230966
cy = 460.7633711007789

k1 = 0.5801358893633602
k2 = 0.1488700681725693
k3 = -0.5065295812560362
k4 = 0.2129197100544011

densify_w = 0
densify_h = 0

cam_model = 2

# ## 配置读取类
class Reconstruction:
    views: Dict[Any, Any]

    def __init__(self):
        self.cameras = {}           #相机信息{id:Camera}
        self.views = {}             #SLAM帧信息{id:View}
        self.points3d = {}          #空间点绝对坐标目录{id:Point}
        self.min_view_id = -1       #最小SLAM帧ID
        self.max_view_id = -1       #最大SLAM帧ID
        self.image_folder = ""      #SLAM帧信息存储目录路径
    
    def ViewIds(self):
        return list(self.views.keys())

    #获取某ID帧的前后关键帧
    def GetNeighboringKeyframes(self, view_id):
        previous_keyframe = -1
        next_keyframe = -1
        for idx in range(view_id - 1, self.min_view_id, -1):
            if idx not in self.views:
                continue
            if self.views[idx].IsKeyframe():
                previous_keyframe = idx
                break
        for idx in range(view_id + 1, self.max_view_id):
            if idx not in self.views:
                continue
            if self.views[idx].IsKeyframe():
                next_keyframe = idx
                break
        if previous_keyframe < 0 or next_keyframe < 0:
            return np.array([])
        return [previous_keyframe, next_keyframe]

    #获取某ID帧的前后参照帧
    def GetReferenceFrames(self, view_id):
        kf = self.GetNeighboringKeyframes(view_id)  #获取前后邻居关键帧的ID
        if (len(kf) < 2):   #若前后没有关键帧则返回空列表
            return []
        dist = np.linalg.norm(self.views[kf[1]].Position() - self.views[kf[0]].Position()) / 2   #计算前后两DSO关键帧的相机位姿距离
        pos = self.views[view_id].Position()    #获取本帧相机位置
        ref = []
        for idx in range(view_id + 1, self.max_view_id):    #向后面的帧视图搜索，找到第一个与本视图相机距离大于前后DSO关键帧相机距离的帧
            if idx not in self.views:
                continue
            if (np.linalg.norm(pos - self.views[idx].Position()) > dist):
                ref.append(idx)
                break
        for idx in range(view_id - 1, self.min_view_id, -1):
            if idx not in self.views:
                continue
            if (np.linalg.norm(pos - self.views[idx].Position()) > dist):   #向钱面的帧视图搜索，找到第一个与本视图相机距离大于前后DSO关键帧相机距离的帧
                ref.append(idx)
                break
        return ref

    def GetImage(self, view_id):
        return self.views[view_id].GetImage(self.image_folder)
    
    def GetSparseDepthMap(self, frame_id):
        camera = self.cameras[self.views[frame_id].camera_id]
        view = self.views[frame_id]
        view_pos = view.Position()
        depth_map = np.zeros((camera.height, camera.width), dtype=np.float32)
        for point_id, coord in view.points2d.items():
            pos3d = self.points3d[point_id].position3d
            depth = np.linalg.norm(pos3d - view_pos)
            depth_map[int(coord[1]), int(coord[0])] = depth
        return depth_map
    
    def Print(self):
        print("Found " + str(len(self.views)) + " cameras.")
        for id in self.cameras:
            self.cameras[id].Print()
        print("Found " + str(len(self.views)) + " frames.")
        for id in self.views:
            self.views[id].Print()

class Point:
    def __init__(self):
        self.id = -1
        self.position3d = np.zeros(3, float)
            
class Camera:

    def __init__(self):
        self.id = -1
        self.width = 0
        self.height = 0
        self.focal = np.zeros(2,float)
        self.principal = np.zeros(2,float)
        self.model = ""
    
    def Print(self):
        print("Camera " + str(self.id))
        print("-Image size: (" + str(self.width) +             ", " + str(self.height) + ")")
        print("-Focal: " + str(self.focal))
        print("-Model: " + self.model)
        print("")

class View:    
    def __init__(self):
        self.id = -1
        self.orientation = Quaternion()
        self.translation = np.zeros(3, float)
        self.points2d = {}
        self.camera_id = -1
        self.name = ""
    
    def IsKeyframe(self):
        return len(self.points2d) > 0
    
    def Rotation(self):
        return self.orientation.rotation_matrix
    
    def Position(self):
        return self.orientation.rotate(self.translation)
    
    def GetImage(self, image_folder):
        mat = cv2.imread(image_folder + "/" + self.name)
        # Check that we loaded correctly.
        assert mat is not None,             "Image " + self.name + " was not found in "             + image_folder
        return mat
    
    def Print(self):
        print("Frame " + str(self.id) + ": " + self.name)
        print("Rotation: \n" +             str(self.Rotation()))
        print("Position: \n" +             str(self.Position()))
        print("")
        
def ReadColmapCamera(filename):     #获取COLMAP文件cameras.txt内相机信息，初始化Reconstruction.cameras，本例中仅使用1个相机
    file = open(filename, "r")
    line = file.readline()
    cameras = {}
    while (line):
        if (line[0] != '#'):
            tokens = line.split()
            id_value = int(tokens[0])
            cameras[id_value] = Camera()
            cameras[id_value].id = id_value
            cameras[id_value].model = tokens[1]
            # Currently we're assuming that the camera model
            # is in the SIMPLE_RADIAL format
            assert(cameras[id_value].model == "PINHOLE")
            cameras[id_value].width = int(tokens[2])
            cameras[id_value].height = int(tokens[3])
            cameras[id_value].focal[0] = float(tokens[4])
            cameras[id_value].focal[1] = float(tokens[5])
            cameras[id_value].principal[0] = float(tokens[6])
            cameras[id_value].principal[1] = float(tokens[7])
        line = file.readline()
    return cameras;

def ReadColmapImages(filename):     #用images.txt初始化Reconstruction的views数据（）
    file = open(filename, "r")
    line = file.readline()
    views = {}
    while (line):
        if (line[0] != '#'):
            tokens = line.split()
            id_value = int(tokens[0])
            views[id_value] = View()
            views[id_value].id = id_value   #ID
            views[id_value].orientation = Quaternion(float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4]))    #相机姿态四元数
            views[id_value].translation[0] = float(tokens[5])   #相机位置
            views[id_value].translation[1] = float(tokens[6])   #相机位置
            views[id_value].translation[2] = float(tokens[7])   #相机位置
            views[id_value].camera_id = int(tokens[8])          #所属相机ID
            views[id_value].name = tokens[9]
            line = file.readline()
            tokens = line.split()
            views[id_value].points2d = {}
            for idx in range(0, len(tokens) // 3):      #开始录入SLAM算法点数据，点格式为（x,y,id）相同不同视图中具有相同ID的点为同一空间点，这是SLAM算法的核心数据之一
                point_id = int(tokens[idx * 3 + 2])
                coord = np.array([float(tokens[idx * 3 + 0]),                          float(tokens[idx * 3 + 1])])
                views[id_value].points2d[point_id] = coord
            
            # Read the observations...
        line = file.readline()
    return views
           
def ReadColmapPoints(filename):     #用points3D.txt初始化各id点的三维坐标信息
    file = open(filename, "r")
    line = file.readline()
    points = {}
    while (line):
        if (line[0] != '#'):
            tokens = line.split()
            id_value = int(tokens[0])
            points[id_value] = Point()
            points[id_value].id = id_value
            points[id_value].position3d = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            
        line = file.readline()
    return points

def ReadColmap(poses_folder, images_folder):
    # Read the cameras (intrinsics)
    recon = Reconstruction()
    recon.image_folder = images_folder
    recon.cameras = ReadColmapCamera(poses_folder + "/cameras.txt")     #初始化相机数据
    recon.views = ReadColmapImages(poses_folder + "/images.txt")        #录入SLAM帧数据
    recon.points3d = ReadColmapPoints(poses_folder + "/points3D.txt")   #录入各ID点的三维空间坐标（因为用的是绝对空间坐标，所以不存在同一ID的点坐标不同的情况）
    recon.min_view_id = min(list(recon.views.keys()))                   #录入ID最小的SLAM帧
    recon.max_view_id = max(list(recon.views.keys()))                   #录入ID最大的SLAM帧
    #assert len(recon.views) == (recon.max_view_id - recon.min_view_id) + 1, "Min\max: " + str(recon.max_view_id) + " " + str(recon.min_view_id)
    return recon

#稠密化代码
import flow_color

dis = DISOpticalFlow.create(2)
def GetFlow(image1, image2):
    flow = np.zeros((image1.shape[0], image1.shape[1], 2), np.float32)
    flow = dis.calc(        cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY),        cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY), flow)
    return flow

def AbsoluteMaximum(images):
    assert(len(images) > 0)
    output = images[0]
    for i in range(1,len(images)):
        output[np.abs(images[i]) > np.abs(output)] = images[i][np.abs(images[i]) > np.abs(output)]
    return output

#获取x/y方向的光流梯度强度
def GetImageGradient(image):
    xr,xg,xb = cv2.split(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5))   #x方向边缘检测
    yr,yg,yb = cv2.split(cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5))   #y方向边缘检测
    img_grad_x = AbsoluteMaximum([xr,xg,xb])
    img_grad_y = AbsoluteMaximum([yr,yg,yb])
    
    return img_grad_x, img_grad_y

#获取图像梯度强度
def GetGradientMagnitude(img_grad_x, img_grad_y):
    img_grad_magnitude = cv2.sqrt((img_grad_x * img_grad_x) + (img_grad_y * img_grad_y))
    return img_grad_magnitude

#获取光流梯度强度与置信值
def GetFlowGradientMagnitude(flow, img_grad_x, img_grad_y):
    x1,x2 = cv2.split(cv2.Sobel(flow,cv2.CV_64F,1,0,ksize=5))
    y1,y2 = cv2.split(cv2.Sobel(flow,cv2.CV_64F,0,1,ksize=5))
    flow_grad_x = AbsoluteMaximum([x1,x2])
    flow_grad_y = AbsoluteMaximum([y1,y2])
    flow_gradient_magnitude = cv2.sqrt((flow_grad_x * flow_grad_x) + (flow_grad_y * flow_grad_y))
    reliability = np.zeros((flow.shape[0], flow.shape[1]))

    #为每一个点赋予置信值
    for x in range(0, flow.shape[0]):
        for y in range(1, flow.shape[1]):
            #magn = (img_grad_x[x,y] * img_grad_x[x,y]) + (img_grad_y[x,y] * img_grad_y[x,y])
            gradient_dir = np.array((img_grad_y[x,y], img_grad_x[x,y]))         #梯度方向
            if (np.linalg.norm(gradient_dir) == 0):                             #梯度方向信息缺失处理
                reliability[x,y] = 0
                continue
            gradient_dir = gradient_dir / np.linalg.norm(gradient_dir)          #单位化梯度方向
            center_pixel = np.array((x,y))                                      #当前处理的点坐标
            p0 = center_pixel + gradient_dir
            p1 = center_pixel - gradient_dir
            if p0[0] < 0 or p1[0] < 0 or p0[1] < 0 or p1[1] < 0 or p0[0] >= flow.shape[0] or p0[1] >= flow.shape[1] or p1[0] >= flow.shape[0] or p1[1] >= flow.shape[1]:
                #若p0/p1其中一点不在图像坐标范围内
                reliability[x,y] = -1000
                continue
            f0 = flow[int(p0[0]), int(p0[1])].dot(gradient_dir)
            f1 = flow[int(p1[0]), int(p1[1])].dot(gradient_dir)
            reliability[x,y] = f1 - f0

    return flow_gradient_magnitude, reliability

#获取深度轮廓滤镜
def GetSoftEdges(image, flows):
    img_grad_x, img_grad_y = GetImageGradient(image)
    img_grad_magnitude = GetGradientMagnitude(img_grad_x, img_grad_y)       #使用SOBEL算子获取原图像梯度（用于混合中的置信值计算）
    if (dump_debug_images):
        plt.imsave(output_folder + "/image_gradient_" + recon.views[frame].name, img_grad_magnitude)
    flow_gradient_magnitude = np.zeros(img_grad_magnitude.shape)
    
    max_reliability = np.zeros(flow_gradient_magnitude.shape)
    i = 0
    for flow in flows:
        magnitude, reliability = GetFlowGradientMagnitude(flow, img_grad_x, img_grad_y)     #获取某向光流场梯度强度及其各点置信值
        if (dump_debug_images):
            plt.imsave(output_folder + "/flow_" + str(i) + "_" + recon.views[frame].name, flow_color.computeImg(flow))
            plt.imsave(output_folder + "/reliability_" + str(i) + "_" + recon.views[frame].name,                     reliability)
        flow_gradient_magnitude[reliability > max_reliability] = magnitude[reliability > max_reliability]   #通过置信值混合前后向光流场
        i += 1
        
    if (dump_debug_images):
        plt.imsave(output_folder + "/flow_gradient_" + recon.views[frame].name, flow_gradient_magnitude)
    flow_gradient_magnitude = cv2.GaussianBlur(flow_gradient_magnitude,(k_F, k_F),0)      #将深度轮廓滤镜进行高斯模糊
    flow_gradient_magnitude *= img_grad_magnitude
    flow_gradient_magnitude /= flow_gradient_magnitude.max()                #将深度轮廓滤镜信息单位标准化
    return flow_gradient_magnitude
    
def Canny(soft_edges, image):
    image = cv2.GaussianBlur(image, (k_I, k_I), 0)
    xr,xg,xb = cv2.split(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5))
    yr,yg,yb = cv2.split(cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5))
    img_gradient = cv2.merge((AbsoluteMaximum([xr,xg,xb]),AbsoluteMaximum([yr,yg,yb])))
    
    TG22 = 13573
    
    gx,gy = cv2.split(img_gradient * (2**15))
    mag = cv2.sqrt((gx * gx)                     + (gy * gy))
    seeds = []
    edges = np.zeros(mag.shape)
    for x in range(1, img_gradient.shape[0] - 1):
        for y in range(1, img_gradient.shape[1] - 1):
            ax = int(abs(gx[x,y]))
            ay = int(abs(gy[x,y])) << 15
            tg22x = ax * TG22
            m = mag[x,y]
            if (ay < tg22x):
                if (m > mag[x,y-1] and                   m >= mag[x,y+1]):
                    #suppressed[x,y] = m
                    if (m > tau_high and soft_edges[x,y] > tau_flow):
                        seeds.append((x,y))
                        edges[x,y] = 255
                    elif (m > tau_low):
                        edges[x,y] = 1
            else:
                tg67x = tg22x + (ax << 16)
                if (ay > tg67x):
                    if (m > mag[x+1,y] and m >= mag[x-1,y]):
                        if (m > tau_high and soft_edges[x,y] > tau_flow):
                            seeds.append((x,y))
                            edges[x,y] = 255
                        elif (m > tau_low):
                            edges[x,y] = 1
                else:
                    if (int(gx[x,y]) ^ int(gy[x,y]) < 0):
                        if (m > mag[x-1,y+1] and m >= mag[x+1,y-1]):
                            if (m > tau_high and soft_edges[x,y] > tau_flow):
                                seeds.append((x,y))
                                edges[x,y] = 255
                            elif (m > tau_low):
                                edges[x,y] = 1
                    else:
                        if (m > mag[x-1,y-1] and m > mag[x+1,y+1]):
                            if (m > tau_high and soft_edges[x,y] > tau_flow):
                                seeds.append((x,y))
                                edges[x,y] = 255
                            elif (m > tau_low):
                                edges[x,y] = 1
    w = img_gradient.shape[0] - 1
    h = img_gradient.shape[1] - 1
    if (dump_debug_images):
        plt.imsave(output_folder + "/edge_seeds_" + recon.views[frame].name,             edges == 255)
        plt.imsave(output_folder + "/edge_all_possible_" + recon.views[frame].name,             edges == 1)
    while len(seeds) > 0:
        (x,y) = seeds.pop()
        
        if (x < w and y < h and edges[x+1,y+1] == 1):
            edges[x+1,y+1] = 255
            seeds.append((x+1,y+1))
        if (x > 0 and y < h and edges[x-1,y+1] == 1):
            edges[x-1,y+1] = 255
            seeds.append((x-1,y+1))
        if (y < h and edges[x,y+1] == 1):
            edges[x,y+1] = 255
            seeds.append((x,y+1))
        if (x < w and y > 0 and edges[x+1,y-1] == 1):
            edges[x+1,y-1] = 255
            seeds.append((x+1,y-1))
        if (x > 0 and y > 0 and edges[x-1,y-1] == 1):
            edges[x-1,y-1] = 255
            seeds.append((x-1,y-1))
        if (y > 0 and edges[x,y-1] == 1):
            edges[x,y-1] = 255
            seeds.append((x,y-1))
        if (x < w and edges[x+1,y] == 1):
            edges[x+1,y] = 255
            seeds.append((x+1,y))
        if (x > 0 and edges[x-1,y] == 1):
            edges[x-1,y] = 255
            seeds.append((x-1,y))
    edges[edges == 1] = 0
    return edges

#初始化粗稠密信息（初始解，用来后续的优化）
def GetInitialization(sparse_points, last_depth_map):
    initialization = sparse_points.copy()   #关键字：倒数
    if last_depth_map.size > 0:
        #initialization矩阵中那些last_depth_map数值大于0的索引对应数据替换成1.0 / last_depth_map矩阵中那些last_depth_map数值大于0的索引对应数据
        initialization[last_depth_map > 0] = 1.0 / last_depth_map[last_depth_map > 0]
    
    w = int(densify_h)
    h = int(densify_w)
    last_known = -1
    first_known = -1
    for col in range(0,w):
        for row in range(0,h):
            if (sparse_points[col, row] > 0):
                last_known = 1.0 / sparse_points[col, row]
            elif (initialization[col, row] > 0):
                last_known = initialization[col, row]
            if (first_known < 0):
                first_known = last_known
            initialization[col, row] = last_known
    initialization[initialization < 0] = first_known
    
    return initialization
    
#深度辐射函数(稀疏深度信息矩阵sparse_points、深度轮廓hard_edges、深度轮廓滤镜soft_edges、上一帧的稠密深度信息)
def DensifyFrame(sparse_points, hard_edges, soft_edges, last_depth_map):
    w = sparse_points.shape[0]
    h = sparse_points.shape[1]
    num_pixels = w * h
    A = scipy.sparse.dok_matrix((num_pixels * 3, num_pixels), dtype=np.float32)
    A[A > 0] = 0
    A[A < 0] = 0
    b = np.zeros(num_pixels * 3, dtype=np.float32)
    x0 = np.zeros(num_pixels, dtype=np.float32)
    num_entries = 0
    #提取深度轮廓反滤镜（值越高代表平滑度越高）
    smoothness = np.maximum(1 - soft_edges, 0)
    smoothness_x = np.zeros((w,h), dtype=np.float32)
    smoothness_y = np.zeros((w,h), dtype=np.float32)
    initialization = GetInitialization(sparse_points, last_depth_map)
                             
    if (dump_debug_images):
        plt.imsave(output_folder + "/solver_initialization_" + recon.views[frame].name,                 initialization)
        plt.imsave(output_folder + "/sparse_points_" + recon.views[frame].name,                 sparse_points)
        plt.imsave(output_folder + "/soft_edges_" + recon.views[frame].name,                 soft_edges)
        plt.imsave(output_folder + "/hard_edges_" + recon.views[frame].name,                 hard_edges)

    #图像最边缘部分不计算
    for row in range(1,h - 1):
        for col in range(1,w - 1):
            x0[col + row * w] = initialization[col, row]
            # 深度约束
            if (sparse_points[col, row] > 0.00):
                A[num_entries, col + row * w] = lambda_d
                b[num_entries] = (1.0 / sparse_points[col, row]) * lambda_d
                num_entries += 1
            elif (last_depth_map.size > 0 and last_depth_map[col, row] > 0):
                A[num_entries, col + row * w] = lambda_t
                b[num_entries] = (1.0 / last_depth_map[col, row]) * lambda_t
                num_entries += 1
    
            # 平滑约束
            smoothness_weight = lambda_s * min(smoothness[col, row], smoothness[col - 1, row])
            if (hard_edges[col, row] == hard_edges[col - 1, row]):
                smoothness_x[col,row] = smoothness_weight
                A[num_entries, (col - 1) + row * w] = smoothness_weight
                A[num_entries, col + row * w] = -smoothness_weight
                b[num_entries] = 0
                num_entries += 1
            
            smoothness_weight = lambda_s * min(smoothness[col,row], smoothness[col, row - 1])
            if (hard_edges[col,row] == hard_edges[col, row - 1]):
                smoothness_y[col,row] = smoothness_weight
                A[num_entries, col + (row - 1) * w] = smoothness_weight
                A[num_entries, col + row * w] = -smoothness_weight
                b[num_entries] = 0
                num_entries += 1
    
    
    # Solve the system
    if (dump_debug_images):
        plt.imsave(output_folder + "/solver_smoothness_x_" + recon.views[frame].name,                 smoothness_x)
        plt.imsave(output_folder + "/solver_smoothness_y_" + recon.views[frame].name,                 smoothness_y)

    [x,info] = scipy.sparse.linalg.cg(A.transpose() * A, A.transpose() * b, x0, 1e-05, num_solver_iterations)
    depth = np.zeros(sparse_points.shape, dtype=np.float32)

    # Copy back the pixels
    for row in range(0,h):
        for col in range(0,w):
            if x[col + row * w] == 0:
                depth[col, row] = 0
            else:
                depth[col,row] = 1.0 / x[col + row * w]


    return depth

def TemporalMedian(depth_maps):
    lists = {}
    depth_map = depth_maps[0].copy()            #获取深度图列表第一个成员
    h = depth_map.shape[0]                      #获取深度图高
    w = depth_map.shape[1]                      #获取深度图宽
    for row in range(0,h):                      #依次获取深度图的每一行内容
        for col in range(0,w):                  #依次获取深度图某行的每一元素
            values = []
            for img in depth_maps:
                if (img[row,col] > 0):
                    values.append(img[row, col])
            if len(values) > 0:
                depth_map[row,col] = np.median(np.array(values))    #计算平均值
            else:
                depth_map[row,col] = 0
    return depth_map

def dco_entry(_cam_model, _scaling, _fx, _fy, _cx, _cy, _k1=0.0, _k2=0.0, _k3=0.0, _k4=0.0, obj_depth = 0):
    #原始数据获取与存储阶段=================================================================================================
    cam_model = _cam_model
    scaling_factor = _scaling
    fx = _fx
    fy = _fy
    cx = _cx
    cy = _cy
    k1 = _k1
    k2 = _k2
    k3 = _k3
    k4 = _k4
    global densify_h
    global densify_w
    ori_cap = cv2.VideoCapture(input_video + input_video_name)
    if not ori_cap.isOpened():
        print("error: 视频文件加载失败，请重新上传!")
        exit(0)
    ori_width = ori_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ori_high = ori_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    ori_fps = ori_cap.get(cv2.CAP_PROP_FPS)
    densify_h = int(ori_high // scaling_factor)
    densify_w = int(ori_width // scaling_factor)
    ori_total_frame = ori_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    with open(input_video_params_name, 'w') as ori_video_params:
        str_params = str(ori_width) + " " + str(ori_high) + " " + str(ori_fps)
        ori_video_params.write(str_params)
    with open(ori_params_folder + input_video_cali_params_name, 'w') as ori_cali_params:
        str_contain = ""
        if cam_model == 1:
            str_contain += "Pinhole " + str(fx) + " " + str(fy) + " " + str(cx) + " " + str(cy) + " 0\n" + \
                           ("%d" % (ori_width)) + " " + ("%d" % (ori_high)) + "\n" + \
                           "crop\n" + \
                           ("%d" % (ori_width // scaling_factor)) + " " + ("%d" % (ori_high // scaling_factor))
            ori_cali_params.write(str_contain)
        elif cam_model == 2:
            str_contain += "EquiDistant " + str(fx) + " " + str(fy) + " " + str(cx) + " " + str(cy) + " " + \
                           str(k1) + " " + str(k2) + " " + str(k3) + " " + str(k4) + "\n" + \
                           ("%d" % (ori_width)) + " " + ("%d" % (ori_high)) + "\n" + \
                           "crop\n" + \
                           ("%d" % (ori_width // scaling_factor)) + " " + ("%d" % (ori_high // scaling_factor))
            ori_cali_params.write(str_contain)
        else:
            print("error: 相机模型类型读取错误")
            clearProject()
            exit(0)
    print("note: 相机参数载入完毕。。。")
    # 帧拆分阶段===========================================================================================================

    rat, ori_frame = ori_cap.read()
    ori_frame_count = 0
    while rat:
        rat, ori_frame = ori_cap.read()
        if not rat:
            break
        ori_frame_name = "%07d.png" % ori_frame_count
        cv2.imwrite(ori_frame_folder + ori_frame_name,ori_frame)
        ori_frame_count += 1
    print("note: 帧分析完毕。。。")
    # 稀疏深度C++调用阶段===================================================================================================
    print("note: 开始计算稀疏深度")
    GetSparseDepth.GenSparseFn()
    print("note: 稀疏深度计算完毕")
    # 稠密化阶段===========================================================================================================
    recon = ReadColmap(input_colmap, input_frames)

    time_start = 0.0
    time_end = 0.0

    last_depths = []            #深度信息窗口，用于中值滤波，大小维持于K_T
    last_depth = np.array([])   #记录前一个深度运算结果，用于下一帧二次优化用
    depth_frames = []
    depth_images = []
    mix_frames = []

    # 利用前两个关键帧初始化本阶段算法
    # 用于初始化的这些关键帧将不会保存对应深度图
    skip_frames = recon.GetNeighboringKeyframes(recon.GetNeighboringKeyframes(recon.ViewIds()[30])[0])[1]

    print("note: 利用前 " + str(skip_frames) + " 帧进行初始化 (这些帧不会保存).")

    for frame in recon.ViewIds():
        reference_frames = recon.GetReferenceFrames(frame)      #获取前后参照帧
        if (len(reference_frames) == 0):
            print("note: 跳过第 %d 帧[%.2f]" % ((frame + 1),(frame / ori_total_frame)))
            continue
        print("note: 正在处理第 %d 帧 " % (frame + 1))
        base_img = recon.GetImage(frame)  # 获取当前帧图像

        flows = []
        time_start = time.perf_counter()
        for ref in reference_frames:                            #计算前后光流场
            ref_img = recon.GetImage(ref)
            flows.append(GetFlow(base_img, ref_img))
        time_end = time.perf_counter()
        time_start = time.perf_counter()
        soft_edges = GetSoftEdges(base_img, flows)              #获取软边缘数据（深度轮廓滤镜）
        time_end = time.perf_counter()
        time_start = time.perf_counter()
        edges = Canny(soft_edges, base_img)                     #结合Canny计算深度轮过边缘
        time_end = time.perf_counter()

        last_keyframe = frame                                   #记录当前帧
        if not recon.views[frame].IsKeyframe():                 #若当前帧不是关键帧，则将当前帧的前一关键帧作为last_keyframe
            neighboring_keyframes = recon.GetNeighboringKeyframes(frame)
            assert(len(neighboring_keyframes) > 1)
            last_keyframe = neighboring_keyframes[0]

        time_start = time.perf_counter()
        depth = DensifyFrame(recon.GetSparseDepthMap(last_keyframe), edges, soft_edges, last_depth)     #稠密化（二次优化问题）(存在问题,因为存在INF点，所以保存图片时将其余值压缩成0了，该问题已解决)
        time_end = time.perf_counter()

        last_depths.append(depth)
        if (len(last_depths) > k_T):        #将时间轴向的中值滤波窗口维持在K_T大小
            last_depths.pop(0)
        time_start = time.perf_counter()
        filtered_depth = TemporalMedian(last_depths)        #在时间轴向上进行均值滤波
        time_end = time.perf_counter()
        #跳过前若干帧图像，确保二次优化后的深度信息完全覆盖整个图片
        if (frame >= skip_frames):
            #plt.imsave(output_folder + "/" + recon.views[frame].name, filtered_depth)
            mix_frames.append(base_img)
            depth_img = filtered_depth.copy()
            cv2.normalize(filtered_depth,depth_img,255,0,cv2.NORM_MINMAX)
            for row in range(0,densify_h):
                 for col in range(0,densify_w):
                     depth_img[row][col] = 255 - depth_img[row][col]
            depth_images.append(depth_img)
            #cv2.imwrite(output_folder + "/" + recon.views[frame].name, depth_img)
            plt.imsave(output_folder + "/" + recon.views[frame].name, depth_img)
            depth_frames.append(filtered_depth)
        last_depth = depth
        print("note: 第 %d 帧处理完毕[%.2f]" %((frame + 1),frame/ori_total_frame))
    # #开始渲染和组装视频====================================================================================================
    #
    print("note: 深度稠密化完毕，开始生成视频")
    output_size = (densify_w,densify_h)
    f_path = "./output_frames/depthframes/"
    name_list = os.listdir(f_path)
    vwriter = cv2.VideoWriter(output_video + depth_video_name,cv2.VideoWriter_fourcc('D','I','V','X'),ori_fps,output_size)
    for file_name in name_list:
        curframe = cv2.imread(output_folder + file_name,cv2.IMREAD_COLOR)
        vwriter.write(curframe)
    vwriter.release()
    print("note: 稠密深度视频生成完毕")
    print("note: 开始渲染混合场景")
    virtual_obj = cv2.imread("./virtual_obj.png")
    virtual_obj = cv2.cvtColor(virtual_obj,cv2.COLOR_BGR2GRAY,virtual_obj)
    vir_w = virtual_obj.shape[1]
    vir_h = virtual_obj.shape[0]
    vir_ratio = vir_w / vir_h
    vir_aim_w = int(densify_w * 0.2)
    vir_aim_h = int(vir_aim_w / vir_ratio)
    virtual_obj = cv2.resize(virtual_obj,(vir_aim_w,vir_aim_h),interpolation=cv2.INTER_LINEAR)
    top = densify_h - vir_aim_h - 30
    bottom = densify_h - 30
    left = 30
    right = 30 + vir_aim_w

    vwriter.open(output_video + mix_video_name, cv2.VideoWriter_fourcc('M','J','P','G'),ori_fps,output_size)
    #=======================================
    # vw = cv2.VideoWriter(output_video + mix_video_name, cv2.VideoWriter_fourcc('M','J','P','G'),ori_fps,output_size)
    # densift_frames_path = "./output_frames/depthframes/"
    # ori_frames_path = "./sparse_data/frames/"
    # for index in range(42,192):
    #     depth_info = cv2.imread(densift_frames_path + ("%07d.png" % index),cv2.IMREAD_GRAYSCALE)
    #     ori_img = cv2.imread(ori_frames_path + ("%07d.png" % index),cv2.IMREAD_GRAYSCALE)
    #     for row in range(top, bottom + 1):
    #         for col in range(left, right + 1):
    #             if depth_info[row][col] > 255 * obj_depth:
    #                 ori_img[row][col] = virtual_obj[row - densify_h + 30 + vir_aim_h - 1][col - 30 - 1]
    #     ori_img = cv2.cvtColor(ori_img,cv2.COLOR_GRAY2BGR)
    #     vw.write(ori_img)
    # vw.release()
    # vwriter.release()
    #=======================================

    for index in range(0,len(mix_frames)):
        for row in range(top, bottom + 1):
            for col in range(left, right + 1):
                temp = cv2.normalize(depth_frames[index],0,1,cv2.NORM_MINMAX)
                if temp[row][col] > obj_depth:
                    mix_frames[index][row][col] = virtual_obj[row - densify_h + 30 + vir_aim_h-1][col - 30-1]
    count = 0

    for mix_frame in mix_frames:
        cv2.imwrite(output_folder_mixed + ("%07d.png" % count),mix_frame)
        mix_frame = cv2.cvtColor(mix_frame,cv2.COLOR_GRAY2BGR)
        vwriter.write(mix_frame)
        count = count + 1
    vwriter.release()
    print("note: 混合场景渲染完毕")
    print("======================SUCCESS======================")


if __name__  == '__main__': #针孔模型-1/鱼眼相机-2  图像缩小倍数  fx  fy  cx  cy  若选择鱼眼相机则填写(k1 k2 k3 k4)否则该部分参数无需输入
    dco_entry(1,1,800.0,800.0, 640.0, 360.0, 0.5801358893633602,\
              0.1488700681725693, -0.5065295812560362, 0.2129197100544011, 0.73)

