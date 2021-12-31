# DCOAlgDemo

Depth contour occlusion depth map generation algorithm

**Input video, camera internal parameters and virtual pictures to be displayed to generate contour depth map video and corresponding virtual real occlusion composite video**

 

## Input

1.Video file [at least 5 seconds is required, and the resolution is recommended to be 720p (1280 x 720) or above, otherwise there is a chance that sparse depth generation will fail]

> Save video files in the directory: <b>/input/video/ori_video.mp4</b>

2.Virtual picture

> Unlimited picture size
> The image file is saved in the algorithm root directory <b>/virtual_obj.png</b>

3.Camera model

> The user selects one of the <b>pinhole camera model</b> and <b> fisheye camera model</b> according to the actual situation

4.Camera parameters

> Pinhole camera model: fx fy cx cy   
> Fisheye camera model: fx fy cx cy k1 k2 k3 k4  
> fx fy is the focal length parameter, and the unit is pixel. **if accurate Fx / Fy cannot be obtained**, it can be uniformly assigned as lens focal length, conversion: <b>F (pixel) = f (mm) / photosensitive unit size (mm)</b>  
> cx cy is the coordinate of the image principal point, <b>if accurate cx/ cy cannot be obtained,cx = video frame width / 2 and cy = video frame height / 2</b>  
> If the fisheye camera model is selected, the distortion correction parameters <b>k1, k2, k3 and k4must have</b>   

## Call

1.Environment dependency: opencv34+/ numpy/ pyquaterion/ matplotlib/ scipy  
2.Interface description:  
Prototype: <b>DCO_entry(cam_model, scaling, fx, fy, cx, cy, d=0.5, k1=0, k2=0, k3=0, k4=0)</b>  
> <b>[cam_model]</b> camera model, 1 indicates that the camera model is a pinhole camera model, and 2 indicates that the camera model is a fish eye camera model  
> <b>[scaling]</b>process scaling, and the final processed and output video frame size is <b> original video size / scaling</b>  
> <b>[fx /fy / cx / cy]</b> camera internal parameter data entered by the user    
> <b>[d]</b> the depth of the virtual image ranges from 0 to 1. The closer it is to 1, the greater the depth of the virtual image, and vice versa  
> <b>[k1 / k2 / k3 / k4]</b> if the user selects a pinhole camera model, these four parameters can be left blank. If the user selects a fisheye camera model, these four distortion parameters must be entered  

## Output

1.Depth video

> The video file is saved in the directory:**/output_video/depth.mp4** 

2.Composite video

> The video file is saved in the directory:**/output_video/mixed.mp4** 

## Project directory structure

```ASN.1
+── dcoalgdemo [project assembly root directory]
│ +── input_Video [store input video]
│ │ +── ori_video.XXX [original video stored]
│ +── sparse_Data [store data needed to generate sparse depth]
│ │ + -- frames [store sparse depth input video frames]
│ │ │ +── 0000000.png 
│ │ │ +── 0000001.png 
│ │ │ +── 0000002.png 
│ │ │ +── ... 
│ │ + -- Calibration [camera parameters required for storing sparse depth]
│ │ │ +── camera.txt 
│ +── densify_Data [home directory of all data required for densification]
│ │ + - frames [store corrected frame sequence]
│ │ │ +── 0000000.png 
│ │ │ +── 0000001.png 
│ │ │ +── 0000002.png 
│ │ │ +── ... 
│ │ + - reconstruction [store all parameters required for densification]
│ │ │ +── cameras.txt 
│ │ │ +── images.txt 
│ │ │ +── points3D.txt 
│ +── output_Frames [store result frames to be assembled into video]
│ │ + -- depthframes
│ │ │ +── depth_0000000.png 
│ │ │ +── depth_0000001.png 
│ │ │ +── depth_0000002.png 
│ │ │ +── ... 
│ │ + -- ocluframes [frame for storing occlusion effect]
│ │ │ +── oclu_0000000.png 
│ │ │ +── oclu_0000001.png 
│ │ │ +── oclu_0000002.png 
│ │ │ +── ... 
│ +── output_Video [store final output video]
│ │ +── depth_video.mp4 
│ │ +── oclu_video.mp4 
│ +── DCO_Demo.Py [main project]
│ +── flow_color.Py [visual correlation function]
│ +── videoParams.Txt [parameters related to assembly video, including target resolution and frame rate]
│ +── virtual_obj.Png [virtual picture to be mixed]
│ +── ...[necessary dynamic link library]
```

## Contact

This repository is maintained by Naye Ji (conniemy). Feel free to reach out directly at jinaye@cuz.edu.cn with any questions or comments. Thanks to contributors: Haoxiang Zhang
