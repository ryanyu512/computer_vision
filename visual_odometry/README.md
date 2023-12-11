# visual_odometry

mono (Straight)            |  stereo (Straight)
:-------------------------:|:-------------------------:
![image](https://github.com/ryanyu512/visual_odometry/assets/19774686/75a5c96d-bddf-4ba6-a7a6-2a6e112dd91d) | ![image](https://github.com/ryanyu512/visual_odometry/assets/19774686/03c0fe3c-236d-4833-bb01-394dff76afd0)
mono (Curve 1)            |  stereo (Curve 1)
![image](https://github.com/ryanyu512/visual_odometry/assets/19774686/2165ef7d-e4ff-4b63-984f-6ede72fae18b)  |  ![image](https://github.com/ryanyu512/visual_odometry/assets/19774686/362a08aa-b27a-4885-a31e-003e889e8288)
mono (Curve 2)            |  stereo (Curve 2)
![image](https://github.com/ryanyu512/visual_odometry/assets/19774686/f591133e-0d91-4d13-a0d9-11cc6380e5df) | ![image](https://github.com/ryanyu512/visual_odometry/assets/19774686/4f49d1c9-0737-49de-90ec-0a9619277f88)


A very small set of KITTI data is used for proof of concept. The above is the result of mono visual odometry vs stereo visual odometry. 

This mini project study stereo visual odometry vs mono visual odometry. And, it shows that the performance of stereo visual odometry is more stable than mono visual odometry at the expense of more expensive computation resources. Also, we could note that certain drift errors are observed. Therefore, I would study the complete pipeline of SLAM to see if the drift problem could be resolved.  

1. mono_visual_odometry.py/stereo_visual_odometry.py: define visual odometry method
2. demo.ipynb/stereo_demo.ipynb: used for demo
3. KITTI_sequence_1/2/3: evaluation data for proof of concept

Credit: Most ideas are originated from this repository: https://github.com/niconielsen32/ComputerVision/tree/master/VisualOdometry. For learning purposes, I read, understand, re-implement and improve the original codes a bit. Also, one more curve data is added to further verify the stability of stereo visual odometry
