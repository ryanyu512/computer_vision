# visual_odometry

mono (Straight)            |  stereo (Straight)
:-------------------------:|:-------------------------:
![image]![b1324992-6cd6-4f53-bbb7-8f7451f468de](https://github.com/ryanyu512/computer_vision/assets/19774686/f6607237-4573-443b-b1f7-55135fecd4b6) | ![22ba5ec0-f383-4b1e-9dd6-1852b1b91239](https://github.com/ryanyu512/computer_vision/assets/19774686/c9536dcd-ee0b-4b2d-be9a-dfcbbaa2c0b7)
mono (Curve 1)            |  stereo (Curve 1)
![92a6dd81-d80c-409d-8328-b5feea4809cd](https://github.com/ryanyu512/computer_vision/assets/19774686/6c5a1ab6-5f85-4089-8b90-f3f1413fa87f) | ![049932d4-5d90-47b3-bc3d-1bfd37134a90](https://github.com/ryanyu512/computer_vision/assets/19774686/89771993-3f01-4ac2-8b4f-b25f6fb9f51c)
mono (Curve 2)            |  stereo (Curve 2)
![5633312b-9aa7-470a-b807-746626a4957e](https://github.com/ryanyu512/computer_vision/assets/19774686/6b3e7391-1d66-4c4e-81bf-7df954580cbe)|![1165e8ef-c5c9-4f4b-9172-e872938df200](https://github.com/ryanyu512/computer_vision/assets/19774686/07e3f5a6-ac52-4feb-a80d-858bb5532467)


A very small set of KITTI data is used for proof of concept. The above is the result of mono visual odometry vs stereo visual odometry. 

This mini project study stereo visual odometry vs mono visual odometry. And, it shows that the performance of stereo visual odometry is more stable than mono visual odometry at the expense of more expensive computation resources. Also, we could note that certain drift errors are observed. Therefore, I would study the complete pipeline of SLAM to see if the drift problem could be resolved.  

1. mono_visual_odometry.py/stereo_visual_odometry.py: define visual odometry method
2. demo.ipynb/stereo_demo.ipynb: used for demo
3. KITTI_sequence_1/2/3: evaluation data for proof of concept

Credit: Most ideas are originated from this repository: https://github.com/niconielsen32/ComputerVision/tree/master/VisualOdometry. For learning purposes, I read, understand, re-implement and improve the original codes a bit. Also, one more curve data is added to further verify the stability of stereo visual odometry
