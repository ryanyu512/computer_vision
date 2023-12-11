# cat and dog detection based on yolo v1 concept
![image](https://user-images.githubusercontent.com/19774686/235753814-847b08f2-b95d-4364-99eb-c6d4200b550a.png)

PROJECT BRIEF

  Based on concepts of YOLO v1(https://arxiv.org/abs/1506.02640), a deep learning model for cat and dog detection is trained and validated. Since the detection targets of this project are cat and dog, the architechure needs not to be as large as YOLO v1. But, the loss function and corresponding data augmentation mentioned in the paper are implemented in this project. 

  Data is downloaded from this [link](https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection). The data size is around 1GB, contains 3686 cat and dog images with bounding box labels. Since the data size is quite small, then the data is just splitted into 2 sets training and validation. The training/validation split ratio is 0.9/0.1 for each class. 
  
  Based on IOU_threshold@0.5, The average precision and recall of validation data are 0.92 and 0.88 respectively. Since the detection model size is too large, I choose not to upload to github. 

FILE BRIEF
1. explore_det_data.ipynb: aims at exploring how to extract data from label
2. explore_data_quality.ipynb: explore the data quality, such as any duplicated image or wrong label
3. det_network_explore.ipynb: explore the defined cat and dog network architechure
4. prepare_data.ipynb: split raw data into training and validation and explore data augmentation for training
5. img_example.ipynb: aims at demonstrating the ability of the trained model 
6. network.py: the defined network architechure
7. train_model.py: custom library for training model
8. train_network.py: define setting for training model
9. utilities.py: utiliy functions for training and testing 
10. macro.py: store less frequently changed parameters
11. visualise.py: for visualising bounding box on the images
12. prepare_data.py: custom library for preparing data
