# covid_project
Pipeline which outputs statistics to estimate risks of covid spreading within the camera field of view

## packages

pip install opencv-python
pip install cameratransform

## files directory tree
```bash
|-- video_scraping
|   |-- serbia
|   |   |-- serbia_1592484664941.jpg
|   |   |-- serbia_1592484664942.jpg
|   |   |-- ...
|   |
|   |-- himmelried
|   |   |-- himmelried_1592484664941.jpg
|   |   |-- himmelried_1592484664942.jpg
|   |   |-- ...
|
|-- camera_params
|   |-- serbia_camera_params.json
|   |-- himmelried_camera_params.json
|
|-- models
|   |-- yolo_coco
|   |   |-- coco.names
|   |   |-- yolov3.cfg
|   |   |-- yolov3.weights
|
|   |-- overweight
|   |   |-- overweight_detection_model.pickle
|
|   |-- age
|   |   |-- vgg_face_weights.h5
|   |   |-- classification_age_model_v2.hdf5
|   |   |-- age_model_weights.h5
```
