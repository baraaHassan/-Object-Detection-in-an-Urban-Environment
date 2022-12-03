# Object Detection in an Urban Environment

## Project overview
In this project, the task needed is to create a model that can detect cars, cyclists and pedsterians for self-driving cars system.

## Set up

### Data

For this project, data from the [Waymo Open dataset](https://waymo.com/open/) will be used.

- The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records.

### Structure

#### Data

The data you will use for training, validation and testing is organized as follow:
```
/home/workspace/data/waymo
    - training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test - contains 3 files to test your model and create inference videos
```

The `training_and_validation` folder contains file that have been downsampled: one every 10 frames is selected from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling.

You will split this `training_and_validation` data into `train`, and `val` sets by completing and executing the `create_splits.py` file.

#### Experiments
The experiments folder will be organized as follow:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment0/ - create a new folder for each experiment you run
    - experiment1/ - create a new folder for each experiment you run
    - experiment2/ - create a new folder for each experiment you run
    - label_map.pbtxt
    ...
```

### Prerequisites

#### Local Setup

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.

#### Download and process the data

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file.

You can run the script using the following command:
```
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

You are downloading 100 files (unless you changed the `size` parameter) so be patient! Once the script is done, you can look inside your `data_dir` folder to see if the files have been downloaded and processed correctly.

### Instructions

#### Exploratory Data Analysis

You should use the data already present in `/home/workspace/data/waymo` directory to explore the dataset! This is the most important task of any machine learning project. To do so, open the `Exploratory Data Analysis` notebook. In this notebook, your first task will be to implement a `display_instances` function to display images and annotations using `matplotlib`.

#### Create the training - validation splits
You will have to create your own training and validation sets using the files located in `/home/workspace/data/waymo`. The `split` function in the `create_splits.py` file does the following:
* create three subfolders: `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`
* split the tf records files between these three folders by symbolically linking the files from `/home/workspace/data/waymo/` to `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`

Use the following command to run the script once your function is implemented:
```
python create_splits.py --data-dir /home/workspace/data
```

#### Edit the config file

Now you are ready for training. The Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

#### Training

You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder. Now launch the training process:
* a training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`. You will report your findings in the writeup.

#### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it.


#### Creating an animation
##### Export the trained model
Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```

This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /data/waymo/testing/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```


### Dataset
#### Dataset analysis
The code for creating data analysis can be found in Exploratory Data Analysis.ipynb file.
##### Qualitative analysis
By visualizing random images, to see the diversity of the dataset and how it looks like. The following are random samples of the dataset.
![download (17)](https://user-images.githubusercontent.com/49837627/205456662-770590be-c9c6-46bc-bb05-ff150bde32b0.png)
![download (18)](https://user-images.githubusercontent.com/49837627/205456668-41625460-aa54-428f-bb83-8c0874fe3a20.png)
##### Quantitive analysis
The quantitive analysis shows that the dataset contains more vehicles than pedestrians and cyclists. The following is a chart that shows the distribution for a sample of the dataset.
![download (19)](https://user-images.githubusercontent.com/49837627/205458524-4e7b1fea-fec9-4bde-9601-970ae86f4bbd.png)

#### Cross validation
The data is divided into train,validate and test with ratios 80:10:10.

### Training
#### Reference experiment
In the reference experiment, the model couldn't converge normally, because different factors, number of iterations wasn't enough, and the batch size was very small. The following settings, are the settings used for the reference experiment.

```
Model: SSD Resnet 50 640x640 
No. of epochs: 2500
batch size = 2
warmup steps: 200
Augmentation: - random_crop_image
              - random_horizontal_flip
```

It can be seen from the tensorboard charts the model couldn't converge.

![Screenshot from 2022-12-03 19-54-31](https://user-images.githubusercontent.com/49837627/205457386-6a118fff-318a-4cc2-b223-451fbb264ad7.png)
![Screenshot from 2022-12-03 19-56-18](https://user-images.githubusercontent.com/49837627/205457452-6864789d-d4fa-4367-9cbd-6fa137b75c42.png)
![Screenshot from 2022-12-03 19-56-34](https://user-images.githubusercontent.com/49837627/205457466-68db2e64-7f6b-4413-add2-d9c44bbcf694.png)

The following are samples of the evluation, that shows the model couldn't converge. The model output is on the left side, and the ground truth is on the right side.

![imageData (10)](https://user-images.githubusercontent.com/49837627/205458366-ea9975c9-e193-4912-bee7-2a71a3034a19.png)
![imageData (12)](https://user-images.githubusercontent.com/49837627/205458373-7d1a6b2a-8dc2-4f72-8a40-b897d7d2f9a8.png)
![imageData (13)](https://user-images.githubusercontent.com/49837627/205458377-f53f80ba-0e6c-4636-b1ca-948571a72035.png)


#### Improve on the reference
To improve the reference model, I increased the number of epochs, amount of batches, and added different augmentation options.
The following settings, are the settings used for the improvement experiment.

```
Model: SSD Resnet 50 640x640 
No. of epochs: 25000
batch size = 4
warmup steps: 2000
Augmentation: - random_crop_image
              - random_horizontal_flip
              - random_adjust_hue
              - random_adjust_contrast
              - random_adjust_saturation
              - random_adjust_brightness
              - random_square_crop_by_scale
              - random_horizontal_flip
              - random_crop_image
```
              
It can be seen from the tensorboard charts the model was able to converge.

![Screenshot from 2022-12-03 20-09-10](https://user-images.githubusercontent.com/49837627/205457829-1c2258a2-3cd6-403b-b130-192fc4fbf577.png)
![Screenshot from 2022-12-03 20-21-11](https://user-images.githubusercontent.com/49837627/205458172-a34f2bf6-6b24-4f32-8646-b8273c7da93c.png)
![Screenshot from 2022-12-03 20-09-23](https://user-images.githubusercontent.com/49837627/205457836-e4db51df-3e76-46b8-86db-685dc21de2c5.png)

The following are samples of the evluation, that shows the model was able to converge. The model output is on the left side, and the ground truth is on the right side.

![imageData](https://user-images.githubusercontent.com/49837627/205458434-9caa7963-164c-45f6-ac2d-5c872403b3bb.png)
![imageData (3)](https://user-images.githubusercontent.com/49837627/205458438-367dab25-a32d-4641-8e38-19885c777eb6.png)
![imageData (9)](https://user-images.githubusercontent.com/49837627/205458453-ed0a0ad1-ee90-49d0-90d9-2c1adad3269c.png)

Note: the animation of the different experiments output can be found in the folder animations.

Note: there are more experiments done, than the two detailed here in the readme file, you can find those experiments in the experiments folder (each experiment in its own folder).

=======
