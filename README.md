## Summary

This notebook details my solution to Kaggle's [Airbus Ship Detection challenge](https://www.kaggle.com/competitions/airbus-ship-detection/overview).
The aim of the competition is to analyze satellite images, detect the ships and produce segmentation masks of the ships.
We have original images (around 100 k of them) and corresponding masks for each occuring ship in .csv format. 

## Repository content

To download dataset you need to accept the competition conditions. Then, after installing the Kaggle CLI, run the following command inside it:

`kaggle competitions download -c airbus-ship-detection -o -w`.

The dataset has satellite images of ships in ocean or on docks as th input images. The output are given in **train_segmentations_v2.csv** and **sample_submission_v2.csv** for training and testing respectively. Those data are in the form of ImageId -> RLE Encoded Vector. The output of the images in data set are encoded using Run Length Encoding (RLE), the expected output for the problem is a RLE mask of ships with background as a two color image. Also the folders with images you can find in **train_v2** and **test_v2**.

Code with data analysis, decoding images, splitting data is in **model_training.py**, creating model, fitting data, creating loss function you can find in **model_inference.py**. Also **run.ipynb** contains the same code with some plots and results.

## Model

The solution is based on U-Net model and the goal of choosing U-Net model is also based on the data set we have. After analyzing the dataset and observing the positive samples (i.e. those samples which have at least one ship in the input image) in the dataset U-Net is one choice for solution. U-Net is a segmentation model which uses a strong data augmentation to use the available annotated samples more efficiently. It's architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization.
As for the model summary I use *activation = relu*, to preserve dimensions I use *padding = same* and then concatenate layers that are supposed to have the same dimensions. The *setting batch_size = 5* and *steps_per_epoch = 100*, *optimizer = adam*. 
There are a lot of built-in metrics, but default metrics are not always good idea. In this case we need to create a function for measuring quality of model. This metric will be callling *Intersection over Union*. It's arguments would present ground truth mask of a ship and predicted mask. When this mattching is perfect, metric value is 1 and the lower predicting precison is, the lower is this value (down to zero). 
As for loss function, we can represent it as a negative value of IoU metric. In this way we obtain function that decreases when our prediction is improving and increases otherwise.
So, I get the result *accuracy = 0.9983* and *loss = -0.6425*. It seems like overfitting, maybe it is related with insufficiently balanced data or with small number of epoch. I think I should try to create more layers and play with parameters.
