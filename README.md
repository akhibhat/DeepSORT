# Deep-SORT

In this work, we implement a popular object tracking algorithm, DeepSORT for tracking thermal images. We have modified the Github [repository](https://github.com/nwojke/deep_sort.git) to track thermal images from the FLIR dataset.


## Train

- Thermal object detections were trained in the following [repository](https://github.com/rodri651/thermal_object_detection)

## Data

In order to be able to test the repository, you will need to download some data first.

- To test the tracker on RGB images, you will need the [MOT16 benchmark](https://motchallenge.net/data/MOT16/) sequences. This code assumes that the MOT16 benchmark data is in `./MOT16`.
- You will also need `.npy` files which store the detections and their features. The [resources](https://drive.google.com/drive/folders/18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp?usp=sharing) folder needs to be extracted to the root directory of this repository.
- The thermal dataset can be found on their [website](https://www.flir.com/oem/adas/adas-dataset-form/). Make sure the images are stored in `./Thermal/img1/` in the root repository.
- Also download the `.npy` files for the Thermal dataset and store them in the `./Thermal/` folder.

## Run the code

Once you have the above files you can run the tracker with the following command:

```
python3 deep_sort_app.py \
	--sequence_dir=./Thermal/ \
	--detection_file=./Thermal/seq1.npy \
	--min_confidence=0.8 \
	--nn_budget=100 \
	--display=True
```

