# Grasp Everything With Anything

## Installation
cd GEWA
```[bash]
conda create -n GEWA python=3.11
pip install -r requirements.txt
pip install dist/GEWA-1.0-py3-none-any.whl
```

## Dataset
-Download the Acronym and ShapeNet datasets and unzip to the data folder
https://sites.google.com/view/graspdataset
https://huggingface.co/datasets/ShapeNet/ShapeNetSem-archive

-Clone the Manifold repository to the root directory https://github.com/hjwdzh/Manifold and follow the instruction in the repo to build the Manifold package

-Make the meshes watertight using Manifold with the following command:
```
python dataset/preprocess_shapenet.py
```
-After the preprocessing, the data folder should look like this:
data/
├── splits
├── acronym
├── ShapeNetSem-backup
├── simplified_obj

The split folder contains the training and testing splits used in the ContactNet paper.

Run the following command to generate the Touch Point Pair (TPP) or Approach dataset:
```
python dataset/create_tpp_dataset.py
```
```
python dataset/create_approach_dataset.py
```

## Inference

## Training
