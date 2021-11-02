# ACME

## Description
This repository contains the code and documentation to implement ACME (Automatic Cell Migration Examination):

```
ACME: Automatic feature extraction for Cell Migration Examination through intravital microscopy imaging,
Miguel Molina-Moreno, Iván González Díaz and Fernando Díaz de María
Medical Image Analysis, 2021 (CoRR)
```

```
Behavioral immune landscapes of inflammation,
Georgiana Crainiciuc, Miguel Palomino-Segura, Miguel Molina-Moreno, Jon Sicilia and others.
Nature, 2021 (accepted for publication)
```

This code is partly based on the implementations of [U-Net 3D](https://github.com/MIC-DKFZ/medicaldetectiontoolkit) and [PhagoSight](https://github.com/phagosight/phagosight).

## License

ACME code is released under the GNU GPLv3 License (refer to the `LICENSE` and `COPYING` files for details).

## Citing ACME

If you find ACME useful in your research, please consider citing:

	@ARTICLE{acme,
		author = {Miguel Molina-Moreno, Iv\'an Gonz\'alez D\'iaz and Fernando D\'iaz de Mar\'ia},
		title = {{ACME}: Automatic feature extraction for Cell Migration Examination through intravital microscopy imaging},
		journal = {Medical Image Analysis},
		year = {2021},
		volume={},
		number={},
		pages={},
		doi={},
		ISSN={}
	}
  
  	@ARTICLE{behavioral,
		author = {Georgiana Crainiciuc, Miguel Palomino Segura, Miguel Molina Moreno, Jon Sicilia and others},
		title = {Behavioral immune landscapes of inflammation},
		journal = {Nature},
		year = {2021},
		volume={},
		number={},
		pages={},
		doi={},
		ISSN={}
	}
  

## Requirements

ACME is implemented to not require any additional modules. MATLAB code has been developed with the R2017b version. The Python code has been tested with Pytorch 1.3.1, torchvision 0.4.2 and CUDA 10.1.

Before executing the Python code, it is necessary to compile the functions in `custom_extensions folder`, through the `setup_roi_align.py` script, with the `CUDA_HOME` environment variable pointing to your CUDA installation. 

## Demo

We describe the pipeline of execution of the code below. 

1. First, the `matlab/config.m` script contains different configurable parameters for the experiment: venule and cell channels, group denomination, voxel size, etc. and the different modules of the system.  
2. Once the parameters are set and the sequences are stored in the corresponding folder, the `matlab/extract_data.m` must be used to store each 3D temporal volume of the 4D sequence (in `.mat` format) in the directory `data/annotation`, to facilitate the annotation of the volumes (if you want to test how the segmentation and tracking process is working with your samples you can annotate your own volumes).
3. Now, the `matlab/generate_database.m` script can be used to generate the database folder with the volumes that will be used as inputs to the 3D Joint Segmentation Module.
4. The data are prepared for the inference process. This is done by the `python/detection_inference.py` function. The inference results are stored in the database folder.
5. After this step, we return to MATLAB implementation to process the ACME segmentations with the script `matlab/process_ACME_segmentations`, which accumulates the cell and blood vessel segmentations per capture and performs the 3D three-pass tracking. At this point of the pipeline you can print the images (see the `matlab/config.m` JPEG variable) in the `database/jpgs_results`.
6. The next step consists of extracting the instantaneous and dynamic features with the `extract_instantaneous_features.m` and the `extract_dynamic_features.m` scripts, respectively.
7. The pipeline of cell detection ends with the `matlab/cell_selection_module.m` script. From the extracted features, it applies the trained classifier to detect those cells that are well segmented (in terms of the fixed precision level), see `matlab/config.m` file.
8. Finally, the hierarchical explainability module (`hierarchical_explainability.m`) detects the behaviors, builds the hierarchy from them and offers the most relevant features for each partition of the hierarchy (there are configurable parameters in `config.m`). In addition, the `python/visualization.py` function is able to arrange the data in t-SNE or UMAP plots for a better understanding.

The reported performance of each module for our scenario is presented below:

| Component                     |   Precision (%)  |     Recall (%)  |    IoU (blood vessel) (%)  | 
|-------------------------------|------------------|-----------------|----------------------------|
| 3D Joint Segmentation module  |       67.13      |       78.46     |            88.09           |
| 3D three-pass Tracking module |       66.45      |       75.67     |            88.09           |
| Cell detection module         |       95.28      |       30.48     |            88.09           |


## Installation

To start using ACME, download and unzip this repository.
```
git clone https://github.com/miguel55/ACME
```

## More info

See `doc\doc.tex` for more details.
