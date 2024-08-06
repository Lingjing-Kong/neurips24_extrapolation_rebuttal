## The Instruction to Run TeSLA-Extrapolation

### Install Environment
Please install necessary packages by running `pip install -r requirements.txt`

### Download Data and Pre-trained Source Models
Please download the target datasets including CIFAR-10C, CIFAR-100C, and ImageNet-C. If you would like to use the source statistical information, please also download the source datasets CIFAR-10, CIFAR-100, and ImageNet. The Download links and extracted paths are listed below. Please download the Imagenet class labels information at [link of class labels](https://raw.githubusercontent.com/sayakpaul/robustness-vit/master/analysis/masking/imagenet_class_index.json) and put it here `../Datasets/ImageNet-C/imageNet_labels.json`. Please also download the pre-trained source models in the following links.

**Datasets Download Links**

|Dataset Name      	| Download Link                                                                                      	| Extract to Relative Path               	|
|-------------------	|----------------------------------------------------------------------------------------------------	|----------------------------------------	|
| CIFAR-10C         	| [click here](https://zenodo.org/record/2535967 )                                                   	| ../Datasets/cifar_dataset/CIFAR-10-C/  	|
| CIFAR-100C        	| [click here](https://zenodo.org/record/3555552)                                                    	| ../Datasets/cifar_dataset/CIFAR-100-C/ 	|
| ImageNet-C        	| [click here](https://zenodo.org/records/2235448)                                                    	| ../Datasets/ImageNet-C/         	|             	|
| CIFAR-10         	| [click here](https://www.cs.toronto.edu/~kriz/cifar.html )                                                   	| ../Datasets/cifar_dataset/cifar-10-batches-py/  	|
| CIFAR-100        	| [click here](https://www.cs.toronto.edu/~kriz/cifar.html)                                                    	| ../Datasets/cifar_dataset/cifar-100-python/ 	|
| ImageNet        	| [click here](https://www.kaggle.com/c/imagenet-object-localization-challenge/data?select=ILSVRC)                                                    	| ../Datasets/ImageNet/          	|             	|


**Pre-trained Source Models Links**

| Dataset Name 	| Download Link                                                                                         	| Extract to Relative Path       	|
|--------------	|-------------------------------------------------------------------------------------------------------	|--------------------------------	|
| CIFAR-10     	| [click here](https://drive.google.com/drive/folders/1bwf3qnaquRcfnoTfxKDwikVd_LnCitAm?usp=sharing)    	| ../Source_classifiers/cifar10  	|
| CIFAR-100    	| [click here](https://drive.google.com/drive/folders/1bnnkYORAwrjWI0jNhfVm_w0MvZH_DwJC?usp=share_link) 	| ../Source_classifiers/cifar100 	|
| ImageNet     	| PyTorch Default                                                               	|                                	|


### Run Scripts

You can run the commands below for the experiments:
```
bash scripts/cifar10_extra.sh
bash scripts/cifar100_extra.sh
bash scripts/imagenet_extra.sh
```

