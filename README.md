# aml_project

## Installation
Clone repository:<br>
  <ins>SSH:</ins><br>
  ```git clone git@github.com:grashidi/aml_project.git```<br>
  <ins>HTTPS:</ins><br>
    ```git clone https://github.com/grashidi/aml_project.git```<br>
  
Change into code directory:<br>
  ```cd aml_project/code```<br>
  
Make installation script executable:<br>
  ```chmod +x install.sh```<br>
  
Run installation script:<br>
  ```./install.sh```<br>

The installation script creates a virtual environment and installs all packages within this virtual environment.<br>
To activate the virtual environment run<br>
  ```source env/bin/activate```<br>
  
## Data
The tree structure below presents the data folder's different subfolders.<br>
<b>ct_scan</b>: Contains the CT-scans and the corresponding data split for DenseNet121 and ResNet18.<br>
<b>x-ray</b>: Contains the x-ray images and the corresponding data split for DenseNet121 and ResNet18.<br>
<b>segmentation</b>: Contains the x-ray images, CT-scans and the corresponding data split for UNet.<br>
<b>data_split</b>: Contains three text files with image names assigned to training, validation and testing.<br>
<b>images</b>: Contains the actual images.<br>
```
├── ct_scan
│   ├── data_split
│   │   ├── COVID
│   │   └── NonCOVID
│   └── images
│       ├── COVID
│       └── NonCOVID
├── segmentation
│   ├── ct_scan
│   │   ├── data_split
│   │   │   ├── images
│   │   │   └── masks
│   │   ├── images
│   │   └── masks
│   └── xray
│       ├── data_split
│       │   ├── images
│       │   └── masks
│       ├── images
│       └── masks
└── xray
    ├── data_split
    │   ├── COVID
    │   └── NonCOVID
    └── images
        ├── COVID
        └── NonCOVID
 ```
 
## Running the code
#### DenseNet121
#### ResNet18
Change into resnet18 directory<br>
```cd code/resnet18```<br>
Now you have to options train ResNet18 without ROI mask application or train ResNet18 with ROI mask application.<br><br>
For training without ROI mask application change into the <b>no_ROI</b> directory and run the main.py file.<br>
```cd no_ROI```<br>
```python main.py```<br><br>

  For training with ROI_mask application change into the <b>with_ROI</b> directory and run the main.py file<br>
```cd with_ROI```<br>
```python main.py```<br><br>

A model_backup folder will be created storing the trained model and the training statisics. If you also want to save the test statistics
you will have to provide the test function in the main.py file with a test stastics file path e.g.:<br>
```test(resnet18, criterion, test_loader, "some_test_statistics_file_path", additional_stats_enabled=True)```<br><br>

#### Grad-CAM investigation
If you want to conduct the Grad-CAM investigation change into the directory of the corresponding model e.g.:<br>
```cd resnet18/no_ROI```<br><br>

Make sure that the correct model is given to the Grad-CAM algorithm in the grad_cam.py file e.g.:<br>
```model_name = "resnet18_e10_bs10_12-09-2021_10:46:17.pt"```<br><br>

Then run the grad_cam.py file.<br>
```python grad_cam.py```<br><br>

This will create a grad_cam folder with a subfolder of the particular grad_cam run, storing the created grad_cam images with the heat map overlay.<br>
The images' labels and the model's predictions will be indicated in the created images.<br>
  
#### ROI mask channel replacement or ROI mask overlay
You can control the ROI mask application method the CovidDatset's overlay parameter.<br>
If you set the overlay parameter to <b>True</b> the ROI mask overlay will be applied e.g.:<br>

```CovidDataset(root_dir=root_dir,
                txt_COVID=txt_COVID + "train.txt",
                txt_NonCOVID=txt_NonCOVID + "train.txt",
                train=True,
                unet=unet,
                overlay=True,
                use_cache=USE_CACHE)
  ```
 Otherwise input image's last channel will be replaced with the ROI mask. The default value is set to <b>False</b>
