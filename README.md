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
#### ResNe18
Change into resnet18 directory<br>
```cd code/resnet18```<br>
Now you have to options train ResNet18 without ROI mask application or train ResNet18 with ROI mask application.<br><br>
For training without ROI mask application change into the <b>no_ROI<b/> directory and run the main.py file.<br>
```cd no_ROI```<br>
```python main.py```<br><br>

  For training with ROI_mask application change into the <b>with_ROI</b> directory and run the main.py file<br><br>
```cd with_ROI```<br>
```python main.py```<br><br>
  
#### ROI mask channel replacement or ROI mask overlay
To change the ROI application method open the code/util/covid_dataset.py file. Then go the mask_transform method in line 141.<br><br>

```
def mask_transform(self, x):
        """
        Set pixels outside of mask to zero

        Args:
            x (tensor): Tensor to be masked

        Returns:
            x (tensor): Masked tensor
        """
        if self.unet:
            threshold = 0.5
            mask = self.unet(x[None,:,:,:])
            mask = self.normalize_to_range_0_1(mask)
            zero = torch.zeros_like(mask, dtype=torch.long)
            one = torch.ones_like(mask, dtype=torch.long)
            mask = torch.where(mask >= threshold, one, zero)
            x[-1,:,:] = mask[0,0,:,:] # replace last channel with mask
            # x[mask[0,:,:,:].repeat(3,1,1) == 0] = torch.min(x) # set pixels outside of mask to min value
        return x
```<br>
In version above the input image's last channel will be replace with ROI mask. If you want to apply the ROI mask overlay method which sets all pixels located outside of the mask to the minimum value of all pixels just comment out this line of code:<br>
```# x[-1,:,:] = mask[0,0,:,:] # replace last channel with mask```<br><br>
And activate the line of code below:<br><br>
```x[mask[0,:,:,:].repeat(3,1,1) == 0] = torch.min(x) # set pixels outside of mask to min value```<br><br>

 

  
