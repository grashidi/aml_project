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

  
