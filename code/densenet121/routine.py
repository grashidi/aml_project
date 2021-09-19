import os
from datetime import datetime
flag = ["no_ROI","with_ROI"]

model_training = [name + "/main.py" for name in flag]
grad_cam = [name + "/grad_cam.py" for name in flag]

switch = input("Computation Mode (Train, Grad-CAM, Full): ")

def densenet():
    for train in model_training:
        print(str(train))
        print(datetime.now(tz=None))
        os.system("python " + train)

def cam():
    for img in grad_cam:
        print(str(img))
        print(datetime.now(tz=None))
        os.system("python " + img)

if switch == "Train":
    print("MODE: Training")
    densenet()
elif switch == "Grad-CAM":
    print("MODE: Grad-CAM")
    cam()
elif switch == "Full":
    print("MODE: Full")
    densenet()
    cam()       
else:
    print("Script Terminated")