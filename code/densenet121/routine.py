import os
from datetime import datetime
flag = ["no_ROI","with_ROI"]

model_training = [name + "/main.py" for name in flag]
grad_cam = [name + "/grad_cam.py" for name in flag]

# models traning
for train in model_training:
    print(datetime.now(tz=None))
    os.system("python" + train)

# grad cam computation
for img in grad_cam:
    print(datetime.now(tz=None))
    os.system("python" + img)