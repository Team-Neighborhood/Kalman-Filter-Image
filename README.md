# Kalman filter for image

Basic kalman filter for image object tracking, noise remove. <br>
 - 2D optimization code (replace matrix inverse --> matrix multiplication)
 - pre&post process interface and example
 - Only depends on "numpy"

### keypoints stablizing & tracking
<p align="center">
    <img src='./result/KF_result.gif' width=60%>
    <br>
    :metal: **rock 'n' roll** :metal:
</p>

## Requirements
- python 3
- numpy
- pandas
- opencv-contrib-python


## Usage  

First, install libs

    pip install opencv-contrib-python
    pip install numpy
    pip install pandas

**Just run!** <br>

    python main.py