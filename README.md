Needed for usage:
- pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

# Pedestrian Traffic Stats

This project uses PyTorch and a small Yolov5 model to generate statistics on 'foot traffic'.

## Description

By leveraging Kera's functional architecture we can specify and compile a custom model for identifying one of 8 possible
diagnoses when presented with a patients ocular scans. This will allow faster, more accurate, and more convenient diagnosis
for many common issues that are traditionally found manually during an optometrist's review.

## Usage
There are a few parameters that control the way the program works, for example:
* video_file_path - if you want to run detection on a video specify this path
* model_repo - the path to the repo you want to pull models from
* model - the model you want to pull from the specified repo, default is yolov5s ( aka small yolo )
* confidence - the threshold for returning a positive detection after an inference
* input_size - desired input size to shape the data too for your chosen model
* augmentation - apply image augmentation to balance contrast for high / low exposure domains
* demo_mode - run a 'demo' on some test images to demonstrate usage
* demo_path - path to test images, if you would like to use your own set this
* debug - shows more information including the image being inferenced ( for your reference )


### Dependencies
* Cv2: 4.6.0
* Numpy: 1.21.5
* Torch: 1.12.1
* imutils: 0.5.4

### Installing and Running

* Clone repo
* pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt  # yolo dependencies
* Run demo: ``` python3 PeopleVsBikes.py --debug --demo_mode ``` ( debug is optional, but does show live image )
* Run on video file: ``` python3 PeopleVsBikes.py --debug ```
* Run with web-cam: ``` python3 PeopleVsBikes.py --debug ```

### Help
For errors opening the camera ( video source at index 0 unavailable )
* Ensure you have a webcam or external camera connected 
* In terminal, ''' ls /dev/video* '''. This should show a few video devices. If not troubleshoot device connection / drivers.
