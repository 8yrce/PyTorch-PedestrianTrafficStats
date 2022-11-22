"""
PeopleVsBikes.py
  Detect and count the number of pedestrians using a path on foot vs on a bicycle
Bryce Harrington
11/21/22
"""
import os  # so we can grab existing images off the disk
import time  # so we can delay how often we take images from cam
import torch  # for handling our detection model
import cv2  # to handle input video streams
import imutils  # for helper functions like aspect ratio maintaining resize operations / CLAHE
from dataclasses import dataclass  # for more clean class instantiation
from argparse import ArgumentParser  # for cmd line operation

parser = ArgumentParser(description="Run people vs bike detection")
parser.add_argument("--video_file_path", metavar='v', type=str,
                    help="Path to input video file ( don't use this arg for webcam use )",
                    default=None)
parser.add_argument("--model_repo", metavar='r', type=str,
                    help="The repo to search for models in",
                    default="ultralytics/yolov5")
parser.add_argument("--model", metavar='m', type=str,
                    help="The model to grab from the repo for inference ( yolov5s, yolov5m, yolov5l, yolov5x )",
                    default="yolov5s")
parser.add_argument("--confidence", metavar='c', type=float,
                    help="Threshold for the minimum confidence value we will accept for a detection",
                    default=0.5)
parser.add_argument("--input_size", metavar='s', type=int,
                    help="Size to resize our input to before passing into model",
                    default=640)
parser.add_argument("--augmentation",
                    help="Apply augmentation ( CLAHE ) to improve performance in difficult domains",
                    action='store_true', default=False)
parser.add_argument("--demo_mode",
                    help="Run a demo on some sample images",
                    action='store_true', default=False)
parser.add_argument("--demo_path", metavar='d', type=str,
                    help="Path to the demo image set",
                    default="./demo_images/")
parser.add_argument("--debug", action="store_true",
                    help="Debug mode shows more information", default=False)
args = parser.parse_args()


@dataclass
class PersonOrBike:
    """
    This class allows us to open a video stream and determine the count of foot vs bike traffic
    """
    model = None
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    video_source = 1 if args.video_file_path is None else args.video_file_path
    video_feed = None
    people, bikes = 0, 0

    def __init__(self):
        """
        Main logic loop
        """
        # create model
        self.create_model()

        if args.demo_mode:
            # begin our demo using images
            self.demo_mode()
        else:
            # start the video feed
            self.start_video()

    # load model
    def create_model(self):
        """
        Create the object detection model for us to use and set our params
        :sets self.model:
        """
        try:
            self.model = torch.hub.load(args.model_repo, args.model)
            self.model.conf = args.confidence
            # since we only want people and bicycles
            self.model.classes = [0, 1]

        except Exception as e:
            print(f"ERROR: Unable to load model '{args.model}' from repo '{args.model_repo}', error:{e}")

    # demo mode
    def demo_mode(self):
        """
        Run a demo on sample images
        """
        for image in os.listdir(args.demo_path):
            im = cv2.imread(os.path.join(os.getcwd(), args.demo_path+image))
            self.score_detections(self.model_inference(im))
            cv2.imshow('Demo', im)
            cv2.waitKey(0)

    # open video feed
    def start_video(self):
        """
        Opens our video feed from the selected source
        :sets self.video_feed:
        """
        try:
            self.video_feed = cv2.VideoCapture(self.video_source)
            # grab first empty buffer
            _, image = self.video_feed.read()
            start_t = time.time()
            # while we have frames coming in
            while image is not None:
                # grab image
                _, image = self.video_feed.read()
                # check our delay
                if time.time() - start_t > 5:
                    # augment and inference with model
                    output = self.model_inference(self.data_augmenter(image))
                    # check inferences ( any that count get saved for reference )
                    self.score_detections(output)
                    start_t = time.time()

        except Exception as e:
            print(f"ERROR: Unable to open specified video feed {self.video_source}, error:{e}")

    # augment input image for model
    def data_augmenter(self, image):
        """
        Augment the incoming data to an optimal state for use with object detection model
        :param image: the image to augment
        :return image: the augmented image
        """
        # resize and pad to maintain size
        image = imutils.resize(image, width=args.input_size)
        image = cv2.copyMakeBorder(image, top=(args.input_size - image.shape[0]), bottom=0,
                                    left=(args.input_size - image.shape[1]), right=0,
                                    borderType=cv2.BORDER_CONSTANT)

        # debug
        if args.debug:
            cv2.imshow('Unequalized contrast', image)
            cv2.waitKey(5)

        if args.augmentation:
            # convert to HSV
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # split channels for CLAHE
            chans = cv2.split(image)
            # apply histogram equalization
            volume = self.clahe.apply(chans[2])
            # merge channels and convert back to bgr
            image = cv2.merge([chans[0], chans[1], volume])
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

            # debug
            if args.debug:
                cv2.imshow('CLAHE', image)
                cv2.waitKey(5)

        return image

    # inference with model
    def model_inference(self, image):
        return self.model(image)

    # check and see if any of our detections of class person/bike have crossed our end line
    def score_detections(self, detections):
        """
        'Score' our detections to keep track of how many people / bikes are present in the scene
        :param detections: 
        :return: 
        """
        results = detections.pandas().xyxy[0]
        self.people += len([x for x in results['class'] if x == 0])
        self.bikes += len([x for x in results['class'] if x == 1])
        if args.debug:
            print(detections)
            print(f"People: {self.people} | Bikes: {self.bikes}")


if __name__ == "__main__":
    PersonOrBike()
