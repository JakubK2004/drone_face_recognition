import cv2
import os
import numpy as np
from PIL import Image
import pickle
import time
import pygame
from pygame.locals import *
from djitellopy import Tello

# Speed of the drone
S = 60
# Frames per second of the pygame window display
FPS = 30

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


def reco_init():
    recognizer.read("trainer.yml")

    with open("labels.pkl", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}
        return labels


class lock:
    def __init__(self):
        #  live video
        self.cap_ = cv2.VideoCapture(0)
        #  images
        self.label_ids = {}

    def start(self):
        """ The drone video will be processed in here to get frame by frame all of the faces, if possible with
        a 5 second delay for the display screen"""
        while True:
            #  Live video
            ret, frame = self.cap_.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.82, minNeighbors=5)
            #  ROI for live feed
            for (x, y, w, h) in faces:
                #  print(x, y, w, h)
                roi_gray = gray[y: y + h, x: x + w]
                roi_color = frame[y:y + h, x:x + w]

                id__, conf = recognizer.predict(roi_gray)

                if 60 <= conf <= 99.9:
                    print(reco_init()[id__])
                    print(conf)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    name = reco_init()[id__]
                    color = (200, 200, 200)
                    stroke = 2
                    cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

                img_item = 'my-image.png'
                cv2.imwrite(img_item, roi_gray)

                color = (255, 255, 255)
                stroke = 2
                width = x + w
                height = y + h
                cv2.rectangle(frame, (x, y), (width, height), color, stroke)
            cv2.imshow('frame', frame)

            if cv2.waitKey(20) & 0xFF == ord('z'):
                self.cap_.release()
                cv2.destroyAllWindows()
                break

    def recognize(self):

        BASE_dir = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(BASE_dir)

        current_id = 0
        y_labels = []
        x_train = []
        #  loading in images
        print('yup1')
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith('png') or file.endswith('jpg'):
                    path = os.path.join(root, file)

                    label = os.path.basename(os.path.dirname(path)).replace(" ", '-').lower()

                    if label not in self.label_ids:
                        self.label_ids[label] = current_id
                        current_id += 1
                    id_ = self.label_ids[label]

                    pil_image = Image.open(path).convert("L")
                    image_array = np.array(pil_image, 'uint8')
                    face = faceCascade.detectMultiScale(image_array, scaleFactor=1.82, minNeighbors=5)

                    #  ROI for images
                    for (x, y, w, h) in face:
                        print('yup2')
                        roi = image_array[y: y + h, x: x + w]
                        x_train.append(roi)
                        y_labels.append(id_)

        filesmap = []
        for root__, dirs__, files__ in os.walk('coding/projects/pycharm/faceRecv1/main'):
            for filew in files__:
                filesmap.append(filew)
        if 'label.pkl' not in filesmap:
            with open("labels.pkl", 'wb') as f:
                pickle.dump(self.label_ids, f)
        else:
            pass

        #  train cv recognizer
        recognizer.train(x_train, np.array(y_labels))
        recognizer.save("trainer.yml")
        print('done and saved')


class FrontEnd(object):
    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations
            - W and S: Up and down.
    """

    def __init__(self):
        # Init pygame
        pygame.init()

        # Creat pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 15

        self.send_rc_control = False

        # create update timer
        pygame.time.set_timer(USEREVENT + 1, 50)

    def run(self):

        if not self.tello.connect():
            print("Tello not connected")
            return

        if not self.tello.set_speed(self.speed):
            print("Not set speed to lowest possible")
            return

        # In case streaming is on. This happens when we quit this program without the escape key.
        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.streamon():
            print("Could not start video stream")
            return

        frame_read = self.tello.get_frame_read()

        should_stop = False
        while not should_stop:

            for event in pygame.event.get():
                if event.type == USEREVENT + 1:
                    self.update()
                elif event.type == QUIT:
                    should_stop = True
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == KEYUP:
                    self.keyup(event.key)

            if frame_read.stopped:
                frame_read.stop()
                break

            self.screen.fill([0, 0, 0])
            frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
            framex = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2GRAY)
            #  detecting
            faces = faceCascade.detectMultiScale(framex, scaleFactor=1.82, minNeighbors=5)
            #  ROI for live feed
            for (x, y, w, h) in faces:
                #  print(x, y, w, h)
                roi_gray = framex[y: y + h, x: x + w]

                id__, conf = recognizer.predict(roi_gray)

                if 60 <= conf <= 99.9:
                    print(reco_init()[id__])
                    print(conf)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    name = reco_init()[id__]
                    color_ = (200, 200, 200)
                    stroke = 2
                    cv2.putText(frame, name, (x, y), font, 1, color_, stroke, cv2.LINE_AA)

                img_item = 'my-image.png'
                cv2.imwrite(img_item, roi_gray)

                color__ = (255, 255, 255)
                stroke = 2
                width = x + w
                height = y + h
                cv2.rectangle(frame, (x, y), (width, height), color__, stroke)
            cv2.imshow('frame', frame)
            #  stop detecting part

            frame = np.rot90(frame)
            frame = np.flipud(frame)
            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            time.sleep(1 / FPS)

        self.tello.end()

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -S
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = S
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = S
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -S
        elif key == pygame.K_a:  # set yaw clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_d:  # set yaw counter clockwise velocity
            self.yaw_velocity = S

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            self.tello.land()
            self.send_rc_control = False

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)


def run():
    instance = lock()
    #  frontend = FrontEnd()
    onlyfiles = [f for f in os.listdir('E:\\projects\\coding\\projects\\pycharm\\faceRecv1\\main')
                 if os.path.isfile(os.path.join('E:\\projects\\coding\\projects\\pycharm\\faceRecv1\\main', f))]
    if "trainer.yml" in onlyfiles:
        reco_init()
        instance.start()
        #  frontend.run()
    else:
        instance.recognize()
        instance.start()
        reco_init()
        #  frontend.run()


if __name__ == "__main__":
    run()
