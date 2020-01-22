# drone_face_recognition
This was a little side project where I used a ryze tello drone to recognize faces using its camera.
The 'LOCK FACE' file has all the code that is needed to run, 
you only need create a file within the same directory with your photo's ( obviously update the paths!)
Then it will do everything automaticly, but remember if you want to use your PC's camera run the instance.start() method 
and if using the tello drone, first connect to it using normal wifi, then replace the instance.start() with frontend.run().
You can use your PC to fly around and if it detects a face it will recognize it!

if you get an error I suggest look in these links for an answer.
these are my sources that I used writing and merging this code.

https://github.com/codingforentrepreneurs/OpenCV-Python-Series
https://github.com/shantnu/Webcam-Face-Detect

https://www.youtube.com/watch?v=PmZ29Vta7Vc
