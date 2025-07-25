from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

class FaceCam(BoxLayout):
    def __init__(self, **kwargs):
        super(FaceCam, self).__init__(**kwargs)
        self.orientation = 'vertical'

        self.cam = Camera(play=True, resolution=(640, 480))
        self.add_widget(self.cam)

        self.detect_button = Button(text='Capture & Detect Face')
        self.detect_button.bind(on_press=self.detect_face)
        self.add_widget(self.detect_button)

        self.result = Image()
        self.add_widget(self.result)

    def detect_face(self, instance):
        # Save current frame to file
        self.cam.export_to_png("frame.png")

        # Read the image using OpenCV
        image = cv2.imread("frame.png")

        # Detect faces
        faces, confidences = cv.detect_face(image)
        output = draw_bbox(image, faces, confidences)

        # Convert OpenCV image to Kivy texture
        buf = cv2.flip(output, 0).tobytes()
        texture = Texture.create(size=(output.shape[1], output.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.result.texture = texture

class FaceApp(App):
    def build(self):
        return FaceCam()

if __name__ == '__main__':
    FaceApp().run()