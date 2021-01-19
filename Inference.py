#needed imports
import cv2
from tensorflow import keras


def Predict(image_path):
    #read image
    input_image = cv2.imread(image_path)
    #load our previously saved model
    CNN_model = keras.models.load_model("CNN_model.h5")
    #resize image
    input_image = cv2.resize(input_image, (28, 28))
    #show image
    cv2.imshow("label", input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #predict value, print and return
    predicted_value = CNN_model.predict_classes(input_image.reshape(1, 28, 28, 3))
    print("Predicted label: ", int(predicted_value))
    return predicted_value

#image path
path = ("test_image.jpg")
#test prediction
predicted = Predict(path)