from PIL import Image
import scipy
from scipy import ndimage
from LogisticRegression import *

# Loading the data (cat/non-cat)
num_px, train_set_x, train_set_y, test_set_x, test_set_y, classes = load_dataset()
num_iterations = 2000
learning_rate = 0.005
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations, learning_rate, True)

def inference(my_image):
	# We preprocess the image to fit your algorithm.
    fname = "images/" +my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
    my_predicted_image = predict(d["w"], d["b"], my_image)

    plt.imshow(image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
	
if __name__ == "__main__":
	inference("dog.jpg")   # change this to the name of your image file 
	
	
