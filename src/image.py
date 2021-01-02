import numpy as np
from skimage.transform import resize


class Image:

    def __init__(self, image: np.array):
        self.array = image

        # Representation of what the computer see,
        # to display on streamlit
        self.streamlit = None

        # Image array sanitized, to make a prediction
        self.prediction = None

    def get_streamlit_displayable(self):
        """
        Create a streamlit-ready version of the image, showing how the
        computer actually see what the user draw.
        :return: A numpy array of the image, with shape (w, h), rescaled to
            [0, 1] and resized to 28x28 pixels.
        :rtype: np.array
        """

        # If a streamlit version has not yet created, create it
        if self.streamlit is None:
            self.streamlit = self.resize(self.rescale(self.grayscale(self.array)))

        # Return the streamlit-ready displayable image
        return self.streamlit

    def get_prediction_ready(self):
        """
        Create a prediction-ready sanitized version of the image. This is what
        the model needs to make a prediction.
        :return: a numpy array of the image, with shape (1, w, h, 1), rescaled
            to [0, 1], resized to 28x28 pixels, with float32 dtype.
        :rtype: np.array
        """

        # If a prediction version has not yes created, create it
        if self.prediction is None:

            # Get the streamlit version
            image_streamlit = self.get_streamlit_displayable()

            # Transform it into a prediction-ready array
            self.prediction = self.to_float32(self.reshape(image_streamlit))

        return self.prediction

    def is_empty(self):

        if np.max(self.streamlit) == np.min(self.streamlit) == .0:
            return True

        return False

    @staticmethod
    def grayscale(image: np.array):
        """
        Transform a (width, height, n) n-channeled image (rbg or rgba)
        into a gray scaled (width, height) one.
        :param image: A numpy array of the image (width, height, n).
        :type image: np.array
        :return: A numpy array of the gray scaled image.
        :rtype: np.array
        """

        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    @staticmethod
    def rescale(image: np.array):
        """
        Rescale a given image from [0, 255] to [0, 1].
        :param image: A numpy array of the image, any shape.
        :type image: np.array
        :return: A numpy array of the rescaled image.
        :rtype: np.array
        """

        return image / 255

    @staticmethod
    def resize(image: np.array, width: int = 28, height: int = 28):
        """
        Resize a given image to a given width and height.
        :param image: A numpy array of the image, any shape.
        :param width: The new width of the image.
        :param height: The new height of the image.
        :type image: np.array
        :type width: int
        :type height: int
        :return: A numpy array of the resized image.
        :rtype: np.array
        """

        # Use the skimage.resize method
        return resize(image, (width, height))

    @staticmethod
    def reshape(image: np.array):
        """
        Reshape a given image into a single sample with 1 channel:
        Usable only for the model, to make prediction. Not displayable
        with streamlit.
        :param image: A numpy array of the image, any shape.
        :type image: np.array
        :return: A (1, width, height, 1) array of the image.
        :rtype: np.array
        """

        # Retrieve the width and the height image
        width, height, *_ = image.shape

        # Return a reshaped, prediction ready array of the image.
        return image.reshape(1, width, height, 1)

    @staticmethod
    def to_float32(image: np.array):
        """
        Change the dtype of a given numpy array's image into float32.
        :param image: A numpy array of the image, any shape.
        :type image: np.array
        :return: A float32 version of the array.
        :rtype: np.array
        """

        return image.astype('float32')