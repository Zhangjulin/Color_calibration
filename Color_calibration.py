import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from scipy import linalg as la
from skimage.feature import greycomatrix, greycoprops

# Open an image
def open_image(path):
    image = Image.open(path)
    return image

# Save image
def save_image(image, path):
    image.save(path, 'tif')

# Create a new image with the given size
def create_image(i, j):
    image = Image.new("RGB", (i, j), "white")
    return image

# Get the pixel from the given image
def get_pixel(image, i, j):
    # Inside image bounds?
    width, height = image.size
    if i > width or j > height:
        return None

    # Get pixel
    pixel = image.getpixel((i, j))
    return pixel

# Create a uncalibrated grayscale version of the image
def convert_grayscale(image):
    # Get size
    width, height = image.size

    # Create new image and a pixel map
    new = create_image(width, height)
    pixels = new.load()
    pixel_list = []

    # Transform to grayscale
    for i in range(width):
        for j in range(height):
            # Get pixel
            pixel = get_pixel(image, i, j)

            # Get RGB tristimulus values (int from 0 to 255)
            red =   pixel[0]
            green = pixel[1]
            blue =  pixel[2]

            # Transform to grayscale
            gray = (red * 0.299) + (green * 0.587) + (blue * 0.114)

            # Set pixel in new image
            gray_int = int(round(gray))
            pixels[i, j] = (gray_int, gray_int, gray_int)
            pixel_list.append(gray_int)

    dict = {}
    for key in pixel_list:
        dict[key] = dict.get(key, 0) + 1

    return new, dict

# Get the gray level histograms of images
def pdf(dict):
    gray_list = []
    for gray_level in range(256):
        if gray_level in dict:
          gray_list.append(dict[gray_level])
        else:
          gray_list.append(0)
    gray_pdf = [x / float(sum(gray_list)) for x in gray_list]  # Gray level histogram

    return gray_pdf

# Remove the outlier pixels (e.g. dust) of the color checker
def remove_outlier(mean, pixels, weight):
    new_pixels = [] # Create a list for the normal pixel values
    for pixel in np.array(pixels):
        if (pixel >= (mean - weight)) & (pixel <= (mean + weight)): # |pixel_value - mean_value| <= weight
            new_pixels.append(pixel)
        if new_pixels == []:
            return mean
    return np.mean(np.array(new_pixels))

# Get the RGB tristimulus measurements of a certain patch on the color checker
# RGB tristimulus measurements are determined by the mean values of all the pixels (remove the outliers) in a patch on the color checker
def color_check_mean(image, width_1, width_2, height_1, height_2, weight):
    # Create a list for RGB tristimulus measurements of a given color checker patch
    red_pixels = []
    green_pixels = []
    blue_pixels = []

    for i in range(width_1, width_2+1):
        for j in range(height_1, height_2+1):
            pixel = get_pixel(image, i, j)
            red = pixel[0]
            green = pixel[1]
            blue = pixel[2]
            red_pixels.append(red)
            green_pixels.append(green)
            blue_pixels.append(blue)

    # Mean value of RGB tristimulus values of a color checker patch
    red_mean = np.mean(np.array(red_pixels))
    green_mean = np.mean(np.array(green_pixels))
    blue_mean = np.mean(np.array(blue_pixels))

    # Remove the outlier of RGB tristimulus values of a color checker patch
    red_mean_new = remove_outlier(red_mean, red_pixels, weight)
    green_mean_new = remove_outlier(green_mean, green_pixels, weight)
    blue_mean_new = remove_outlier(blue_mean, blue_pixels, weight)

    return red_mean_new, green_mean_new, blue_mean_new

# Get the parameters of the calibration function
def calibration_color_check(image, weight):
    # the RGB tristimulus values ( ground truth) of color check
    Y_r = np.array([115, 194, 98, 87, 133, 103, 214, 80, 193, 94, 157, 224, 56, 70, 175, 231, 187, 8, 243, 200, 160, 122, 85, 52])
    Y_g = np.array([82, 150, 122, 108, 128, 189, 126, 91, 90, 60, 188, 163, 61, 148, 54, 199, 86, 133, 243, 200, 160, 122, 85, 52])
    Y_b = np.array([68, 130, 157, 67, 177, 170, 44, 166, 99, 108, 64, 46, 150, 73, 60, 31, 149, 161, 242, 200, 160, 121, 85, 52])
    Y = np.vstack([Y_r, Y_g, Y_b])

    # Create a list for the calibrated RGB tristimulus values
    x_r = []
    x_g = []
    x_b = []

    ch_width, ch_height = image.size # Get the color checker image size
    pos_width = np.array([1.5, 4.5, 9.1, 12.1, 16.7, 19.7, 24.3, 27.3, 31.9, 34.9, 39.5, 42.5])
    pos_width = np.round(pos_width / 44.0 * ch_width) # Get the width positions of patches on the color checker
    pos_height = np.array([1.5, 4.5, 9.1, 12.1, 16.7, 19.7, 24.3, 27.3])
    pos_height = np.round(pos_height / 28.8 * ch_height) # Get the height positions of patches on the color checker

    for j in range(4):
        for i in range(6):
            red_mean, green_mean, blue_mean = color_check_mean(image, width_1=int(pos_width[2*i]), width_2=int(pos_width[2*i+1]), height_1=int(pos_height[2*j]), height_2=int(pos_height[2*j+1]), weight=weight)
            x_r.append(red_mean)
            x_g.append(green_mean)
            x_b.append(blue_mean)
    x_r = np.array(x_r)
    x_g = np.array(x_g)
    x_b = np.array(x_b)

    # Linear calibration metehod
    X = np.vstack([x_r ** 0, x_r, x_g, x_b])

    # the parameters of the calibration function
    sol, r, rank, s = la.lstsq(X.T, Y.T)

    return sol

def RGB_to_HSI(red, green, blue):
  smooth = 1e-10
  intensity = (red + green + blue) / 3.0

  minimum = np.minimum(np.minimum(red, green), blue)
  saturation = 1 - minimum / intensity

  sqrt_calc = np.sqrt((red - green) ** 2 + (red - blue) * (green - blue)) + smooth
  hue = np.arccos(((0.5 * ((red - green) + (red - blue)) + smooth) / sqrt_calc))
  if (green < blue):
      hue = 2 * np.pi - hue
  hue = hue * 180 / np.pi
  return hue, saturation, intensity

# Create a calibrated grayscale version of the image
def convert_grayscale_calib(image, sol):
    # Get size
    width, height = image.size

    # Create new image and a pixel map
    new = create_image(width, height)
    pixels = new.load()
    pixel_list, pixel_matrix = [], [[0]*width for _ in range(height)]
    hue_list, saturation_list, intensity_list = [], [], []

    # Transform to grayscale
    for i in range(width):
        for j in range(height):
            # Get Pixel
            pixel = get_pixel(image, i, j)

            # Get the calibrated RGB tristimulus values (int from 0 to 255)
            # Linear calibration
            red, green, blue = np.dot(np.array([1, pixel[0], pixel[1], pixel[2]]), sol)
            gray = (red * 0.299) + (green * 0.587) + (blue * 0.114)

            # Set Pixel in new image
            gray_int = int(round(gray))
            pixels[i, j] = (gray_int, gray_int, gray_int)
            pixel_list.append(gray_int)
            pixel_matrix[j][i] = gray_int

            # Get the HSI components
            hue, saturation, intensity = RGB_to_HSI(red/255.0, green/255.0, blue/255.0)
            hue_list.append(hue)
            saturation_list.append(saturation)
            intensity_list.append(intensity)

    # Get the mean and standard values of gray level histograms
    mean_pixels = np.mean(pixel_list)
    std_pixels = np.std(pixel_list)

    print('The mean of gray level is %f' % mean_pixels)
    print('The standard deviation of gray level is %f' % std_pixels)

    # Print HSI components
    print('The mean of hue is %f' % np.mean(hue_list))
    print('The standard deviation of hue is %f' % np.std(hue_list))
    print('The mean of saturation is %f' % np.mean(saturation_list))
    print('The standard deviation of saturation is %f' % np.std(saturation_list))
    print('The mean of intensity is %f' % np.mean(intensity_list))
    print('The standard deviation of intensity is %f' % np.std(intensity_list))

    dict = {}
    for key in pixel_list:
        dict[key] = dict.get(key, 0) + 1
    # save_image(new, '/path/new.tiff')
    return pixel_matrix, dict

# Ostu threshold algorithm for image binarization
# Get the proportion of dark pixels
def Ostu_threshold(gray_pdf):
    pdf = np.array(gray_pdf)
    min = np.inf
    threshold = 0
    area = 0
    for i in np.arange(255):
        weight_back = np.sum(pdf[:i+1])
        weight_fore = np.sum(pdf[i+1:])

        if (weight_back != 0) & (weight_fore != 0):
            mean_back = np.dot(pdf[:i+1], np.arange(i+1)) / weight_back
            var_back = np.dot((np.arange(i+1) - mean_back)**2, pdf[:i+1]) / weight_back

            mean_fore = np.dot(pdf[i+1:], np.arange(i+1, 256)) / weight_fore
            var_fore = np.dot((np.arange(i+1, 256) - mean_fore)**2, pdf[i+1:]) / weight_fore

            variance = weight_back * var_back + weight_fore * var_fore

            if variance < min:
                min = variance
                threshold = i
                area = weight_back / (weight_back + weight_fore)

    return threshold, area


# Main
def main():
    # Load image. Type in the image number
    pic_num = np.array([32])
    # pic_num = np.array(range(1,60))

    nums = len(pic_num) # number of images
    gray_pdfs = [] # create a list for the uncalibrated gray level intensity histograms of images
    calib_pdfs = [] # create a list for the calibrated gray level intensity histograms of images

    for num in range(nums):
        path = '/Users/path/%d.tiff' % pic_num[num]  # the path of the rock image
        original = open_image(path)

        # Convert to gray level and save
        new, dict = convert_grayscale(original)
        # save_image(new, '/Gray.tiff')

        # Get the uncalibrated gray level intensity histogram of the image
        gray_pdf = pdf(dict)
        gray_pdfs.append(gray_pdf)

        # Load the color checker image
        path_bar = '/Users/path/bar_%d.tiff' % pic_num[num]  # the path of the image of the color checker
        image = open_image(path_bar)

        # Get the parameters of the calibration function
        sol = calibration_color_check(image, weight=20)

        # Get the calibrated gray level intensity histograms of the image
        pixel_matrix, dict2 = convert_grayscale_calib(original, sol)
        # save_image(new2, 'Chem/27_new2.tiff')

        calib_pdf = pdf(dict2)
        calib_pdfs.append(calib_pdf)
        props = ['contrast', 'homogeneity', 'energy', 'correlation']
        glcm = greycomatrix(pixel_matrix, [1], [0, np.pi/4, np.pi/2, np.pi/4*3, np.pi, np.pi/4*5, np.pi/4*6, np.pi/4*7], 256, symmetric=True, normed=True)

        # Get the haralick features from gray-level co-occurrence matrix
        for prop in props:
            res = np.mean(greycoprops(glcm, prop))
            print('The %s is %f' % (prop, res))


    # Plot the uncalibrated gray level intensity histograms
    for i in range(nums):
        plt.plot(range(256), gray_pdfs[i], linewidth=2.0)
    plt.ylabel('Relative frequency', fontname='Arial', fontsize=15)
    plt.xlabel('Gray level intensity', fontname='Arial', fontsize=15)
    plt.xticks(fontname='Arial', fontsize=13)
    plt.yticks(fontname='Arial', fontsize=13)
    plt.tick_params(direction='in', width=2.0)
    ax = plt.gca()
    # ax.legend(('Indoor', 'Outdoor'), prop={"family": "Arial", 'size': 15})
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(2.0)
    plt.show()

    # Plot the calibrated gray level intensity histograms
    for j in range(nums):
        plt.plot(range(256), calib_pdfs[j], linewidth=2.0)
    plt.ylabel('Relative frequency', fontname='Arial', fontsize=15)
    plt.xlabel('Gray level intensity', fontname='Arial', fontsize=15)
    plt.xticks(fontname='Arial', fontsize=13)
    plt.yticks(fontname='Arial', fontsize=13)
    plt.tick_params(direction='in', width=2.0)
    ax = plt.gca()
    # ax.legend(('Indoor', 'Outdoor'), prop={"family": "Arial", 'size': 15})
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(2.0)
    plt.show()

    # Get the proportion of dark pixels with Ostu thresholding algorithm
    for num in range(nums):
        threshold, area = Ostu_threshold(calib_pdfs[num])
        print('The dark mineral proportion is %f' % area)

main()
