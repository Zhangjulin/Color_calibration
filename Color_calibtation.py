import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from scipy import linalg as la
from scipy.special import erf
import pickle

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

# Create a calibrated grayscale version of the image
def convert_grayscale_calib(image, sol):
    # Get size
    width, height = image.size

    # Create new image and a pixel map
    new = create_image(width, height)
    pixels = new.load()
    pixel_list = []

    # Transform to grayscale
    for i in range(width):
        for j in range(height):
            # Get Pixel
            pixel = get_pixel(image, i, j)

            # Get the calibrated RGB tristimulus values (int from 0 to 255)
            # Linear calibration
            red, green, blue = np.dot(np.array([1, pixel[0], pixel[1], pixel[2]]), sol)

            # Transform to grayscale
            gray = (red * 0.299) + (green * 0.587) + (blue * 0.114)

            # Set Pixel in new image
            gray_int = int(round(gray))
            pixels[i, j] = (gray_int, gray_int, gray_int)
            pixel_list.append(gray_int)

    # Get the mean and standard values of gray level histograms
    mean_pixels = np.mean(pixel_list)
    std_pixels = np.std(pixel_list)
    # print('the mean of the histogram is %f' % mean_pixels)
    # print('the standard of the histogram is %f' % std_pixels)
    print('%f %f' % (mean_pixels, std_pixels))
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
    # print('the red pixels are %s' %x_r)
    # print('the green pixels are %s' %x_g)
    # print('the blue pixels are %s' %x_b)

    # Linear calibration metehod
    X = np.vstack([x_r ** 0, x_r, x_g, x_b])

    # the parameters of the calibration function
    sol, r, rank, s = la.lstsq(X.T, Y.T)

    return sol

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
    # Load image
    # pic_num = np.arange(101, 135)
    # pic_num = np.array([203, 245])
    pic_num = np.array([203, 205, 210, 216, 220, 224, 228, 233, 237, 250, 256, 261, 264, 268, 272, 275, 281, 284, 290, 295]) # outdoor with filter
    # pic_num = np.array([201, 205, 209, 215, 219, 223, 227, 232, 236, 248, 254, 258, 263, 268, 271, 275, 279, 283, 288, 294]) # outdoor without filter
    # pic_num = np.array([118, 219, 220, 221])

    nums = len(pic_num) # number of images
    gray_pdfs = [] # create a list for the uncalibrated gray level intensity histograms of images
    calib_pdfs = [] # create a list for the calibrated gray level intensity histograms of images

    for num in range(nums):
        path = '/Users/%d.tiff' % pic_num[num] # the path of the rock image
        original = open_image(path)

        # Convert to gray level and save
        new, dict = convert_grayscale(original)
        # save_image(new, '/Gray.tiff')

        # Get the uncalibrated gray level intensity histogram of the image
        gray_pdf = pdf(dict)
        gray_pdfs.append(gray_pdf)

        # Load the color checker image
        path_bar = '/Users/bar_%d.tiff' % pic_num[num] # the path of the image of the color checker
        image = open_image(path_bar)

        # Get the parameters of the calibration function
        sol = calibration_color_check(image, weight=20)

        # Get the calibrated gray level intensity histograms of the image
        new2, dict2 = convert_grayscale_calib(original, sol)
        calib_pdf = pdf(dict2)
        calib_pdfs.append(calib_pdf)

    # Plot the uncalibrated gray level intensity histograms
    for i in range(nums):
        plt.plot(range(256), gray_pdfs[i], linewidth=2.0)
    plt.ylabel('Relative frequency', fontname='Arial', fontsize=15)
    plt.xlabel('Gray level intensity', fontname='Arial', fontsize=15)
    plt.xticks(fontname='Arial', fontsize=13)
    plt.yticks(fontname='Arial', fontsize=13)
    plt.tick_params(direction='in', width=2.0)
    ax = plt.gca()
    ax.legend(('Indoor', 'Outdoor'), prop={"family": "Arial", 'size': 15})
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(2.0)
    # plt.savefig('/Users/zhangjulin/Downloads/' + 'Fig8N.tif', dpi=600)
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
    ax.legend(('Indoor', 'Outdoor'), prop={"family": "Arial", 'size': 15})
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(2.0)
    # plt.savefig('/Users/zhangjulin/Downloads/' + 'Fig8O.tif', dpi=600)
    plt.show()

    # Get the proportion of dark pixels with Ostu thresholding algorithm
    for num in range(nums):
        threshold, area = Ostu_threshold(calib_pdfs[num])

    # print('the mean of peak 1 is %f, the area is %f' %(pars_1[1], area1/(area1+area2)))
    # print('the mean of peak 2 is %f, the area is %f' %(pars_2[1], area2/(area1+area2)))
    # print('%f %f' %(area1/(area1+area2), area2/(area1+area2)))
        print('%f %f' %(threshold, area))

main()