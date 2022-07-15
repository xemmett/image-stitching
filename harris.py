import matplotlib.pyplot as plt

import numpy

from scipy.ndimage import filters

from PIL import Image


def harris_repsonse(im, sigma):
    """
    Grey scale img
    Computes harris corner response for each pixel.
    """

    # derivatives
    im_x = numpy.zeros(im.shape)
    filters.gaussian_filter(im, 1, (0, 1), im_x)
    im_y = numpy.zeros(im.shape)
    filters.gaussian_filter(im, 1, (1, 0), im_y)
    
    # compute components of harris matrix
    Wxx = filters.gaussian_filter(im_x * im_x, sigma)
    Wyy = filters.gaussian_filter(im_y * im_y, sigma)
    Wxy = filters.gaussian_filter(im_x * im_y, sigma)

    # determinant and trace
    determinant = (Wxx * Wyy) - (Wxy ** 2)
    trace = Wxx + Wyy + 1
    return determinant / trace

def find_harris_points(img, threshold, min_dist=10):
    """
    Return corners from a Harris response image min_dist is the miin number of pixels seping corners and image boundary
    """

    corner_threshold = img.max() * threshold

    harris_t = (img > corner_threshold)

    # Get coords of candidates
    coords = numpy.array(harris_t.nonzero()).T

    # their values
    candidate_values = numpy.array([img[c[0], c[1]] for c in coords])

    # Sort candidates
    index = numpy.argsort(candidate_values)

    # store allowed point locations in array
    allowed_locations = numpy.zeros(img.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    # select the best points taking min_dist into account.
    filtered_coords = []
    for i in index[::-1]:
        r, c = coords[i]
        if( allowed_locations[r, c]):
            filtered_coords.append(coords[i])

            allowed_locations[(coords[i, 0]-min_dist):(coords[i, 0]+min_dist), (coords[i, 1]-min_dist):(coords[i, 1]+min_dist)] = 0

    return filtered_coords

def plot_harris_points(img, filtered_coords):
    """
    Plots corners found in image
    """
    plt.figure()
    plt.gray()
    plt.imshow(img)

    x = [p[1] for p in filtered_coords]
    y = [p[0] for p in filtered_coords]

    plt.plot(x, y, '*')

def plot_comparison(original, filtered, filter_name):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')


def harris_corner_detection(harrisim, min_dist=10):

    thresholds = [x/10 for x in range(1, 10)]

    filtered_coords = []
    for i in thresholds:
        filtered_coords += find_harris_points(harrisim, threshold=i, min_dist=min_dist)

    # plot_harris_points(img, filtered_coords)
    return filtered_coords

def get_descriptors(img, corner_coords, width):
    descriptors = []
    for coords in corner_coords:
        x = coords[0]
        y = coords[1]
        corner_patch = img[x-width:x+width+1, y-width:y+width+1].flatten()

        # plt.imshow(corner_patch, cmap='gray')
        # plt.show()
        
        corner_patch_mean = numpy.mean(corner_patch)
        corner_patch = corner_patch - corner_patch_mean
        
        normalized_corner_patch = corner_patch / numpy.linalg.norm(corner_patch)
        descriptors.append(normalized_corner_patch)
    return descriptors




def plot_matches(image1, image2, coords1, coords2, pairs):
    rows1 = image1.shape[0]
    rows2 = image2.shape[0]

    if( rows1 < rows2 ):
        image1 = numpy.concatenate((image1, numpy.zeros( (rows2 - rows1, image1.shape[1]))))
    else:
        image2 = numpy.concatenate((image2, numpy.zeros( (rows1 - rows2, image2.shape[1]))))

    stitched_image = numpy.concatenate((image1, image2), axis=1)

    plt.imshow(stitched_image, cmap=plt.cm.gray)

    for i1, i2 in pairs:
        plt.plot(
            [
                coords1[i1][1], coords2[i2][1] + image1.shape[1]
            ], 
            [
                coords1[i1][0], coords2[i2][0]
            ], 'c'
        )

def ransac_algorithm(pairs, coords1, coords2, distance):
    d2 = distance**2

    offsets = numpy.zeros((len(pairs), 2))

    for i, pair in enumerate(pairs):
        i1, i2 = pair
        offs1 = coords1[i1][0] - coords2[i2][0]
        offs2 = coords1[i1][1] - coords2[i2][1]
        num_of_matches = 1

        if( offs1 - 1e6)**2 + (offs2 - 1e6)**2 >= d2:
            sum_row, sum_col = offs1, offs2

            for j, pair in enumerate(pairs):
                if(j != i):
                    offsj1 = offsets[j, 0]
                    offsj2 = offsets[j, 1]

                    if( (offs1 - offsj1)**2 + (offs2 - offsj2)**2 < d2):
                        sum_row += offsj1
                        sum_col += offsj2

                        num_of_matches += 1

            if( sum_row and sum_col):
                row_offset = sum_row / num_of_matches
                col_offset = sum_col / num_of_matches

    return row_offset, col_offset


def get_pairs(descriptors1, descriptors2, threshold = 0.95):
    array1 = numpy.asarray(descriptors1, dtype = numpy.float32)
    array2 = numpy.asarray(descriptors2, dtype = numpy.float32).T # note the Transpose
    
    # Find the maximum values of array1, array2 and the dot product
    responseMatrix = numpy.dot(array1, array2)
    max1 = array1.max()
    max2 = array2.max()
    maxDotProduct = responseMatrix.max()
    
    # Initial, non-thresholded dot product - compared with the thresholded version below
    originalMatrix = Image.fromarray(responseMatrix * 255)
    
    pairs = []
    for r in range(responseMatrix.shape[0]):
        rowMaximum = responseMatrix[r][0]
        for c in range(responseMatrix.shape[1]):
            if (responseMatrix[r][c] > threshold) and (responseMatrix [r][c] > rowMaximum):
                pairs.append((r,c))
            else:
                responseMatrix[r][c] = 0
    return responseMatrix, pairs

def appendImages(image1, image2, rowOffset, columnOffset):
    # Convert floats to ints
    rowOffset = int(rowOffset)
    columnOffset = int(columnOffset)

    canvas = Image.new(image1.mode, (image1.width + abs(columnOffset), image1.height + abs(
        rowOffset)))  # create new 'canvas' image with calculated dimensions
    if columnOffset < 0:
        image1, image2 = image2, image1
        columnOffset *= -1
        rowOffset *= -1
    if rowOffset < 0:
        canvas.paste(image1, (0, canvas.height - image1.height))  # paste image1
        canvas.paste(image2, (columnOffset, canvas.height - image1.height + rowOffset))  # paste image2
    else:
        canvas.paste(image1, (0, 0))  # paste image1
        canvas.paste(image2, (columnOffset, rowOffset))  # paste image2

    # plot final composite image
    plt.figure('Final Composite Image')
    plt.imshow(canvas)
    plt.axis('off')
    plt.show()

    return canvas

colour_imgs = []
images = []
harrisims = []
corners = []
descriptors = []

for i, name in enumerate(['1.jpg', '2.jpg']):
    colour_img = Image.open(f"car_set/{name}")
    # if('arch2' in name):
    #     colour_img = colour_img.transform()
    colour_imgs.append(colour_img)

    greyscale_im = colour_img.convert('L')
    img = numpy.array(greyscale_im)
    images.append(img)

    harrisim = harris_repsonse(im=img, sigma=1)
    harrisims.append(harrisim)
    
    corner_coords = harris_corner_detection(harrisim, min_dist=10)
    corners.append(corner_coords)

    descriptor_vectors = get_descriptors(img, corner_coords, width=5)
    descriptors.append(descriptor_vectors)


    plt.imshow(harrisim)
    plt.show()

plot_harris_points(images[0], corners[0])
plot_harris_points(images[1], corners[1])
plt.show()

threshoded_matrix, pairs = get_pairs(descriptors[0], descriptors[1], threshold=0.9)

plot_matches(images[0], images[1], corners[0], corners[1], pairs)

row_offset, col_offset = ransac_algorithm(pairs, corners[0], corners[1], distance=10)

appendImages(colour_imgs[0], colour_imgs[1], row_offset, col_offset)
