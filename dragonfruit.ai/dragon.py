"""
author: Jae Park
email: jaewoop@usc.edu

references:
    https://en.wikipedia.org/wiki/Graham_scan
    https://web.archive.org/web/20161018194403/https://www.niksula.hut.fi/~hkankaan/Homepages/metaballs.html
    https://stackoverflow.com/questions/3587704/good-way-to-procedurally-generate-a-blob-graphic-in-2d
    https://en.wikipedia.org/wiki/Point_in_polygon
    https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
    https://academic.mu.edu/phys/matthysd/web226/L0425.htm
"""

import os
# uncomment the line below to download missing python packages using pip
# os.system('python3 -m pip install cmath matplotlib')

import time
import cmath
import random
import numpy as np
import matplotlib.pyplot as plt

from math import pi
from math import atan2


INT_MAX = 100000

IMG_SIZE = 1000 # IMG_SIZE = 100000
MU = IMG_SIZE / 2
SIGMA = IMG_SIZE / 5 #5


# Graham's scan to find the convex hull of a finite set of points 
def convexHull(pts):
    xleftmost, yleftmost = min(pts)
    by_theta = [(atan2(x-xleftmost, y-yleftmost), x, y) for x, y in pts]
    by_theta.sort()
    as_complex = [complex(x, y) for _, x, y in by_theta]
    chull = as_complex[:2]
    for pt in as_complex[2:]:
        # perp product.
        while ((pt - chull[-1]).conjugate() * (chull[-1] - chull[-2])).imag < 0:
            chull.pop()
        chull.append(pt)
    
    return [(pt.real, pt.imag) for pt in chull]


# helper function for interpolateSmoothly
# discrete Fourier Transform
def dft(xs):    
    return [sum(x * cmath.exp(2j*pi*i*k/len(xs)) 
                for i, x in enumerate(xs)) for k in range(len(xs))]

                
def interpolateSmoothly(xs, N):
    # For each point, add N points
    fs = dft(xs)
    half = (len(xs) + 1) // 2
    fs2 = fs[:half] + [0]*(len(fs)*N) + fs[half:]
    
    return [x.real / len(xs) for x in dft(fs2)[::-1]]


# helper function for is_inside_polygon
# Given three collinear points p, q, r, 
# the function checks if point q lies on line segment 'pr'
def onSegment(p:tuple, q:tuple, r:tuple) -> bool:
     
    if ((q[0] <= max(p[0], r[0])) &
        (q[0] >= min(p[0], r[0])) &
        (q[1] <= max(p[1], r[1])) &
        (q[1] >= min(p[1], r[1]))):
        return True
         
    return False


# helper function for is_inside_polygon
# To find orientation of ordered triplet (p, q, r).
# The function returns following values
# 0 --> p, q and r are collinear
# 1 --> Clockwise
# 2 --> Counterclockwise
def orientation(p:tuple, q:tuple, r:tuple) -> int:
     
    val = (((q[1] - p[1]) *
            (r[0] - q[0])) -
           ((q[0] - p[0]) *
            (r[1] - q[1])))
            
    if val == 0:
        return 0
    if val > 0:
        return 1 # Points are collinear
    else:
        return 2 # Clock or counterclock


# helper function for is_inside_polygon
def doIntersect(p1, q1, p2, q2) -> bool:
     
    # Find the four orientations needed for 
    # general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
 
    # General case
    if (o1 != o2) and (o3 != o4):
        return True
     
    # p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if (o1 == 0) and (onSegment(p1, p2, q1)):
        return True
 
    # p1, q1 and p2 are collinear and q2 lies on segment p1q1
    if (o2 == 0) and (onSegment(p1, q2, q1)):
        return True
 
    # p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if (o3 == 0) and (onSegment(p2, p1, q2)):
        return True
 
    # p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if (o4 == 0) and (onSegment(p2, q1, q2)):
        return True
 
    return False
 

# Returns true if the point p lies 
# inside the polygon[] with n vertices
def is_inside_polygon(points, p) -> bool:
     
    n = len(points)
     
    # There must be at least 3 vertices
    # in polygon
    if n < 3:
        return False
         
    # Create a point for line segment
    # from p to infinite
    extreme = (INT_MAX, p[1])
     
    # To count number of points in polygon
    # whose y-coordinate is equal to
    # y-coordinate of the point
    decrease = 0
    count = i = 0
     
    while True:
        next = (i + 1) % n
         
        if(points[i][1] == p[1]):
            decrease += 1
         
        # Check if the line segment from 'p' to 
        # 'extreme' intersects with the line 
        # segment from 'polygon[i]' to 'polygon[next]'
        if (doIntersect(points[i], points[next], p, extreme)):          
            # If the point 'p' is collinear with line 
            # segment 'i-next', then check if it lies 
            # on segment. If it lies, return true, otherwise false
            if orientation(p, extreme, points[i]) == 0 and onSegment(p, points[i], extreme):
                return True
                  
            count += 1
            
        i = next
        if (i == 0):
            break
             
    # Reduce the count by decrease amount
    # as these points would have been added twice
    count -= decrease
     
    # Return true if count is odd, false otherwise
    return (count % 2 == 1)


def generate_blob(img_size = IMG_SIZE):
    print("a sample random image generation started...")
    # find convex hull from randomly generated points by Gaussian distribution
    pts = convexHull([(random.gauss(MU, SIGMA), random.gauss(MU, SIGMA)) for _ in range(10)])
    
    # connect all the points via smooth interpolation
    xs, ys = [interpolateSmoothly(zs, 10) for zs in zip(*pts)]
    blob_points = [list(a) for a in zip(xs, ys)]

    # # uncomment this section to view the sample
    # # microorganism blob generated from above
    # plt.plot(xs+[xs[0]], ys+[ys[0]])
    # plt.xlim([0, img_size])
    # plt.ylim([0, img_size])
    # plt.show()

    positive_pixel_count = 0

    rows, cols = (img_size, img_size)
    arr = [[int(0) for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            if is_inside_polygon(points = blob_points, p = [j,i]):
                arr[i][j] = 1
                positive_pixel_count += 1
            else: # point not in the enclosed region
                arr[i][j] = 0

    end = time.time()
    print("%.3f seconds took to generate a random microorgasm image of size %dx%d" % (end-start, img_size, img_size))
    print("\t%.2f percent of the image is occupied with the microorganism blob" % (positive_pixel_count / (img_size**2) * 100))

    return arr


# Run Length Encoding (RLE) of Binary Image 
def image_encoding(img, bits = 10):
    print("%.6fs image encoding started..." % (time.time()-start))
    encoded = []
    count = 0
    prev = None
    
    for row in img:
        for pixel in row:
            if prev==None:
                prev = pixel
                count+=1
            else:
                if prev!=pixel:
                    encoded.append((count, prev))
                    prev=pixel
                    count=1
                else:
                    if count < (2**bits)-1:
                        count+=1
                    else:
                        encoded.append((count, prev))
                        prev=pixel
                        count=1
    
    encoded.append((count, prev))
    partial_encoded_message = str(encoded[:7])
    print("\tencoded data: %s ..." % (partial_encoded_message))
    print("%.6fs encoding ended..." % (time.time()-start))

    return encoded


# Run Length Decoding of Binary Image 
def image_decoding(encoded, shape):
    print("%.6fs image decoding started..." % (time.time()-start))
    decoded=[]
    for rl in encoded:
        r,p = rl[0], rl[1]
        decoded.extend([p]*r)
    dimg = np.array(decoded).reshape(shape)
    print("%.6fs decoding ended..." % (time.time()-start))
    
    return dimg


def generate_cancer_img(img_size = IMG_SIZE):
    print("\na sample cancer image generation started...")

    rows, cols = (img_size, img_size)
    arr = [[int(0) for _ in range(cols)] for _ in range(rows)]

    for _ in range(20): # try generating 20 cancer cells in the sample image
        rand_x = random.randrange(0, IMG_SIZE)
        rand_y = random.randrange(0, IMG_SIZE)

        pts = convexHull([(random.gauss(rand_x, SIGMA/50), random.gauss(rand_y, SIGMA/50)) for _ in range(5)])
        xs, ys = [interpolateSmoothly(zs, 10) for zs in zip(*pts)]
        
        cancer_cell_bounding_coordinates = [list(a) for a in zip(xs, ys)]

        for i in range(rows):
            for j in range(cols):
                if is_inside_polygon(points = cancer_cell_bounding_coordinates, p = [j,i]):
                    arr[i][j] = 1
                else: # point not in the enclosed region
                    arr[i][j] = 0

    end = time.time()
    print("%.3f seconds took to generate a random cancer image of size %dx%d" % (end-start, img_size, img_size))
    # print("\t%.2f percent of the image is occupied with the microorganism blob" % (positive_pixel_count / (img_size**2) * 100))

    return arr

def display_image(img, title):
    plt.title(title)
    plt.imshow(img, cmap = 'gray')
    plt.xlim([0, IMG_SIZE])
    plt.ylim([0, IMG_SIZE])
    plt.show()


if __name__ == "__main__":
    global start 
    start = time.time()

    random_blob_image = generate_blob(img_size = IMG_SIZE)

    compressed_img = image_encoding(np.asarray(random_blob_image))

    decompressed_img = image_decoding(compressed_img, (IMG_SIZE, IMG_SIZE))
    
    display_image(decompressed_img, "micro-organism microscopic high resolution image")

    random_cancer_image = generate_cancer_img(img_size = IMG_SIZE)

    display_image(random_cancer_image, "dye sensor image")
