

import os
import sys
import cv2
from  numpy import math as math
import numpy as np
import matplotlib.pyplot as plt


def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Perspective warping")
   print("2 Cylindrical warping")
   print("3 Bonus perspective warping")
   print("4 Bonus cylindrical warping")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image1] " + "[path to input image2] " + "[path to input image3] " + "[output directory]")

'''
Detect, extract and match features between img1 and img2.
Using SIFT as the detector/extractor, but this is inconsequential to the user.

Returns: (pts1, pts2), where ptsN are points on image N.
    The lists are "aligned", i.e. point i in pts1 matches with point i in pts2.

Usage example:
    im1 = cv2.imread("image1.jpg", 0)
    im2 = cv2.imread("image2.jpg", 0)
    (pts1, pts2) = feature_matching(im1, im2)

    plt.subplot(121)
    plt.imshow(im1)
    plt.scatter(pts1[:,:,0],pts1[:,:,1], 0.5, c='r', marker='x')
    plt.subplot(122)
    plt.imshow(im2)
    plt.scatter(pts1[:,:,0],pts1[:,:,1], 0.5, c='r', marker='x')
'''
def feature_matching(img1, img2, savefig=False):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches2to1 = flann.knnMatch(des2,des1,k=2)
    xrange = range
    matchesMask_ratio = [[0,0] for i in xrange(len(matches2to1))]
    match_dict = {}
    for i,(m,n) in enumerate(matches2to1):
        if m.distance < 0.7*n.distance:
            matchesMask_ratio[i]=[1,0]
            match_dict[m.trainIdx] = m.queryIdx

    good = []
    recip_matches = flann.knnMatch(des1,des2,k=2)
    matchesMask_ratio_recip = [[0,0] for i in xrange(len(recip_matches))]

    for i,(m,n) in enumerate(recip_matches):
        if m.distance < 0.7*n.distance: # ratio
            if m.queryIdx in match_dict and match_dict[m.queryIdx] == m.trainIdx: #reciprocal
                good.append(m)
                matchesMask_ratio_recip[i]=[1,0]



    if savefig:
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask_ratio_recip,
                           flags = 0)
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,recip_matches,None,**draw_params)

        plt.figure(),plt.xticks([]),plt.yticks([])
        plt.imshow(img3,)
        plt.savefig("feature_matching.png",bbox_inches='tight')

    return ([ kp1[m.queryIdx].pt for m in good ],[ kp2[m.trainIdx].pt for m in good ])

'''
Warp an image from cartesian coordinates (x, y) into cylindrical coordinates (theta, h)
Returns: (image, mask)
Mask is [0,255], and has 255s wherever the cylindrical images has a valid value.
Masks are useful for stitching

Usage example:

    im = cv2.imread("myimage.jpg",0) #grayscale
    h,w = im.shape
    f = 700
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
    imcyl = cylindricalWarpImage(im, K)
'''
def cylindricalWarpImage(img1, K, savefig=False):
    f = K[0,0]

    im_h,im_w = img1.shape

    # go inverse from cylindrical coord to the image
    # (this way there are no gaps)
    cyl = np.zeros_like(img1)
    cyl_mask = np.zeros_like(img1)
    cyl_h,cyl_w = cyl.shape
    x_c = float(cyl_w) / 2.0
    y_c = float(cyl_h) / 2.0
    for x_cyl in np.arange(0,cyl_w):
        for y_cyl in np.arange(0,cyl_h):
            theta = (x_cyl - x_c) / f
            h     = (y_cyl - y_c) / f

            X = np.array([math.sin(theta), h, math.cos(theta)])
            X = np.dot(K,X)
            x_im = X[0] / X[2]
            if x_im < 0 or x_im >= im_w:
                continue

            y_im = X[1] / X[2]
            if y_im < 0 or y_im >= im_h:
                continue

            cyl[int(y_cyl),int(x_cyl)] = img1[int(y_im),int(x_im)]
            cyl_mask[int(y_cyl),int(x_cyl)] = 255


    if savefig:
        plt.imshow(cyl, cmap='gray')
        plt.savefig("cyl.png",bbox_inches='tight')

    return (cyl,cyl_mask)

'''
Calculate the geometric transform (only affine or homography) between two images,
based on feature matching and alignment with a robust estimator (RANSAC).

Returns: (M, pts1, pts2, mask)
Where: M    is the 3x3 transform matrix
       pts1 are the matched feature points in image 1
       pts2 are the matched feature points in image 2
       mask is a binary mask over the lists of points that selects the transformation inliers

Usage example:
    im1 = cv2.imread("image1.jpg", 0)
    im2 = cv2.imread("image2.jpg", 0)
    (M, pts1, pts2, mask) = getTransform(im1, im2)

    # for example: transform im1 to im2's plane
    # first, make some room around im2
    im2 = cv2.copyMakeBorder(im2,200,200,500,500, cv2.BORDER_CONSTANT)
    # then transform im1 with the 3x3 transformation matrix
    out = cv2.warpPerspective(im1, M, (im1.shape[1],im2.shape[0]), dst=im2.copy(), borderMode=cv2.BORDER_TRANSPARENT)

    plt.imshow(out, cmap='gray')
    plt.show()
'''
def getTransform(src, dst, method='affine'):
    pts1,pts2 = feature_matching(src,dst)

    src_pts = np.float32(pts1).reshape(-1,1,2)
    dst_pts = np.float32(pts2).reshape(-1,1,2)

    if method == 'affine':
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
        #M = np.append(M, [[0,0,1]], axis=0)

    if method == 'homography':
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()

    return (M, pts1, pts2, mask)
'''
Function for Laplacian blending
'''
def laplacian_pyramid_blending(img_in1, img_in2, img_in3):
    # Write laplacian pyramid blending codes here
    A = img_in1
    B = img_in2
    C = img_in3

    #B = B[:B.shape[0], :B.shape[0]]
    #A = A[:A.shape[0], :A.shape[0]]
    #C = C[:C.shape[0], :C.shape[0]]
    xrange = range
    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in xrange(4):
        G = cv2.pyrDown(G)
        gpA.append(G)
    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in xrange(4):
        G = cv2.pyrDown(G)
        gpB.append(G)
    # generate Gaussian pyramid for C
    G = C.copy()
    gpC = [G]
    for i in xrange(4):
        G = cv2.pyrDown(G)
        gpC.append(G)
    # generate Laplacian Pyramid for A
    lpA = [gpA[3]]
    for i in xrange(3, 0, -1):

        GE = cv2.pyrUp(gpA[i])
        GE = cv2.resize(GE, (gpA[i - 1].shape[1], gpA[i - 1].shape[0]))
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)
    # generate Laplacian Pyramid for B
    lpB = [gpB[3]]
    for i in xrange(3, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        GE = cv2.resize(GE, (gpB[i - 1].shape[1], gpB[i - 1].shape[0]))
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)
    # generate Laplacian Pyramid for C
    lpC = [gpC[3]]
    for i in xrange(3, 0, -1):
        GE = cv2.pyrUp(gpC[i])
        GE = cv2.resize(GE, (gpC[i - 1].shape[1], gpC[i - 1].shape[0]))
        L = cv2.subtract(gpC[i - 1], GE)
        lpC.append(L)
    # Now add left and right halves of images in each level
    LS = []
    for la, lb,lc in zip(lpA, lpB,lpC):
        rowsa, colsa = la.shape
        rowsb, colsb = lb.shape
        rowsc, colsc = lc.shape
        ls = np.hstack((la[:, 0:int(0.45*colsa)], lb[:, int(0.45*colsb):int(0.55*colsb)],lc[:, int(0.55*colsc):]))
        LS.append(ls)
    # now reconstruct
    ls_ = LS[0]
    for i in xrange(1, 4):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.resize(ls_, (LS[i].shape[1], LS[i].shape[0]))
        ls_ = cv2.add(ls_, LS[i])

    img_out = ls_  # Blending result

    return True, img_out



# ===================================================
# ================ Perspective Warping ==============
# ===================================================
def Perspective_warping(img1, img2, img3):

    # Adding border to the center image
    img1 = cv2.copyMakeBorder(img1, 200,200,500,500, cv2.BORDER_CONSTANT)
    # Stiching image 1 and image 2
    # Transforming image 2 wrt image 1
    (M_R, pts1_R, pts2_R, mask_R) = getTransform(img2, img1, method='homography')
    # then transform image3 with the 3x3 transformation matrix
    out_R = cv2.warpPerspective(img2, M_R, (img1.shape[1], img1.shape[0]), dst=img1.copy(), borderMode=cv2.BORDER_TRANSPARENT)

    # Stiching image3 and stiched image1,2(out_R)
    # Transforming image 3 wrt image obtained in previous tranform
    (M_L, pts1_L, pts2_L, mask_L) = getTransform(img3, out_R, method='homography')
    out_L = cv2.warpPerspective(img3, M_L, (out_R.shape[1], out_R.shape[0]), dst=out_R.copy(), borderMode=cv2.BORDER_TRANSPARENT)

    # Uncomment to check the difference between the expted o/p and generated o/p
    # target_image1 = cv2.imread("example_output1.png", 0)
    # diff=RMSD(1,target_image1,out_L)
    # print(diff)

    # Write out the result
    output_name = sys.argv[5] + "output_homography.png"
    cv2.imwrite(output_name, out_L)

    return True

# ===================================================
# === Perspective Warping using Laplacian Blending===
# ===================================================

def Bonus_perspective_warping(img1, img2, img3):

    img1 = cv2.copyMakeBorder(img1, 200, 200, 500, 500, cv2.BORDER_CONSTANT)
    # Stiching image 1 and image 2
    # Transforming image 2 wrt image 1
    (M_R, pts1_R, pts2_R, mask_R) = getTransform(img2, img1, method='homography')
    # then transform image3 with the 3x3 transformation matrix without providing dst i.e. eliminating img1
    out_R = cv2.warpPerspective(img2, M_R, (img1.shape[1], img1.shape[0]),borderMode=cv2.BORDER_TRANSPARENT)

    # Stiching image3 and stiched image1,2(out_R)
    # Transforming image 3 wrt image obtained in previous tranform without providing dst i.e. eliminating tranformed image
    (M_L, pts1_L, pts2_L, mask_L) = getTransform(img3, out_R, method='homography')
    out_L = cv2.warpPerspective(img3, M_L, (out_R.shape[1], out_R.shape[0]),borderMode=cv2.BORDER_TRANSPARENT)
    # blend all three images
    succeed, output_image = laplacian_pyramid_blending(out_L,img1,out_R)

    # Write out the result
    output_name = sys.argv[5] + "output_homography_lpb.png"
    cv2.imwrite(output_name, output_image)

    return True

# ===================================================
# =============== Cynlindrical Warping ==============
# ===================================================
def Cylindrical_warping(img1, img2, img3):

    # Adding border to the center image
    img1 = cv2.copyMakeBorder(img1,50,50,300,300, cv2.BORDER_CONSTANT)
    h, w = img1.shape
    f = 450
    K1 = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])  # mock calibration matrix
    # Get the cylindrical warp of img1
    imcyl1,imcyl1Mask = cylindricalWarpImage(img1, K1)
    h, w = img2.shape
    f = 450
    K2 = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])  # mock calibration matrix
    # Get the cylindrical warp of img2 and img2. (both have same shape
    imcyl2,imcyl2Mask = cylindricalWarpImage(img2, K2)
    imcyl3,imcyl3Mask = cylindricalWarpImage(img3, K2)
    (M_R, pts1_R, pts2_R, mask_R) = getTransform(imcyl2, imcyl1)

    # then transform im1 with the 3x3 transformation matrix
    out_R = cv2.warpAffine(imcyl2, M_R, (imcyl1.shape[1], imcyl1.shape[0]), dst=imcyl1.copy(),borderMode=cv2.BORDER_TRANSPARENT)
    (M_L, pts1_L, pts2_L, mask_L) = getTransform(imcyl3, out_R)
    out_L = cv2.warpAffine(imcyl3, M_L, (out_R.shape[1], out_R.shape[0]), dst=out_R.copy(),borderMode=cv2.BORDER_TRANSPARENT)
    # masking cyl. of img1 to merge later on
    masked_imcyl1 = np.ma.masked_not_equal(imcyl1,0)
    # merging imcyl with transformed images
    output_image = masked_imcyl1+out_L

    # Uncomment to check the difference between the expted o/p and generated o/p
    # target_image1 = cv2.imread("example_output2.png", 0)
    # diff = RMSD(2, target_image1, output_image)
    # print(diff)

    # Write out the result
    output_name = sys.argv[5] + "output_cylindrical.png"
    cv2.imwrite(output_name, output_image)

    return True

# ===================================================
# === Cylindrical Warping using Laplacian Blending===
# ===================================================

def Bonus_cylindrical_warping(img1, img2, img3):

    # Write your codes here
    img1 = cv2.copyMakeBorder(img1, 50, 50, 300, 300, cv2.BORDER_CONSTANT)
    h, w = img1.shape
    f = 450
    K1 = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])  # mock calibration matrix
    # Get the cylindrical warp of img1
    imcyl1, imcyl1Mask = cylindricalWarpImage(img1, K1)
    h, w = img2.shape
    f = 450
    K2 = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])  # mock calibration matrix
    # Get the cylindrical warp of img2 and img2. (both have same shape
    imcyl2, imcyl2Mask = cylindricalWarpImage(img2, K2)
    imcyl3, imcyl3Mask = cylindricalWarpImage(img3, K2)
    (M_R, pts1_R, pts2_R, mask_R) = getTransform(imcyl2, imcyl1)

    # Then transform imcyl1 with the 3x3 transformation matrix without providing dst i.e. eliminating imcyl1
    out_R = cv2.warpAffine(imcyl2, M_R, (imcyl1.shape[1], imcyl1.shape[0]),borderMode=cv2.BORDER_TRANSPARENT)
    (M_L, pts1_L, pts2_L, mask_L) = getTransform(imcyl3, out_R)
    # Transforming imcyl3 wrt image obtained in previous tranform without providing dst i.e. eliminating tranformed image
    out_L = cv2.warpAffine(imcyl3, M_L, (out_R.shape[1], out_R.shape[0]),borderMode=cv2.BORDER_TRANSPARENT)
    # Do Blending for all 3
    succeed, output_image = laplacian_pyramid_blending(out_L, imcyl1, out_R)

    # Write out the result
    output_name = sys.argv[5] + "output_cylindrical_lpb.png"
    cv2.imwrite(output_name, output_image)

    return True

'''
This exact function will be used to evaluate your results for
Compare your result with master image in Results folder and get the difference
'''

def RMSD(questionID, target, master):
    # Get width, height, and number of channels of the master image
    master_height, master_width = master.shape[:2]
    master_channel = len(master.shape)

    # Get width, height, and number of channels of the target image
    target_height, target_width = target.shape[:2]
    target_channel = len(target.shape)

    # Validate the height, width and channels of the input image
    if (master_height != target_height or master_width != target_width or master_channel != target_channel):
        return -1
    else:
        nonZero_target = cv2.countNonZero(target)
        nonZero_master = cv2.countNonZero(master)

        if (questionID == 1):
           if (nonZero_target < 1200000):
               return -1
        elif(questionID == 2):
            if (nonZero_target < 700000):
                return -1
        else:
            return -1

        total_diff = 0.0;
        master_channels = cv2.split(master);
        target_channels = cv2.split(target);

        for i in range(0, len(master_channels), 1):
            dst = cv2.absdiff(master_channels[i], target_channels[i])
            dst = cv2.pow(dst, 2)
            mean = cv2.mean(dst)
            total_diff = total_diff + mean[0]**(1/2.0)

        return total_diff;


if __name__ == '__main__':
   question_number = -1

   # Validate the input arguments
   if (len(sys.argv) != 6):
      help_message()
      sys.exit()
   else:
      question_number = int(sys.argv[1])
      if (question_number > 4 or question_number < 1):
         print("Input parameters out of bound ...")
         sys.exit()

   input_image1 = cv2.imread(sys.argv[2], 0)
   input_image2 = cv2.imread(sys.argv[3], 0)
   input_image3 = cv2.imread(sys.argv[4], 0)

   function_launch = {
       1: Perspective_warping,
       2: Cylindrical_warping,
       3: Bonus_perspective_warping,
       4: Bonus_cylindrical_warping
   }
   # Call the function
   function_launch[question_number](input_image1, input_image2, input_image3)
