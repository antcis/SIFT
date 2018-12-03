# SIFT
Implementation of SIFT Method (Feature Extraction Technique and Images Matching)



## Getting Started

This Method, in Python, is used to Detect keypoints and extract local descriptors which are invariant from the input images.
The second step is to Match the descriptors between the two images.
The last step would be to use that code for panorama stitching (homography matrix).

## Explaining the tests

We have made tests on several images : 
 * comparison of the keypoints between an image and the same image but resized (trouverPointsCles function)
 * comparison of the keypoints between an image and the same image but rotated 90 degrees clockwise (trouverPointsCles function)
 * matching of keypoints between two cropped parts of an image (trouverDescripteurs function)

## Authors

* **Antonia FRANCIS** (2018)
* **Guillaume VERGNOLLE** (2018)

## Acknowledgments

This project was implemented thanks to the following article.
* Distinctive Image Features from Scale-Invariant Keypoints - David G. Lowe - January 5, 2004
Lowe, D.G. International Journal of Computer Vision (2004) 60: 91. https://doi.org/10.1023/B:VISI.0000029664.99615.94 
* Ives Rey Otero, and Mauricio Delbracio, Anatomy of the SIFT Method, Image Processing On Line, 4 (2014), pp. 370–396. 
