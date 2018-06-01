#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


int main( int argc, char* argv[] ){
    if(argc<4){
        cout<<"./AffineTransformationEstimation FullScene.jpg CropPart.jpg WarpedOut.jpg"<<endl;
        return 1;
    }


    string path_img1=argv[1];
    string path_img2=argv[2];


    Mat img1 = imread( path_img1, IMREAD_GRAYSCALE );
    Mat img2 = imread( path_img2 , IMREAD_GRAYSCALE );

   /* //Needed for visual debbuging
    resize(img1, img1, Size(), 0.1, 0.1, CV_INTER_AREA);
    resize(img2, img2, Size(), 0.1, 0.1, CV_INTER_AREA);
    imshow("Scene", img1);
    imshow("Objet",img2);*/

    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;
    SurfFeatureDetector detector(minHessian);
    vector<KeyPoint> keypoints1, keypoints2;
    detector.detect(img1, keypoints1);
    detector.detect(img2, keypoints2);

    //-- Step 2: Compute the descriptors
    SurfDescriptorExtractor extractor;
    Mat descriptors1, descriptors2;
    extractor.compute(img1, keypoints1, descriptors1);
    extractor.compute(img2, keypoints2, descriptors2);


    //-- Step 3: Matching descriptor vectors with a FLANN based matcher
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.2f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++){
        if (knn_matches[i].size() > 1 && knn_matches[i][0].distance / knn_matches[i][1].distance <= ratio_thresh){
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    //-- Localize the cropped part of the scene
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( size_t i = 0; i < good_matches.size(); i++ ){
        scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
        obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
    }


    //-- Step 3: Estimation of the affine transformation
    Mat T = estimateRigidTransform(obj, scene, true);

    //-- Step 4: Apply the warping
    Mat img1full = imread( path_img1 );
    Mat img2full = imread( path_img2 );
    Mat img1crop ;
    warpAffine( img1full, img1crop, T, img2full.size() );
    imwrite(argv[3],img1crop);


    //Needed for visual debbuging
    //-- Draw matches
   /* Mat img_matches;
    drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );*/

    //Mat H = findHomography( obj, scene, RANSAC );
  /*  Mat T = estimateRigidTransform(scene, obj, false);

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)img2.cols, 0 );
    obj_corners[2] = Point2f( (float)img2.cols, (float)img2.rows );
    obj_corners[3] = Point2f( 0, (float)img2.rows );
    std::vector<Point2f> scene_corners(4);
    //perspectiveTransform( obj_corners, scene_corners, H);
    transform(obj_corners, scene_corners, T);
    //-- Draw lines between the corners (the mapped object in the scene  )
    line( img_matches, scene_corners[0] , scene_corners[1] , Scalar(0, 255, 0),  4 );
    line( img_matches, scene_corners[1] , scene_corners[2] , Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[2] , scene_corners[3] , Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[3] , scene_corners[0] , Scalar( 0, 255, 0), 4 );*/
    //-- Show detected matches
    //imshow("Good Matches & Object detection", img_matches );

   /* resize(img1full, img1full, Size(), 0.1, 0.1, CV_INTER_AREA);
    warpAffine( img1full, img1crop, T, img2.size() );
    imshow("Croped Image", img1crop );
    waitKey();*/




    return 0;
}

