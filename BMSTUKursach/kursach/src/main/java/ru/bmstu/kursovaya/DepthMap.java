package ru.bmstu.kursovaya;

import org.opencv.calib3d.StereoSGBM;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

public class DepthMap {

    public static void startProcessing(Mat leftImage, Mat rightImage) {

        var rectification = new Rectification(leftImage, rightImage);
        Mat[] rectifiedImages;
        rectifiedImages = rectification.startRectification();

        var block_size = 11;
        var min_disp = -128;
        var max_disp = 128;
        var num_disp = max_disp - min_disp;
        var uniquenessRatio = 5;
        var speckleWindowsSize = 200;
        var speckleRange = 2;
        var disp12MaxDiff = 0;

        var stereo = StereoSGBM.create(
                min_disp,
                num_disp,
                block_size,
                8 * block_size * block_size,
                32 * block_size * block_size,
                disp12MaxDiff,
                0,
                uniquenessRatio,
                speckleWindowsSize,
                speckleRange
        );

        var disparity_SGBM = new Mat();
        var normalizedDisparityMap = new Mat();
        stereo.compute(rectifiedImages[0], rectifiedImages[1], disparity_SGBM);
        Core.normalize(disparity_SGBM, normalizedDisparityMap, 255, 0, Core.NORM_MINMAX);

        Imgcodecs.imwrite("../disparityMap/disparity_map.png", disparity_SGBM);
        Imgcodecs.imwrite("../disparityMap/normalized_disparity_map.png", normalizedDisparityMap);
    }
}
