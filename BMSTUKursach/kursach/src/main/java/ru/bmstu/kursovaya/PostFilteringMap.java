package ru.bmstu.kursovaya;

import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class PostFilteringMap {
    private Mat sparseDisparityMap;

    public PostFilteringMap(Mat sparseDisparityMap) {
        this.sparseDisparityMap = sparseDisparityMap;
    }

    public Mat startPostFilteringProcess() {
        return null;
    }
}
