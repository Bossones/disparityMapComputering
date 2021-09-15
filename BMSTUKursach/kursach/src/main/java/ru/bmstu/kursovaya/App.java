package ru.bmstu.kursovaya;

import nu.pattern.OpenCV;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;


public class App {

    public static Mat loadImage(String filesSource) {
        return Imgcodecs.imread(filesSource);
    }

    public static void main( String[] args ) {

        System.out.println( "Hello OpenCV World!" );
        OpenCV.loadShared();
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        Loader.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        var leftImage = loadImage("../left.jpg");
        var rightImage = loadImage("../right.jpg");
        DepthMap.startProcessing(
                PreProcessing.startPreProc(leftImage),
                PreProcessing.startPreProc(rightImage)
        );
    }
}
