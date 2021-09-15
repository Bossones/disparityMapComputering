package ru.bmstu.kursovaya;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class PreProcessing {

    /**
     * Препроцессинг изображения.
     * @param image - входное изображение.
     * @return обработанное изображение перед процессом построения карты глубины.
     */
    public static Mat startPreProc(Mat image) {
        var grayImage = new Mat();
        var gaussianBlurImage = new Mat();
        var medianBlurImage = new Mat();

        Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(grayImage, gaussianBlurImage, new Size(3, 3), 0);
        Imgproc.medianBlur(gaussianBlurImage, medianBlurImage, 3);
        return medianBlurImage;
    }
}
