package ru.bmstu.kursovaya;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.FlannBasedMatcher;
import org.opencv.imgcodecs.Imgcodecs;

import static org.opencv.imgproc.Imgproc.*;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Процесс ректификации левого и правого изображений.
 */
public class Rectification {
    private Mat leftImage;
    private Mat rightImage;
    private Mat leftDescriptor;
    private Mat rightDescriptor;
    private MatOfKeyPoint leftKeyPoints;
    private MatOfKeyPoint rightKeyPoints;
    private Mat lines1;
    private Mat lines2;

    public Rectification() {
        leftImage = null;
        rightImage = null;
        leftKeyPoints = new MatOfKeyPoint();
        rightKeyPoints = new MatOfKeyPoint();
        leftDescriptor = new Mat();
        rightDescriptor = new Mat();
        lines1 = new Mat();
        lines2 = new Mat();
    }

    public Rectification(Mat leftImage, Mat rightImage) {
        this();
        this.rightImage = rightImage;
        this.leftImage = leftImage;
    }

    public Mat[] startRectification() throws RuntimeException {
        if (Objects.nonNull(rightImage)
                && Objects.nonNull(leftImage)) {

            var matches = new ArrayList<MatOfDMatch>();
            var outputMask = new Mat();
            ArrayList<List<Point>> points;
            Mat[] params;
            Mat H1 = new Mat();
            Mat H2 = new Mat();
            var img_left_rectified = new Mat();
            var img_right_rectified = new Mat();

            detectKeyPointsAndDescriptors();
            points = keyPointsMatch(matches);
            params = calculateFundMatrix(points.get(0), points.get(1), outputMask);
            uncalibratedRectification(params[1], params[2], params[0], rightImage.size(), H1, H2);
            warpPerspective(leftImage, img_left_rectified, H1, leftImage.size());
            warpPerspective(rightImage, img_right_rectified, H2, rightImage.size());

            Imgcodecs.imwrite("../rectified/left_rectified.png", img_left_rectified);
            Imgcodecs.imwrite("../rectified/right_rectified.png", img_right_rectified);

            return new Mat[] {img_left_rectified, img_right_rectified};
        } else {
            throw new RuntimeException("Нет входных изображений");
        }
    }

    public Mat[] startRectification(Mat leftImage, Mat rightImage) {
        this.leftImage = leftImage;
        this.rightImage = rightImage;
        return startRectification();
    }

    /**
     * Детектирование ключевых точек левого и правого изображений.
     * Создание дескрипторов для ключевых точек левого и правого изображений.
     */
    private void detectKeyPointsAndDescriptors() {

        var SIFT = org.opencv.features2d.SIFT.create();

        SIFT.detectAndCompute(leftImage, new Mat(), leftKeyPoints, leftDescriptor);
        SIFT.detectAndCompute(rightImage, new Mat(), rightKeyPoints, rightDescriptor);

      //Показать ключевые точки левого и правого изображений
        var leftImageKeyPoints = new Mat();
        var rightImageKeyPoints = new Mat();

        Features2d.drawKeypoints(
                leftImage,
                leftKeyPoints,
                leftImageKeyPoints,
                new Scalar(0, 0, 255),
                Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        );
        Features2d.drawKeypoints(
                rightImage,
                rightKeyPoints,
                rightImageKeyPoints,
                new Scalar(0, 255, 0),
                Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        );

        Imgcodecs.imwrite("../keyPoints/left_keyPoints_image.png", leftImageKeyPoints);
        Imgcodecs.imwrite("../keyPoints/right_keyPoints_image.png", rightImageKeyPoints);
    }

    /**
     * Процесс матчинга ключевых точек левого и правого изображений
     * с использованием FLANN матчера.
     * @param matches - список точек совпадений
     * @return возвращает массив заматченных точек.
     */
    private ArrayList<List<Point>> keyPointsMatch(List<MatOfDMatch> matches) {

        var flannMatcher = new FlannBasedMatcher();

        flannMatcher.knnMatch(leftDescriptor, rightDescriptor, matches, 2);

        List<MatOfByte> matchesMask = new ArrayList<>(matches.size());
        for (int i = 0; i < matches.size(); i++) {
            matchesMask.add(new MatOfByte(new byte[] {0, 0}));
        }

        List<DMatch> good = new ArrayList<>();
        List<Point> pts1 = new ArrayList<>();
        List<Point> pts2 = new ArrayList<>();

        for (int i = 0; i < matches.size(); i++) {
            var m = matches.get(i).toArray()[0];
            var n = matches.get(i).toArray()[1];

            if (m.distance < 0.7 * n.distance) {
                matchesMask.set(i, new MatOfByte(new byte[] {1, 0}));
                good.add(m);
                pts2.add(rightKeyPoints.toArray()[m.trainIdx].pt);
                pts1.add(leftKeyPoints.toArray()[n.queryIdx].pt);
            }
        }

        var matchesKnn = new Mat();
        Features2d.drawMatchesKnn(
                leftImage,
                leftKeyPoints,
                rightImage,
                rightKeyPoints,
                matches.subList(300, 500),
                matchesKnn
        );

        Imgcodecs.imwrite("../keyPointMatches/matches_img.png", matchesKnn);

        var outputList = new ArrayList<List<Point>>();
        outputList.add(new ArrayList<>(pts1));
        outputList.add(new ArrayList<>(pts2));
        return outputList;
    }

    /**
     * Расчет фундаментальной матрицы для матченных точек с левого и правого изображений.
     * Расчет эпиполярных линий.
     * @param pts1 - ключевые точки левого изображения.
     * @param pts2 - ключевые точки правого изображения.
     * @param outputMask - выходная маска.
     * @return рассчитанная фундаментальная матрица.
     */
    private Mat[] calculateFundMatrix(List<Point> pts1, List<Point> pts2, Mat outputMask) {
        MatOfPoint2f points1 = new MatOfPoint2f(copyToArray(pts1));
        MatOfPoint2f points2 = new MatOfPoint2f(copyToArray(pts2));
        Mat fundamentalMatrix;

        fundamentalMatrix = Calib3d.findFundamentalMat(
                points1,
                points2,
                Calib3d.FM_RANSAC,
                3,
                0.99,
                outputMask
        );

        List<Point> modifiedPoints1 = new ArrayList<>();
        List<Point> modifiedPoints2 = new ArrayList<>();

        for (int i = 0; i < outputMask.rows(); i++) {
            if ((int)outputMask.get(i, 0)[0] == 1) {
                modifiedPoints1.add(pts1.get(i));
                modifiedPoints2.add(pts2.get(i));
            }
        }

        Mat pointMat1 = new MatOfPoint2f(copyToArray(modifiedPoints1));
        Mat pointMat2 = new MatOfPoint2f(copyToArray(modifiedPoints2));

        Calib3d.computeCorrespondEpilines(pointMat2, 2, fundamentalMatrix, lines1);

        Calib3d.computeCorrespondEpilines(pointMat1, 1, fundamentalMatrix, lines2);

        var img1 = drawLines(leftImage, rightImage, lines1, pointMat1, pointMat2);
        var img2 = drawLines(rightImage,  leftImage, lines2, pointMat1, pointMat2);

        Imgcodecs.imwrite("../epilines/epilines_img_left.png", img1);
        Imgcodecs.imwrite("../epilines/epilines_img_right.png", img2);

        return new Mat[] {fundamentalMatrix, pointMat1, pointMat2};
    }

    private static Point[] copyToArray(List<Point> pointsList) {
        Point[] points = new Point[pointsList.size()];
        for (int i = 0; i < pointsList.size(); i++) {
            points[i] = new Point(pointsList.get(i).x, pointsList.get(i).y);
        }
        return points;
    }

    private static Mat drawLines(Mat imgLeftSrc, Mat imgRightSrc, Mat lines, Mat pointMat1, Mat pointMat2) {
        var img1Color = imgLeftSrc.clone();
        var img2Color = imgRightSrc.clone();

        var height = img1Color.size().height;
        var width = img1Color.size().width;

        Point xy0 = new Point();
        Point xy1 = new Point();
        var random = ThreadLocalRandom.current();
        double[] color = new double[3];

        for (int i = 0, k = 0, j = 0; i < lines.rows() && k < pointMat1.rows() && j < pointMat2.rows(); i++, k++, j++) {
            for (int col = 0; col < color.length; col++) {
                color[col] = random.nextInt(255);
            }
            xy0.set(new double[]{0, -lines.get(i, 0)[2]/lines.get(i, 0)[1]});
            xy1.set(new double[]{width, -(lines.get(i, 0)[2] + lines.get(i, 0)[0] * width)/lines.get(i, 0)[1]});

            line(img1Color, xy0, xy1, new Scalar(color), 1);
            circle(img1Color, new Point(pointMat1.get(k, 0)), 5, new Scalar(color), -1);
            circle(img2Color, new Point(pointMat2.get(j, 0)), 5, new Scalar(color), -1);
        }

        return img1Color;
    }

    /**
     * Ректификация некалиброванных изображений.
     * @param points1 - ключевые точки левого изображения.
     * @param points2 - ключевые точки правого изображения.
     * @param fundamentalMatrix - фундаментальная матрица.
     * @param imgSize - размер изображения.
     * @param H1 - выходной параметр 1 для дальнейшей ректификации.
     * @param H2 - выходной параметр 2 для дальнейшей ректификации.
     * @return true в случае успешной ректификации, иначе - false.
     */
    private boolean uncalibratedRectification(Mat points1,
                                              Mat points2,
                                              Mat fundamentalMatrix,
                                              Size imgSize,
                                              Mat H1,
                                              Mat H2) {
        return Calib3d.stereoRectifyUncalibrated(points1, points2, fundamentalMatrix, imgSize, H1, H2);
    }
}
