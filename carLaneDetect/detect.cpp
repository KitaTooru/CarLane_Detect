#include <opencv2/opencv.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat region_of_interest(Mat img, vector<Point> vertices)
{
    Mat mask = Mat::zeros(img.size(), img.type());

    Scalar ignore_mask_color;
    if (img.channels() > 1)
    {
        ignore_mask_color = Scalar(255, 255, 255);
    }
    else
    {
        ignore_mask_color = Scalar(255);
    }

    // �������ε��ĸ�����
    Point pt1(vertices[0].x, vertices[0].y);
    Point pt2(vertices[1].x, vertices[1].y);
    Point pt3(vertices[2].x, vertices[2].y);
    Point pt4(vertices[3].x, vertices[3].y);

    // ��������
    vector<vector<Point>> pts{ {pt1, pt2, pt3, pt4} };
    fillPoly(mask, pts, ignore_mask_color);

    Mat masked_image;
    bitwise_and(img, mask, masked_image);

    return masked_image;
}

void fit_lines(vector<Vec4i>& lines, Vec4d& left_line, Vec4d& right_line,
    bool& left_found, bool& right_found, const Vec4d& last_left_line)
{
    const double slope_thresh = 0.5; // ����б����ֵ
    vector<Point> left_points, right_points; // ����洢�󳵵��ߺ��ҳ����ߵ������

    // �������м�⵽���߶�
    for (size_t i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i];

        double slope = (double)(l[3] - l[1]) / (l[2] - l[0]); // ���㵱ǰ�߶ε�б��

        // ����б���ж����󳵵��߻����ҳ�����
        if (slope < -slope_thresh) // �󳵵���
        {

            left_points.push_back(Point(l[0], l[1]));
            left_points.push_back(Point(l[2], l[3])); // ���󳵵��ߵ������˵����left_points����
        }
        else if (slope > slope_thresh) // �ҳ�����
        {
            right_points.push_back(Point(l[0], l[1]));
            right_points.push_back(Point(l[2], l[3])); // ���ҳ����ߵ������˵����right_points����
        }
    }

    // ����󳵵���
    if (!left_points.empty())
    {
        fitLine(left_points, left_line, DIST_L2, 0, 0.01, 0.01); // ʹ����С���˷�����󳵵���
        left_found = true; // �����󳵵������ҵ��ı�־Ϊtrue
    }
    else
    {
        left_line = last_left_line; // ���δ�ҵ��󳵵��ߣ���ʹ����һ֡���󳵵�����Ϊ����
        left_found = false; // �����󳵵������ҵ��ı�־Ϊfalse
    }

    // ����ҳ�����
    if (!right_points.empty())
    {
        fitLine(right_points, right_line, DIST_L2, 0, 0.01, 0.01); // ʹ����С���˷�����ҳ�����
        right_found = true; // �����ҳ��������ҵ��ı�־Ϊtrue
    }
    else
    {
        right_found = false;  // �����ҳ��������ҵ��ı�־Ϊfalse
    }
}

Point intersection_point(const Vec4d& line1, const Vec4d& line2)
{
    double x1 = line1[2], y1 = line1[3], x2 = line1[2] + line1[0], y2 = line1[3] + line1[1];
    double x3 = line2[2], y3 = line2[3], x4 = line2[2] + line2[0], y4 = line2[3] + line2[1];

    double denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    double x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator;
    double y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator;

    return Point(static_cast<int>(x), static_cast<int>(y));
}

int main()
{
    VideoCapture cap("video.mp4");

    if (!cap.isOpened())
    {
        cout << "�޷�����Ƶ�ļ�" << endl;
        return -1;
    }

    Size frame_size = Size((int)cap.get(CAP_PROP_FRAME_WIDTH), (int)cap.get(CAP_PROP_FRAME_HEIGHT));
    int fourcc = VideoWriter::fourcc('D', 'I', 'V', 'X');
    int fps = (int)cap.get(CAP_PROP_FPS);

    VideoWriter writer("output.mp4", fourcc, fps, frame_size);


    Vec4d last_left_line(0, 0, 0, 0);
    bool left_found = false;

    while (true)
    {
        Mat frame;
        cap >> frame;

        if (frame.empty())
        {
            break;
        }

        // �Ҷȴ���
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // ��ֵ��
        Mat thr;
        threshold(gray, thr, 100, 255, THRESH_BINARY);

        // ��ʴ
        Mat eroded;
        Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
        erode(thr, eroded, element);

        // ����
        Mat dilated;
        dilate(eroded, dilated, element);

        // Canny��Ե���
        Mat canny_image;
        Canny(dilated, canny_image, 50, 150);

        int rows = canny_image.rows;
        int cols = canny_image.cols;

        Point left_bottom(130, 540);
        Point right_bottom(870, 540);
        Point apex_left(380, 360);
        Point apex_right(600, 360);

        vector<Point> vertices = { left_bottom, right_bottom, apex_right, apex_left };
        Mat roi_image = region_of_interest(canny_image, vertices);

        // ����任���ֱ��
        vector<Vec4i> lines;
        HoughLinesP(roi_image, lines, 2, CV_PI / 180, 15, 40, 10);

        // ������ҳ�����
        Vec4d left_line, right_line;
        bool right_found;
        fit_lines(lines, left_line, right_line, left_found, right_found, last_left_line);

        if (left_found)
        {
            last_left_line = left_line;
        }

        // ���㽻��
        Point intersection = intersection_point(left_line, right_line);

        // ��ԭͼ�ϻ�����ϵ�ֱ�ߺͽ������µĲ���
        double left_slope = last_left_line[1] / last_left_line[0];
        double left_intercept = last_left_line[3] - left_slope * last_left_line[2];
        Point left_pt1(0, left_intercept);
        Point left_pt2(cols, left_slope * cols + left_intercept);
        line(frame, left_pt1, intersection, Scalar(0, 255, 255), 4, LINE_AA);

        if (right_found)
        {
            double right_slope = right_line[1] / right_line[0];
            double right_intercept = right_line[3] - right_slope * right_line[2];

            // ���㽻���y����
            int intersection_y = static_cast<int>(left_slope * intersection.x + left_intercept);

            // �����Ҳ�ֱ��
            Point right_pt1((intersection_y - right_intercept) / right_slope, intersection_y);
            Point right_pt2((rows - right_intercept) / right_slope, rows);
            line(frame, right_pt1, right_pt2, Scalar(0, 255, 255), 4, LINE_AA);
        }

        imshow("Detected Lines", frame);
        writer.write(frame);  // ��֡д����Ƶ�ļ�

        if (waitKey(1) == 27)
        {
            break;
        }
    }

    cap.release();
    writer.release();
    destroyAllWindows();

    return 0;
}
