#include "LoadDnnModel.h"

const char* class_names[] = { "package" };

LoadDnnModel::LoadDnnModel()
{

}

LoadDnnModel::~LoadDnnModel()
{

}

tuple<Array, Shape, cv::Mat> LoadDnnModel::Read_Image(const string& path, int size)
{
    auto image = cv::imread(path);
    assert(!image.empty() && image.channels() == 3);
    cv::resize(image, image, { size, size });
    Shape shape = { 1, image.channels(), image.rows, image.cols };
    cv::Mat nchw = cv::dnn::blobFromImage(image, 1.0, {}, {}, true) / 255.f;
    Array array(nchw.ptr<float>(), nchw.ptr<float>() + nchw.total());
    return { array, shape, image };
}

void LoadDnnModel::LoadOnnx()
{
	//model_path 형식맞춰주기
	std::string sPath = model_path;
	wchar_t* wPath = new wchar_t[sPath.length() + 1];
	std::copy(sPath.begin(), sPath.end(), wPath);
	wPath[sPath.length()] = 0;

	session_ = new Ort::Session(env, wPath, Ort::SessionOptions{ nullptr });

	delete[] wPath;

	//test
	tuple<Array, Shape, cv::Mat> temp = Read_Image(image_path, 640);
	pair<Array, Shape> temp2 = process_image(session_, std::get<0>(temp), std::get<1>(temp));
	auto image = cv::imread(image_path);
	cv::resize(image, image, { 640, 640 });
	display_image(image, temp2.first, temp2.second);


}
pair<Array, Shape> LoadDnnModel::process_image(Ort::Session* session, Array& array, Shape shape)
{
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	input_tensor_ = Ort::Value::CreateTensor<float>(
		memory_info, array.data(), array.size(), shape.data(), shape.size());

	const char* input_names[] = { "images" };
	const char* output_names[] = { "output" };
	auto output = session->Run({}, input_names, &input_tensor_, 1, output_names, 1);
	shape = output[0].GetTensorTypeAndShapeInfo().GetShape();
	auto ptr = output[0].GetTensorData<float>();
	return { Array(ptr, ptr + shape[0] * shape[1]), shape };
}
void LoadDnnModel::display_image(cv::Mat image, const Array& output, const Shape& shape)
{
	for (size_t i = 0; i < shape[0]; ++i)
	{
		auto ptr = output.data() + i * shape[1];
		int x = ptr[1], y = ptr[2], w = ptr[3] - x, h = ptr[4] - y, c = ptr[5];
		auto color = CV_RGB(0, 255, 0);
		auto name = std::string(class_names[c]) + ":" + std::to_string(int(ptr[6] * 100)) + "%";
		cv::rectangle(image, { x, y, w, h }, color);
		cv::putText(image, name, { x, y }, cv::FONT_HERSHEY_DUPLEX, 1, color);
	}

	cv::imshow("YOLOv7 Output", image);
	cv::waitKey(0);
}

void LoadDnnModel::drawPred(float conf, int left, int top, int right, int bottom, cv::Mat& frame, int classid)   // Draw the predicted bounding box
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 2);

    //Get the label for the class name and its confidence
    string label = cv::format("%.2f", conf);
    label = /*this->class_names[classid]*/"sample";

    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    //rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
    putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 0, 0), 1);
}