#include "som.h"

imgdata::imgdata()
{
}

imgdata::~imgdata()
{
}

void imgdata::loadimg(const std::string & filepath)
{
	Mat src_img = imread(filepath);
	resize(src_img, this->img, Size(WIDTH, HEIGHT), CV_INTER_CUBIC);
	normalize(src_img, src_img, 0, 255, NORM_MINMAX);
	cvtColor(src_img, src_img, CV_RGB2HLS);
	std::vector<Mat> chn;
	split(src_img, chn);
	Mat_<float> tmpdescriptor;
	const int step = 10; 
	std::vector<KeyPoint> keypoint;
	for (int i = step; i<src_img.rows - step; i += step){
		for (int j = step; j<src_img.cols - step; j += step){
			keypoint.push_back(KeyPoint(float(j), float(i), float(step)));
		}
	}
	auto feature = BRISK::create();
	std::vector<Mat> chndescriptor;
	for (auto &i:chn) {
		chndescriptor.emplace_back();
		//Mat tmp;
		feature->compute(i, keypoint, chndescriptor.back());
		//tmp.convertTo(chndescriptor.back(), CV_32F);
	}	
	Mat tmpv(keypoint.size(), chndescriptor.data()->cols * chndescriptor.size(), CV_32F);
	hconcat(chndescriptor, tmpv);
	tmpv.convertTo(this->descriptor, CV_32F);
}

const std::pair<int,int> imgdata::findnear(somaps &smp) {
	using cordinate = std::pair<int, int>;
	std::vector<cordinate> ilist;
	std::array<std::pair<cordinate,float>, HW> tmp;
	auto forsort = [](std::pair< cordinate, float> &x, std::pair< cordinate, float> &y) -> bool {
		if (x.second > y.second)return false; 
		if (x.second < y.second)return true; 
		return false; 
	};
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < smp.size(); i++) {
		for (int j = 0; j < smp.data()->size(); j++) {
			tmp.at(i*W + j) = { {j,i},this->getDistance(*smp.at(i).at(j)) };
		}
	}
	std::sort(tmp.begin(),tmp.end(), forsort);
	for (auto &i:tmp) {
		if (tmp.front().second != i.second)break;
		ilist.push_back(i.first);
	}
	std::uniform_int_distribution<> randin(0, (ilist.size() - 1));
	return ilist.at(randin(this->mt));
}
