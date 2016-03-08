#include "som.h"

void initializemap(imgdatas &imgd, somaps &smp) {
	std::random_device rnd;
	std::mt19937_64 mt(rnd());
	std::uniform_int_distribution<> randc(0, imgd.size() - 1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < smp.size(); i++) {
		for (int j = 0; j < smp.data()->size(); j++) {
			smp.at(i).at(j)->init(imgd.at(randc(mt)));
		}
	}
}

void normalize(imgdatas &imgd){
	std::cout << "Start Normalize" << std::endl;
	Mat_<float> allvec;
	for (size_t i = 0; i != imgd.size();i++) {
		for (auto j = 0; j != imgd.at(i).descriptor.rows; j++) {
			allvec.push_back(imgd.at(i).descriptor.row(j));
		}
	}
	Mat_<int> labels(allvec.rows, 1, CV_32F);        
	Mat_<float> centroids(F, allvec.cols);
	std::cout << "Start Kmeans" << std::endl;
	kmeans(allvec, F, labels, cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 1, KMEANS_PP_CENTERS, centroids);
	std::cout << "Finish Kmeans" << std::endl;
	labels.release();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < imgd.size(); i++) {
		for (auto j = 0; j != imgd.at(i).descriptor.rows; j++) {
			std::vector<std::pair<double, int>> distlist;
			for (auto k = 0; k != centroids.rows; k++) {
				double dist = norm(imgd.at(i).descriptor.row(j), centroids.row(k));
				distlist.push_back(std::make_pair(dist, k));
			}
			std::sort(distlist.begin(), distlist.end());
			imgd.at(i).fvex[distlist.begin()->second]+=(float) 1/ (float)imgd.at(i).descriptor.rows;
		}
		imgd.at(i).descriptor.release();
	}
	std::cout << "Finish Normalize" << std::endl;
}

void showimg(combinedimg &cmb) {
	namedWindow("SOMing", CV_WINDOW_AUTOSIZE);
	Mat img;
	resize(cmb.cmbimg, img, Size(W*WIDTH / 3, H*HEIGHT / 3));
	imshow("SOMing", img);
	waitKey(0);
}
