#include <KNNClassifier.h>

#include <cmath>
#include <algorithm>
#include <fstream>
#include <unordered_map>
#include <iostream>
#include <stdexcept>

using std::vector, std::string, std::ifstream, std::ofstream, std::cout, std::endl, std::runtime_error;

KNNClassifier::KNNClassifier(int k) : k(k) {}

void KNNClassifier::fit(const T& trainImages, const vector<int>& trainLabels)
{
    this->trainImages = trainImages;
    this->trainLabels = trainLabels;
}

vector<int> KNNClassifier::predict(const T& testImages) {
    vector<int> predictions(testImages.numSamples());

    for (size_t i = 0; i < testImages.numSamples(); ++i) {
        vector<std::pair<double, int>> distances;

        for (size_t j = 0; j < trainImages.numSamples(); ++j) {
            double dist = euclideanDistance(testImages[i], trainImages[j]);
            distances.push_back(std::make_pair(dist, trainLabels[j]));
        }

        sort(distances.begin(), distances.end());

        vector<int> kNearestLabels;
        for (int m = 0; m < k; ++m) {
            kNearestLabels.push_back(distances[m].second);
        }

        predictions[i] = mode(kNearestLabels);
    }

    return predictions;
}

