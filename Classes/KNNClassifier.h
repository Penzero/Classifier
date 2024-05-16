#pragma once

#include <Classifier.h>
#include <vector>
#include <string>

class KNNClassifier : public Classifier
{
    private:
        T trainImages;
        vector<int> trainLabels;
        int k;
        
        double euclideanDistance(const vector<double>& a, const vector<double>& b) const;

        int mode(const vector<int>& v);

    public:
        KNNClassifier(int k);

        void fit(const T& trainImages, const vector<int>& trainLabels) override;
        vector<int> predict(const T& testImages) override;
        bool save(const string& filepath) override;
        bool load(const string& filepath) override;
        double eval(const T& testImages, const vector<int>& trainLabels) override;
