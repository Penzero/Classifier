#pragma once

#include <vector>
#include <string>

using std::vector, std::string;

class Classifier
{
    public:
        virtual void fit(const T& trainImages, const vector<int>& trainLabels) = 0;
        virtual vector<int> predict(const T&) = 0;
        virtual bool save(const string& filepath) = 0;
        virtual bool load(const string& filepath) = 0;
        virtual double eval(const T&, const vector<int>& trainLabels) = 0;

};


class T {
public:
    vector<vector<double>> X;

    T(int n, int d) : X(n, vector<double>(d)) {}

    size_t numSamples() const {
        return X.size();
    }

    size_t numFeatures() const {
        return X[0].size();
    }

    vector<double>& operator[](size_t idx) {
        return X[idx];
    }

    const vector<double>& operator[](size_t idx) const {
        return X[idx];
    }
};