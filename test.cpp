#include "convnet.h"

#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std;


int main(int argc, char** argv)
{
  struct conv_shape S;
  S.n = 5;
  S.sizes = vector<int>(S.n);
  S.kernels = vector<array<int, 4>>(S.n);
  S.sizes[0] = 0;   S.kernels[0] = {5, 5, 1, 2};
  S.sizes[1] = 0;   S.kernels[1] = {5, 5, 3, 8};
  S.sizes[2] = 0;   S.kernels[2] = {5, 5, 3, 8};
  S.sizes[3] = 0;   S.kernels[3] = {5, 5, 3, 8};
  S.sizes[4] = 2;   S.kernels[4] = {0, 0, 0, 0};
  //S.sizes[0] = 0;   S.kernels[0] = {5, 5, 1, 2};
  //S.sizes[1] = 0;   S.kernels[1] = {5, 5, 3, 4};
  //S.sizes[2] = 0;   S.kernels[2] = {5, 5, 3, 4};
  //S.sizes[3] = 256; S.kernels[3] = {0, 0, 0, 0};
  //S.sizes[4] = 2;   S.kernels[4] = {0, 0, 0, 0};
  S.sigm = false;
  S.lam = 0.00001;

  //Neuralnet N(&S, false, 0.0);
  Convnet N(&S);//, false, 0.000001);

  //backprop test
  vector<vector<vector<float>>> X_t;
  vector<vector<float>> y_t;
  vector<vector<vector<float>>> X_v;
  vector<vector<float>> y_v;
  //double M = 0.0;
  //double m = 1.0;
  //for (int i=0; i < 16384*2; ++i) {
  for (int i=0; i < 2048*2; ++i) {
    vector<vector<float>> X = vector<vector<float>>(2);
    X[0] = vector<float>(25);
    X[1] = vector<float>(25);
    vector<float> y = vector<float>(2);
    for (int j=0; j < 25; ++j) {
      X[0][j] = (double)rand()/RAND_MAX*2.0 - 1.0;
      X[1][j] = (double)rand()/RAND_MAX*2.0 - 1.0;
      double fm0 = fmod(X[0][j], 0.25);
      double fm1 = fmod(X[1][j], 0.25);
      if      ((0 < fm0 && fm0 <  0.02) || fm0 < -0.23) X[0][j] = X[0][j] + 0.02;
      else if ((0 > fm0 && fm0 > -0.02) || fm0 >  0.23) X[0][j] = X[0][j] - 0.02;
      if      ((0 < fm1 && fm1 <  0.02) || fm1 < -0.23) X[1][j] = X[1][j] + 0.02;
      else if ((0 > fm1 && fm1 > -0.02) || fm1 >  0.23) X[1][j] = X[1][j] - 0.02;
    }
    //M = max(M, fmod(abs(X[0]), 0.25));
    //m = min(m, fmod(abs(X[0]), 0.25));
    int modsum = 0;
    for (int j=0; j < 25; ++j) {
      modsum += (int)(X[0][j]*1.5)%4;
      modsum += (int)(X[1][j]*1.5)%4;
    }
    if (modsum%4 == 0) {
      y[0] = 0.0;
      y[1] = 1.0;
    }
    else {
      y[0] = 1.0;
      y[1] = 0.0;
    }
    //if (i < 16384) {
    if (i < 2048) {
      X_t.push_back(X);
      y_t.push_back(y);
    }
    else {
      X_v.push_back(X);
      y_v.push_back(y);
    }
  }
  //cout << X_v.size() << endl;
  //cout << "min: " << m << "\tmax: " << M << endl;
  //for (float* x : X_t)
  //  cout << x << " " << x[0] << endl;
  //if (argc > 1)
  //  N.train_parallel(X_t, y_t, 400, 0.25);
  //else
  //cout << "backprop test\n";
  //for (int l=1; l < S.n; ++l) {
  //  cout << "[\n";
  //  for (int i=0; i < S.sizes[l-1]; ++i) {
  //    cout << "[ ";
  //    for (int j=0; j < S.sizes[l]; ++j) {
  //      cout << N.W[l-1][i][j] << " ";
  //    }
  //    cout << "]\n";
  //  }
  //  cout << "]\n";
  //}
  N.train(X_t, y_t, X_v, y_v, 99, 0.05, 0.005);
  //cout << "backprop test\n";
  //for (int l=1; l < S.n; ++l) {
  //  cout << "[\n";
  //  for (int i=0; i < S.sizes[l-1]; ++i) {
  //    cout << "[ ";
  //    for (int j=0; j < S.sizes[l]; ++j) {
  //      cout << N.W[l-1][i][j] << " ";
  //    }
  //    cout << "]\n";
  //  }
  //  cout << "]\n";
  //}

  //float value1 = N.W[0][0][0];

  //N.save((char*)"testfile.nn");
  //Neuralnet NN = Neuralnet((char*)"testfile.nn");

  //float value2 = NN.W[0][0][0];
  //
  //if (value1 != value2)
  //  cout << "save error\n";
  //else
  //  cout << "pass\n";

  //for (unsigned int i=0; i < X_t.size(); ++i) {
  //  delete X_t[i];
  //  delete y_t[i];
  //}

  return 0;
}
