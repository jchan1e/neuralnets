#include "neuralnet.h"

#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std;


int main(int argc, char** argv)
{
  struct shape S;
  S.n = 5;
  S.sizes = new int[S.n];
  S.sizes[0] = 2;
  S.sizes[1] = 30;
  S.sizes[2] = 80;
  S.sizes[3] = 30;
  S.sizes[4] = 2;

  Neuralnet N(&S);

  //backprop test
  vector<float*> X_t;
  vector<float*> y_t;
  vector<float*> X_v;
  vector<float*> y_v;
  double M = 0.0;
  double m = 1.0;
  for (int i=0; i < 720*2; ++i) {
    float* X = new float[2];
    float* y = new float[2];
    X[0] = (double)rand()/RAND_MAX*2.0 - 1.0;
    X[1] = (double)rand()/RAND_MAX*2.0 - 1.0;
    double fm0 = fmod(X[0], 0.25);
    double fm1 = fmod(X[1], 0.25);
    if      ((0 < fm0 && fm0 < 0.01) || fm0 < -0.24) X[0] = X[0] + 0.01;
    else if ((0 > fm0 && fm0 > -0.01) || fm0 > 0.24) X[0] = X[0] - 0.01;
    if      ((0 < fm1 && fm1 < 0.01) || fm1 < -0.24) X[1] = X[1] + 0.01;
    else if ((0 > fm1 && fm1 > -0.01) || fm1 > 0.24) X[1] = X[1] - 0.01;
    M = max(M, fmod(abs(X[0]), 0.25));
    m = min(m, fmod(abs(X[0]), 0.25));
    if ((int)(X[0]*2)%2 == (int)(X[1]*2)%2) {
      y[0] = 0.0;
      y[1] = 1.0;
    }
    else {
      y[0] = 1.0;
      y[1] = 0.0;
    }
    if (i < 720) {
      X_t.push_back(X);
      y_t.push_back(y);
    }
    else {
      X_v.push_back(X);
      y_v.push_back(y);
    }
  }
  cout << X_v.size()/sizeof(float*) << endl;
  //cout << "min: " << m << "\tmax: " << M << endl;
  //for (float* x : X_t)
  //  cout << x << " " << x[0] << endl;
  if (argc > 1)
    N.train_parallel(X_t, y_t, 400, 0.25);
  else
    N.train(X_t, y_t, X_v, y_v, 400, 0.25);
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

  float value1 = N.W[0][0][0];

  N.save((char*)"testfile.nn");
  Neuralnet NN = Neuralnet((char*)"testfile.nn");

  float value2 = NN.W[0][0][0];
  
  if (value1 != value2)
    cout << "save error\n";
  else
    cout << "pass\n";

  for (unsigned int i=0; i < X_t.size(); ++i) {
    delete X_t[i];
    delete y_t[i];
  }

  delete S.sizes;

  return 0;
}
