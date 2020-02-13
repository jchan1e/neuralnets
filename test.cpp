#include "neuralnet.h"

#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std;


int main(int argc, char** argv)
{
  struct shape S;
  S.n = 5;
  //S.sizes = new int[S.n];
  S.sizes[0] = 4;
  S.sizes[1] = 50;
  S.sizes[2] = 50;
  S.sizes[3] = 50;
  S.sizes[4] = 2;
  S.sigm = false;
  S.lam = 0.000001;

  //Neuralnet N(&S, false, 0.0);
  Neuralnet N(&S);//, false, 0.000001);

  //backprop test
  vector<float*> X_t;
  vector<float*> y_t;
  vector<float*> X_v;
  vector<float*> y_v;
  double M = 0.0;
  double m = 1.0;
  //for (int i=0; i < 16384*2; ++i) {
  for (int i=0; i < 2048*2; ++i) {
    float* X = new float[4];
    float* y = new float[2];
    X[0] = (double)rand()/RAND_MAX*2.0 - 1.0;
    X[1] = (double)rand()/RAND_MAX*2.0 - 1.0;
    X[2] = (double)rand()/RAND_MAX*2.0 - 1.0;
    X[3] = (double)rand()/RAND_MAX*2.0 - 1.0;
    double fm0 = fmod(X[0], 0.25);
    double fm1 = fmod(X[1], 0.25);
    double fm2 = fmod(X[2], 0.25);
    double fm3 = fmod(X[3], 0.25);
    if      ((0 < fm0 && fm0 <  0.02) || fm0 < -0.23) X[0] = X[0] + 0.02;
    else if ((0 > fm0 && fm0 > -0.02) || fm0 >  0.23) X[0] = X[0] - 0.02;
    if      ((0 < fm1 && fm1 <  0.02) || fm1 < -0.23) X[1] = X[1] + 0.02;
    else if ((0 > fm1 && fm1 > -0.02) || fm1 >  0.23) X[1] = X[1] - 0.02;
    if      ((0 < fm2 && fm2 <  0.02) || fm2 < -0.23) X[2] = X[2] + 0.02;
    else if ((0 > fm2 && fm2 > -0.02) || fm2 >  0.23) X[2] = X[2] - 0.02;
    if      ((0 < fm3 && fm3 <  0.02) || fm3 < -0.23) X[3] = X[3] + 0.02;
    else if ((0 > fm3 && fm3 > -0.02) || fm3 >  0.23) X[3] = X[3] - 0.02;
    M = max(M, fmod(abs(X[0]), 0.25));
    m = min(m, fmod(abs(X[0]), 0.25));
    if (((int)(X[0]*1.5)%2 + (int)(X[1]*1.5)%2 + (int)(X[2]*1.5)%2 + (int)(X[3]*1.5)%2)%2 == 0) {
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
  N.train(X_t, y_t, X_v, y_v, 800, 0.05, 0.005);
  cout << "backprop test\n";
  for (int l=1; l < S.n; ++l) {
    cout << "[\n";
    for (int i=0; i < S.sizes[l-1]; ++i) {
      cout << "[ ";
      for (int j=0; j < S.sizes[l]; ++j) {
        cout << N.W[l-1][i][j] << " ";
      }
      cout << "]\n";
    }
    cout << "]\n";
  }

  //float value1 = N.W[0][0][0];

  //N.save((char*)"testfile.nn");
  //Neuralnet NN = Neuralnet((char*)"testfile.nn");

  //float value2 = NN.W[0][0][0];
  //
  //if (value1 != value2)
  //  cout << "save error\n";
  //else
  //  cout << "pass\n";

  for (unsigned int i=0; i < X_t.size(); ++i) {
    delete X_t[i];
    delete y_t[i];
  }

  return 0;
}
