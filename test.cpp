#include "neuralnet.h"

#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std;


int main()
{
  struct shape S;
  S.n = 5;
  S.sizes = new int[S.n];
  S.sizes[0] = 2;
  S.sizes[1] = 15;
  S.sizes[2] = 15;
  S.sizes[3] = 15;
  S.sizes[4] = 2;

  Neuralnet N(&S);

  //backprop test
  vector<float*> X_t;
  vector<float*> y_t;
  for (int i=0; i < 720; ++i) {
    float* X = new float[2];
    float* y = new float[2];
    X[0] = (float)rand()/RAND_MAX*2.0 - 1.0;
    X[1] = (float)rand()/RAND_MAX*2.0 - 1.0;
    if (fmod(X[0], 1.0) < 0.05)      X[0] += (float)rand()/RAND_MAX*0.9+0.05;
    else if (fmod(X[0], 1.0) > 0.95) X[0] -= (float)rand()/RAND_MAX*0.9+0.05;
    if (fmod(X[1], 1.0) < 0.05)      X[1] += (float)rand()/RAND_MAX*0.9+0.05;
    else if (fmod(X[1], 1.0) > 0.95) X[1] -= (float)rand()/RAND_MAX*0.9+0.05;
    if ((int)(X[0]*4)%2 != (int)(X[1]*4)%2) {
      y[0] = 0.0;
      y[1] = 1.0;
    }
    else {
      y[0] = 1.0;
      y[1] = 0.0;
    }
    X_t.push_back(X);
    y_t.push_back(y);
  }
  //for (float* x : X_t)
  //  cout << x << " " << x[0] << endl;
  N.train(X_t, y_t, 4000, 0.25);
  //cout << "backprop test\n";
  for (int l=1; l < S.n; ++l) {
    //cout << "[\n";
    for (int i=0; i < S.sizes[l-1]; ++i) {
      //cout << "[ ";
      for (int j=0; j < S.sizes[l]; ++j) {
        //cout << N.W[l-1][i][j] << " ";
      }
      //cout << "]\n";
    }
    //cout << "]\n";
  }

  for (unsigned int i=0; i < X_t.size(); ++i) {
    delete X_t[i];
    delete y_t[i];
  }

  delete S.sizes;

  return 0;
}
