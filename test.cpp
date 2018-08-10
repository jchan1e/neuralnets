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
  float X[2];
  float y[2];
  vector<float*> X_t;
  vector<float*> y_t;
  for (int i=0; i < 360; ++i) {
    X[0] = rand()/RAND_MAX*6.0;
    X[1] = rand()/RAND_MAX*6.0;
    if ((int)X[0]%2 != (int)X[1]%2) {
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
  N.train(X_t, y_t, 2000);
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

  delete S.sizes;

  return 0;
}
