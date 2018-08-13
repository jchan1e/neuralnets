#include "neuralnet.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>

Neuralnet::Neuralnet(struct shape* S)
{
  unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
  default_random_engine generator(seed);
  normal_distribution<float> distribution(0.0,1.0);

  s.n = S->n;
  s.sizes = new int[s.n];
  for (int i=0; i < s.n; ++i)
    s.sizes[i] = S->sizes[i];

  n_layers = s.n;
  z  = new float*[s.n];
  a  = new float*[s.n];
  d  = new float*[s.n];
  b  = new float*[s.n-1];
  db = new float*[s.n-1];
  W  = new float**[s.n-1];
  dW = new float**[s.n-1];
  for (int i=0; i < s.n; ++i)
  {
    z[i] = new float[s.sizes[i]];
    a[i] = new float[s.sizes[i]];
    d[i] = new float[s.sizes[i]];
    if (i > 0)
    {
      b[i-1] = new float[s.sizes[i]];
      db[i-1] = new float[s.sizes[i]];
      for (int j=0; j < s.sizes[i]; ++j)
        b[i-1][j] = distribution(generator);
      W[i-1] = new float*[s.sizes[i-1]];
      dW[i-1] = new float*[s.sizes[i-1]];
      for (int j=0; j < s.sizes[i-1]; ++j)
      {
        W[i-1][j] = new float[s.sizes[i]];
        dW[i-1][j] = new float[s.sizes[i]];
        for (int k=0; k < s.sizes[i]; ++k)
          W[i-1][j][k] = distribution(generator);
      }
    }
  }
}

Neuralnet::~Neuralnet()
{
  //cout << "Deleting N\n";
  delete z[0];
  delete a[0];
  delete d[0];
  for (int i=1; i < n_layers; ++i)
  {
    //cout << "deleting z/a/d[" << i << "]\n";
    delete z[i];
    delete a[i];
    delete d[i];
    //delete connections[i-1];
    for (int j=0; j < s.sizes[i-1]; ++j)
    {
      //cout << "deleting W[" << i-1 << "][" << j << "] of " << s.sizes[i-1] << "\n";
      delete W[i-1][j];
      delete dW[i-1][j];
    }
    //cout << "deleting W/b[" << i-1 << "]\n";
    delete b[i-1];
    delete db[i-1];
    delete[] W[i-1];
    delete[] dW[i-1];
  }
  //cout << "deleting top level arrays\n";
  delete[] z;
  delete[] a;
  delete[] d;
  delete[] b;
  delete[] db;
  delete[] W;
  delete[] dW;
}

float Neuralnet::g(float z)
{
  return 1.0/(1.0 + exp(-z));
}
void Neuralnet::g(float* A, float* Z, int n)
{ // Activation function
  for (int i=0; i < n; ++i)
    A[i] = g(Z[i]);
}

float Neuralnet::g_prime(float z)
{
  float gz = g(z);
  return gz * (1.0 - gz);
}
void Neuralnet::g_prime(float* A, float* Z, int n)
{
  for (int i=0; i < n; ++i)
    A[i] = g_prime(Z[i]);
}

float Neuralnet::gradC(float a, float y)
{ // Derivative of the loss function
  return a - y;
}
void Neuralnet::gradC(float* D, float* A, float* Y, int n)
{
  for (int i=0; i < n; ++i)
    D[i] = gradC(A[i], Y[i]) * g_prime(A[i]);
}

void Neuralnet::forward_prop(float* X)
{
//#pragma omp parallel for
  for (int i=0; i < s.sizes[0]; ++i)
  {
    a[0][i] = X[i];
  }

//#pragma omp parallel
  for (int l=1; l< n_layers; ++l)
  {
//#pragma omp for
    for (int j=0; j < s.sizes[l]; ++j)
    {
      z[l][j] = 0.0;
      for (int i=0; i < s.sizes[l-1]; ++i)
      {
        z[l][j] += a[l-1][i] * W[l-1][i][j];
      }
      z[l][j] += b[l-1][j];
      a[l][j] = g(z[l][j]);
    }
//#pragma omp barrier
  }
}

void Neuralnet::back_prop(float* X, float* y)
{
  forward_prop(X);
  gradC(d[s.n-1], a[s.n-1], y, s.sizes[s.n-1]);

//#pragma omp parallel
//{
  for (int l=s.n-2; l >= 0; --l)
  {
    //dW[l] = outer(d[l+1], a[l]);
//#pragma omp for collapse(2)
    for (int i=0; i < s.sizes[l]; ++i)
    {
      for (int j=0; j < s.sizes[l+1]; ++j)
      {
        dW[l][i][j] = d[l+1][j] * a[l][i];
      }
    }

    //db[l] = d[l+1];
//#pragma omp single
//{
    for (int i=0; i < s.sizes[l+1]; ++i)
    {
      db[l][i] = d[l+1][i];
    }
//}

    //d[l]  = inner(W[l], d[l+1]) * g_prime(z[l]);
//#pragma omp for
    for (int i=0; i < s.sizes[l]; ++i)
    {
      d[l][i] = 0.0;
      for (int j=0; j < s.sizes[l+1]; ++j)
      {
        d[l][i] += d[l+1][j] * W[l][i][j];
      }
      d[l][i] *= g_prime(z[l][i]);
    }
//#pragma omp barrier
  }
//}
}

void Neuralnet::eval(float* X, float* y)
{
  forward_prop(X);
  for (int i=0; i < s.sizes[s.n-1]; ++i)
    y[i] = a[s.n-1][i];
}

void Neuralnet::update_weights(float eta)
{
  for (int l=0; l < s.n-1; ++l)
  {
    for (int i=0; i < s.sizes[l]; ++i)
    {
      for (int j=0; j < s.sizes[l+1]; ++j)
        W[l][i][j] -= eta*dW[l][i][j];
    }
    for (int i=0; i < s.sizes[l+1]; ++i)
      b[l][i] -= eta*db[l][i];
  }
}

void Neuralnet::train(vector<float*> X_train, vector<float*> y_train, int num_epochs, float eta)
{
  int interval = 10;
  if (num_epochs > 200)
    interval = 20;
  if (num_epochs > 1000)
    interval = 50;

  cout << "epoch: " << 0 << "\tTrain loss: " << loss(X_train, y_train) << endl;// "   \t";
  //cout << W[s.n-2][0][0] << "\t" << dW[s.n-2][0][0] << endl;

  for (int e=1; e <= num_epochs; ++e)
  {
    vector<int> index;
    for (unsigned int i=0; i < X_train.size(); ++i)
      index.push_back(i);
    random_shuffle(index.begin(), index.end());
//#pragma omp parallel for
    //for (unsigned int j=0; j < index.size(); ++j)
    for (int i : index)
    {
      //int i = index[j];
      back_prop(X_train[i], y_train[i]);
//#pragma omp critical
//      {
      update_weights(eta);
//      }
    }
    if (e%(interval) == 0 || e == num_epochs)
    {
      cout << "epoch: " << e << "\tTrain loss: " << loss(X_train, y_train) << endl;// "   \t";
      //cout << W[s.n-2][0][0] << "  \t" << dW[s.n-2][0][0] << endl;
    }
  }
}

float Neuralnet::loss(vector<float*> X_train, vector<float*> y_train)
{
  //vector<int> index;
  //for (unsigned int i=0; i < X_train.size(); ++i)
  //  index.push_back(i);
  //random_shuffle(index.begin(), index.end());

  float L= 0.0;

  //for (int i : index)
  for (unsigned int i=0; i < X_train.size(); ++i)
  {
    forward_prop(X_train[i]);
    for (int j=0; j < s.sizes[s.n-1]; ++j)
    {
      float diff = a[s.n-1][j] - y_train[i][j];
      if (0.5 * diff * diff < 0)
        cout << "wtf" << endl;
      L += 0.5 * diff * diff;
    }
  }
  return L;
}
