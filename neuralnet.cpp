#include "neuralnet.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <sys/types.h>
#include <unistd.h>

Neuralnet::Neuralnet(struct shape* S) : g(S->sigm?&sig:&relu), g_prime(S->sigm?&sig_prime:&relu_prime)
{
  unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
  default_random_engine generator(seed);

  Lambda = S->lam; // lambda ratio to alpha
  s.lam = S->lam;
  s.n = S->n;
  //s.sizes = new int[s.n];
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
      normal_distribution<float> distribution(0.0,1.0/s.sizes[i]);
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

Neuralnet::Neuralnet(char* filename) : g(), g_prime()
{
  ifstream f;
  f.open(filename, ios::in | ios::binary);

  f.read((char*)&s.sigm, sizeof(bool));

  f.read((char*)&s.lam, sizeof(float));

  f.read((char*)&s.n, sizeof(int));

  //s.sizes = new int[s.n];
  f.read((char*)s.sizes, s.n*sizeof(int));

  g = s.sigm?&sig:&relu;
  g_prime = s.sigm?&sig_prime:&relu_prime;

  //cout << s.n << endl;
  //for (int i=0; i < s.n; ++i)
  //  cout << s.sizes[i] << endl;

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
      //cout << i << "\tb: " << s.sizes[i] << endl;
      f.read((char*)b[i-1], s.sizes[i]*sizeof(float));
      W[i-1] = new float*[s.sizes[i-1]];
      dW[i-1] = new float*[s.sizes[i-1]];

      for (int j=0; j < s.sizes[i-1]; ++j)
      {
        W[i-1][j] = new float[s.sizes[i]];
        dW[i-1][j] = new float[s.sizes[i]];
        //cout << i << "\tW[" << j << "] " << s.sizes[i] << endl;
        f.read((char*)W[i-1][j], s.sizes[i]*sizeof(float));
      }
    }
  }
}

bool Neuralnet::save(char* filename)
{
  ofstream f;
  f.open(filename, ios::binary);
  if (!f)
    return false;
  f.write((char*)&s.sigm, sizeof(bool));
  f.write((char*)&s.lam, sizeof(float));
  f.write((char*)&s.n, sizeof(int));
  f.write((char*)s.sizes, s.n*sizeof(int));
  for (int l=1; l < s.n; ++l)
  {
    f.write((char*)b[l-1], s.sizes[l]*sizeof(float));
    for (int i=0; i < s.sizes[l-1]; ++i)
    {
      f.write((char*)W[l-1][i], s.sizes[l]*sizeof(float));
    }
  }
  return true;
}

bool Neuralnet::load(char* filename)
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

  //const ActivFn g(&sig);
  //const ActivFn g_prime(&sig_prime);

  ifstream f;
  f.open(filename, ios::in | ios::binary);

  if (!f)
    return false;

  f.read((char*)&s.sigm, sizeof(bool));

  f.read((char*)&s.lam, sizeof(float));

  f.read((char*)&s.n, sizeof(int));

  //s.sizes = new int[s.n];
  f.read((char*)s.sizes, s.n*sizeof(int));

  //cout << s.n << endl;
  //for (int i=0; i < s.n; ++i)
  //  cout << s.sizes[i] << endl;

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
      //cout << i << "\tb: " << s.sizes[i] << endl;
      f.read((char*)b[i-1], s.sizes[i]*sizeof(float));
      W[i-1] = new float*[s.sizes[i-1]];
      dW[i-1] = new float*[s.sizes[i-1]];

      for (int j=0; j < s.sizes[i-1]; ++j)
      {
        W[i-1][j] = new float[s.sizes[i]];
        dW[i-1][j] = new float[s.sizes[i]];
        //cout << i << "\tW[" << j << "] " << s.sizes[i] << endl;
        f.read((char*)W[i-1][j], s.sizes[i]*sizeof(float));
      }
    }
  }
  return true;
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

float Neuralnet::relu(float z)
{
  // leaky relu
  if (z > 0.0)
    return z;
  else
    //return 0.0;
    return z/100.0;
}
float Neuralnet::relu_prime(float z)
{
  // leaky relu
  if (z > 0.0)
    return 1.0;
  else
    //return 0.0;
    return 0.01;
}

float Neuralnet::sig(float z)
{
  return 1.0/(1.0 + exp(-z));
}
float Neuralnet::sig_prime(float z)
{
  float gz = sig(z);
  return gz * (1.0 - gz);
}

void Neuralnet::G(float* A, float* Z, int n)
{ // Activation function
  for (int i=0; i < n; ++i)
    A[i] = (*g)(Z[i]);
}
void Neuralnet::G_prime(float* A, float* Z, int n)
{
  for (int i=0; i < n; ++i)
    A[i] = (*g_prime)(Z[i]);
}

float Neuralnet::gradC(float a, float y)
{ // Derivative of the loss function
  return a - y;
}
void Neuralnet::gradC(float* D, float* A, float* Y, int n)
{
  for (int i=0; i < n; ++i)
    D[i] = gradC(A[i], Y[i]) * (*g_prime)(A[i]);
}

void Neuralnet::forward_prop(float* X)
{
  for (int i=0; i < s.sizes[0]; ++i)
  {
    a[0][i] = X[i];
  }

  for (int l=1; l< n_layers; ++l)
  {
    for (int j=0; j < s.sizes[l]; ++j)
    {
      z[l][j] = 0.0;
      for (int i=0; i < s.sizes[l-1]; ++i)
      {
        z[l][j] += a[l-1][i] * W[l-1][i][j];
      }
      z[l][j] += b[l-1][j];
      a[l][j] = (*g)(z[l][j]);
    }
  }
}

void Neuralnet::forward_prop(float* X, float** z, float** a, float** b, float*** W)
{
  for (int i=0; i < s.sizes[0]; ++i)
  {
    a[0][i] = X[i];
  }

  for (int l=1; l< n_layers; ++l)
  {
    for (int j=0; j < s.sizes[l]; ++j)
    {
      z[l][j] = 0.0;
      for (int i=0; i < s.sizes[l-1]; ++i)
      {
        z[l][j] += a[l-1][i] * W[l-1][i][j];
      }
      z[l][j] += b[l-1][j];
      a[l][j] = (*g)(z[l][j]);
    }
  }
}

void Neuralnet::back_prop(float* X, float* y)
{
  forward_prop(X);
  gradC(d[s.n-1], a[s.n-1], y, s.sizes[s.n-1]);

  for (int l=s.n-2; l >= 0; --l)
  {
    //dW[l] = -L*W[l] + outer(d[l+1], a[l]);
    for (int i=0; i < s.sizes[l]; ++i)
    {
      for (int j=0; j < s.sizes[l+1]; ++j)
      {
        dW[l][i][j] = d[l+1][j] * a[l][i];
      }
    }

    //db[l] = d[l+1];
    for (int i=0; i < s.sizes[l+1]; ++i)
    {
      db[l][i] = d[l+1][i];
    }

    //d[l]  = inner(W[l], d[l+1]) * g_prime(z[l]);
    for (int i=0; i < s.sizes[l]; ++i)
    {
      d[l][i] = 0.0;
      for (int j=0; j < s.sizes[l+1]; ++j)
      {
        d[l][i] += d[l+1][j] * W[l][i][j];
      }
      d[l][i] *= (*g_prime)(z[l][i]);
    }
  }
}

void Neuralnet::back_prop(float* X, float* y, float** lz, float** la, float** ld, float** lb, float** ldb, float*** lW, float*** ldW)
{
  forward_prop(X, lz, la, lb, lW);
  gradC(ld[s.n-1], la[s.n-1], y, s.sizes[s.n-1]);

//{
  for (int l=s.n-2; l >= 0; --l)
  {
    //dW[l] = outer(ld[l+1], a[l]);
    for (int i=0; i < s.sizes[l]; ++i)
    {
      for (int j=0; j < s.sizes[l+1]; ++j)
      {
        ldW[l][i][j] = ld[l+1][j] * la[l][i];
      }
    }

    //db[l] = ld[l+1];
//{
    for (int i=0; i < s.sizes[l+1]; ++i)
    {
      ldb[l][i] = ld[l+1][i];
    }
//}

    //ld[l]  = inner(W[l], ld[l+1]) * g_prime(z[l]);
    for (int i=0; i < s.sizes[l]; ++i)
    {
      ld[l][i] = 0.0;
      for (int j=0; j < s.sizes[l+1]; ++j)
      {
        ld[l][i] += ld[l+1][j] * lW[l][i][j];
      }
      ld[l][i] *= (*g_prime)(lz[l][i]);
    }
  }
//}
}

void Neuralnet::eval(float* X, float* y)
{
  forward_prop(X);
  for (int i=0; i < s.sizes[s.n-1]; ++i)
    y[i] = a[s.n-1][i];
}

void Neuralnet::update_weights(float alpha, float lam)
{
  //float lam = 1.0-(alpha*(1.0-lam_len));
  //for (int l=0; l < s.n-1; ++l)
  //{
  //  for (int i=0; i < s.sizes[l]; ++i)
  //  {
  //    for (int j=0; j < s.sizes[l+1]; ++j)
  //    {
  //      W[l][i][j] *= lam;//*W[l][i][j];
  //    }
  //  }
  //}
  for (int l=0; l < s.n-1; ++l)
  {
    for (int i=0; i < s.sizes[l]; ++i)
    {
      for (int j=0; j < s.sizes[l+1]; ++j)
      {
        W[l][i][j] = W[l][i][j]*lam - alpha*dW[l][i][j];
      }
    }
    for (int i=0; i < s.sizes[l+1]; ++i)
    {
      b[l][i] -= alpha*db[l][i];
      //cout << b[l][i] << "   \t" << db[l][i] << endl;
    }
  }
}

void Neuralnet::update_weights(float alpha, float** lb, float** ldb, float*** lW, float*** ldW)
{
  float lam = 1.0-(alpha*(1.0-Lambda));
  //for (int l=0; l < s.n-1; ++l)
  //{
  //  for (int i=0; i < s.sizes[l]; ++i)
  //  {
  //    for (int j=0; j < s.sizes[l+1]; ++j)
  //    {
  //      lW[l][i][j] *= lam;//*lW[l][i][j];
  //    }
  //  }
  //}
  for (int l=0; l < s.n-1; ++l)
  {
    for (int i=0; i < s.sizes[l]; ++i)
    {
      for (int j=0; j < s.sizes[l+1]; ++j)
      {
        lW[l][i][j] = lW[l][i][j]*lam - alpha*ldW[l][i][j];
      }
    }
    for (int i=0; i < s.sizes[l+1]; ++i)
      lb[l][i] -= alpha*ldb[l][i];
  }
}

void Neuralnet::update_weights(float** lb, float** ldb, float*** lW, float*** ldW)
{
  for (int l=0; l < s.n-1; ++l)
  {
    for (int i=0; i < s.sizes[l]; ++i)
    {
      for (int j=0; j < s.sizes[l+1]; ++j)
      {
        float diff = lW[l][i][j] - W[l][i][j];
        dW[l][i][j] -= diff;
      }
    }
    for (int i=0; i < s.sizes[l+1]; ++i)
    {
      float diff = lb[l][i] - b[l][i];
      //if (i == 0)
      //  cout << db[l][i] << "\t" << diff << endl;
      float sqr = db[l][i] * abs(db[l][i]);
      db[l][i] = sqrt(sqr + diff*abs(diff));
    }
  }
}

void Neuralnet::train(vector<float*> X_train, vector<float*> y_train, int num_epochs, float alpha, float decay)
{
  int pid = getpid();
  int interval = 10;
  if (num_epochs >= 200)
    interval = 20;
  if (num_epochs >= 1000)
    interval = 50;

  cout << "epoch: " << 0 << "    alpha: " << alpha << "\tTrain loss: " << loss(X_train, y_train) << endl;// "   \t";
  //cout << W[s.n-2][0][0] << "\t" << dW[s.n-2][0][0] << endl;

  char filename[80];
  sprintf(filename, "/tmp/%d_%d.nn", pid, 0);
  save(filename);

  float a = alpha;
  bool quit = false;
  for (int e=1; quit == false; ++e)
  {
    if (decay > 0.0)
      a = alpha * pow(decay/alpha, (float)(e)/num_epochs);
    //cout << "alpha: " << a << endl;
    vector<int> index;
    for (unsigned int i=0; i < X_train.size(); ++i)
      index.push_back(i);
    random_shuffle(index.begin(), index.end());

    //float lam = 1.0 - a*Lambda;
    //float lam_len = pow(1.0-(Lambda*a), 1.0/index.size()); // = nth-root(Lambda)
    float lam = 1.0 - a/alpha*Lambda;
    for (int i : index)
    {
      back_prop(X_train[i], y_train[i]);
      update_weights(a, lam);
      //for (int l=0; l < s.n-1; ++l)
      //{
      //  for (int i=0; i < s.sizes[l]; ++i)
      //  {
      //    for (int j=0; j < s.sizes[l+1]; ++j)
      //    {
      //      W[l][i][j] *= lam;//*W[l][i][j];
      //    }
      //  }
      //}
      //for (int l=0; l < s.n-1; ++l)
      //{
      //  for (int i=0; i < s.sizes[l]; ++i)
      //  {
      //    for (int j=0; j < s.sizes[l+1]; ++j)
      //    {
      //      W[l][i][j] -= a*dW[l][i][j];
      //    }
      //  }
      //  for (int i=0; i < s.sizes[l+1]; ++i)
      //  {
      //    b[l][i] -= a*db[l][i];
      //    //cout << b[l][i] << "   \t" << db[l][i] << endl;
      //  }
      //}
    }

    if (e%(interval) == 0)
    {
      float losses[3];
      losses[0] = loss(X_train, y_train);
      cout << "epoch: " << e << "    alpha: " << a << "\tTrain loss: " << losses[0] << endl;// "   \t";
      //cout << W[s.n-2][0][0] << "  \t" << dW[s.n-2][0][0] << endl;

      char filename0[80];
      sprintf(filename0, "/tmp/%d_%d.nn", pid, e/interval);
      save(filename0);

      if (e/interval > 1)
      {
        char filename1[80];
        char filename2[80];
        sprintf(filename1, "/tmp/%d_%d.nn", pid, e/interval - 1);
        sprintf(filename2, "/tmp/%d_%d.nn", pid, e/interval - 2);

        Neuralnet M1 = Neuralnet(filename1);
        Neuralnet M2 = Neuralnet(filename2);
        losses[1] = M1.loss(X_train, y_train);
        losses[2] = M2.loss(X_train, y_train);

        if (losses[0] > losses[1] && losses[0] > losses[2])
        {
          quit = true;
          load(filename1);
        }
      }
    }
  }
}

void Neuralnet::train(vector<float*> X_train, vector<float*> y_train, vector<float*> X_valid, vector<float*> y_valid, int num_epochs, float alpha, float decay)
{
  int pid = getpid();
  int interval = 10;
  if (num_epochs >= 200)
    interval = 20;
  if (num_epochs >= 1000)
    interval = 50;

  cout << "epoch: " << 0 << "    alpha: " << alpha << "\tTrain loss: " << loss(X_train, y_train) << "\tValidation Loss: " << loss(X_valid, y_valid) << endl;// "   \t";
  //cout << W[s.n-2][0][0] << "\t" << dW[s.n-2][0][0] << endl;

  char filename[80];
  sprintf(filename, "/tmp/%d_%d.nn", pid, 0);
  save(filename);

  float a = alpha;
  bool quit = false;
  for (int e=1; !quit; ++e)
  {
    vector<int> index;
    for (unsigned int i=0; i < X_train.size(); ++i)
      index.push_back(i);
    random_shuffle(index.begin(), index.end());

    if (decay > 0.0)
      a = alpha * pow(decay/alpha, min(1.0f, (float)(e)/num_epochs));
    //cout << "alpha: " << a << endl;

    //float lam = 1.0 - a*Lambda;
    //float lam_len = pow(1.0-(Lambda*a), 1.0/index.size()); // = nth-root(Lambda)
    float lam = 1.0 - a/alpha*Lambda;
    for (int i : index)
    {
      back_prop(X_train[i], y_train[i]);
      update_weights(a, lam);
      //for (int l=0; l < s.n-1; ++l)
      //{
      //  for (int i=0; i < s.sizes[l]; ++i)
      //  {
      //    for (int j=0; j < s.sizes[l+1]; ++j)
      //    {
      //      W[l][i][j] *= lam;//*W[l][i][j];
      //    }
      //  }
      //}
      //for (int l=0; l < s.n-1; ++l)
      //{
      //  for (int i=0; i < s.sizes[l]; ++i)
      //  {
      //    for (int j=0; j < s.sizes[l+1]; ++j)
      //    {
      //      W[l][i][j] -= a*dW[l][i][j];
      //    }
      //  }
      //  for (int i=0; i < s.sizes[l+1]; ++i)
      //  {
      //    b[l][i] -= a*db[l][i];
      //    //cout << b[l][i] << "   \t" << db[l][i] << endl;
      //  }
      //}
    }

    if (e%(interval) == 0)
    {
      float losses[3];
      losses[0] = loss(X_valid, y_valid);
      cout << "epoch: " << e << "    alpha: " << a << "\tTrain loss: " << loss(X_train, y_train) << "\tValidation Loss: " << losses[0] << endl;// "   \t";
      //cout << W[s.n-2][0][0] << "  \t" << dW[s.n-2][0][0] << endl;

      char filename0[32] = {0};
      sprintf(filename0, "/tmp/%d_%d.nn", pid, e/interval);
      save(filename0);

      if (e/interval > 1)
      {
        char filename1[32] = {0};
        char filename2[32] = {0};
        sprintf(filename1, "/tmp/%d_%d.nn", pid, e/interval - 1);
        sprintf(filename2, "/tmp/%d_%d.nn", pid, e/interval - 2);

        Neuralnet M1 = Neuralnet(filename1);
        Neuralnet M2 = Neuralnet(filename2);
        losses[1] = M1.loss(X_valid, y_valid);
        losses[2] = M2.loss(X_valid, y_valid);

        if (losses[0] > losses[1] && losses[0] > losses[2])
        {
          quit = true;
          load(filename1);
        }
      }
    }
  }
}

//void Neuralnet::train_parallel(vector<float*> X_train, vector<float*> y_train, int num_epochs, float alpha)
//{
//  int interval = 10;
//  if (num_epochs >= 200)
//    interval = 20;
//  if (num_epochs >= 1000)
//    interval = 50;
//
//  cout << "epoch: " << 0 << "\talpha: " << alpha << "\tTrain loss: " << loss(X_train, y_train) << endl;// "   \t";
//  //cout << W[s.n-2][0][0] << "\t" << dW[s.n-2][0][0] << endl;
//
//  for (int e=1; e <= num_epochs; ++e)
//  {
//    vector<int> index;
//    for (unsigned int i=0; i < X_train.size(); ++i)
//      index.push_back(i);
//    random_shuffle(index.begin(), index.end());
//
//    int num_threads;
//{
//    num_threads = omp_get_num_threads();
//    //reset main db/dW
//{
//    for (int i=1; i < s.n; ++i)
//    {
//      for (int j=0; j < s.sizes[i]; ++j)
//        db[i-1][j] = 0.0;
//      for (int j=0; j < s.sizes[i-1]; ++j)
//      {
//        for (int k=0; k < s.sizes[i]; ++k)
//          dW[i-1][j][k] = 0.0;
//      }
//    }
//}
//
//    //declare local copies of all layers
//    // yes this is probably a bit expensive
//    float**  lz  = new float*[s.n];
//    float**  la  = new float*[s.n];
//    float**  ld  = new float*[s.n];
//    float**  lb  = new float*[s.n-1];
//    float**  ldb = new float*[s.n-1];
//    float*** lW  = new float**[s.n-1];
//    float*** ldW = new float**[s.n-1];
//    lz[0] = new float[s.sizes[0]];
//    la[0] = new float[s.sizes[0]];
//    ld[0] = new float[s.sizes[0]];
//    for (int i=1; i < s.n; ++i)
//    {
//      lz[i] = new float[s.sizes[i]];
//      la[i] = new float[s.sizes[i]];
//      ld[i] = new float[s.sizes[i]];
//      lb[i-1] = new float[s.sizes[i]];
//      ldb[i-1] = new float[s.sizes[i]];
//      for (int j=0; j < s.sizes[i]; ++j)
//        lb[i-1][j] = b[i-1][j];
//      lW[i-1] = new float*[s.sizes[i-1]];
//      ldW[i-1] = new float*[s.sizes[i-1]];
//      for (int j=0; j < s.sizes[i-1]; ++j)
//      {
//        lW[i-1][j] = new float[s.sizes[i]];
//        ldW[i-1][j] = new float[s.sizes[i]];
//        for (int k=0; k < s.sizes[i]; ++k)
//          lW[i-1][j][k] = W[i-1][j][k];
//      }
//    }
//
//    for (unsigned int j=0; j < index.size(); ++j)
//    {
//      int i = index[j];
//      back_prop(X_train[i], y_train[i], lz, la, ld, lb, ldb, lW, ldW);
//      update_weights(alpha, lb, ldb, lW, ldW);
//    }
//
//{
//    update_weights(lb, ldb, lW, ldW);
//    //cout << "--------\n";
//}
//
//    delete lz[0];
//    delete la[0];
//    delete ld[0];
//    for (int i=1; i < s.n; ++i)
//    {
//      for (int j=0; j < s.sizes[i-1]; ++j)
//      {
//        delete lW[i-1][j];
//        delete ldW[i-1][j];
//      }
//      delete lb[i-1];
//      delete ldb[i-1];
//      delete lW[i-1];
//      delete ldW[i-1];
//      delete lz[i];
//      delete la[i];
//      delete ld[i];
//    }
//    delete lb;
//    delete ldb;
//    delete lW;
//    delete ldW;
//    delete lz;
//    delete la;
//    delete ld;
//}
//
//    cout << "========\n";
//    update_weights(1.0/sqrt(num_threads));
//    cout << "========\n";
//
//    if (e%(interval) == 0 || e == num_epochs)
//    {
//      cout << "epoch: " << e << "\tTrain loss: " << loss(X_train, y_train) << endl;// "   \t";
//      //cout << W[s.n-2][0][0] << "  \t" << dW[s.n-2][0][0] << endl;
//    }
//  }
//}

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
