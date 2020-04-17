#include "convnet.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <omp.h>

Convnet::Convnet(struct conv_shape* S) : g(S->sigm?&sig:&relu), g_prime(S->sigm?&sig_prime:&relu_prime)
{
  unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
  default_random_engine generator(seed);

  Lambda = S->lam; // lambda ratio to alpha
  s.n = S->n;
  s.sizes = vector<int>(s.n);
  s.kernels = vector<array<int, 4>>(s.n);
  for (int i=0; i < s.n; ++i) {
    s.sizes[i] = S->sizes[i];
    for (int j=0; j < 4; ++j) {
      s.kernels[i][j] = S->kernels[i][j];
    }
  }

  z  = vector<vector<float>>(s.n);
  a  = vector<vector<float>>(s.n);
  d  = vector<vector<float>>(s.n);
  b  = vector<vector<float>>(s.n-1);
  db = vector<vector<float>>(s.n-1);
  W  = vector<vector<vector<float>>>(s.n-1);
  dW = vector<vector<vector<float>>>(s.n-1);

  zk  = vector<vector<vector<float>>>(s.n);
  ak  = vector<vector<vector<float>>>(s.n);
  dk  = vector<vector<vector<float>>>(s.n);
  bk  = vector<vector<vector<float>>>(s.n-1);
  dbk = vector<vector<vector<float>>>(s.n-1);
  Wk  = vector<vector<vector<vector<float>>>>(s.n-1);
  dWk = vector<vector<vector<vector<float>>>>(s.n-1);

  for (int i=0; i < s.n; ++i) {
    if (s.sizes[i] > 0) {
      z[i] = vector<float>(s.sizes[i], 0.0);
      a[i] = vector<float>(s.sizes[i], 0.0);
      d[i] = vector<float>(s.sizes[i], 0.0);
      if (i > 0) {
        normal_distribution<float> distribution(0.0,1.0/s.sizes[i]);

        b[i-1] = vector<float>(s.sizes[i]);
        db[i-1] = vector<float>(s.sizes[i], 0.0);
        for (int j=0; j < s.sizes[i]; ++j)
          b[i-1][j] = distribution(generator);

        if (s.sizes[i-1] > 0) {
          W[i-1]  = vector<vector<float>>(s.sizes[i-1]);
          dW[i-1] = vector<vector<float>>(s.sizes[i-1]);
          for (int j=0; j < s.sizes[i-1]; ++j) {
            W[i-1][j]  = vector<float>(s.sizes[i]);
            dW[i-1][j] = vector<float>(s.sizes[i], 0.0);
            for (int k=0; k < s.sizes[i]; ++k)
              W[i-1][j][k] = distribution(generator);
          }
        }
        else {
          int size = s.kernels[i-1][0] * s.kernels[i-1][1] * s.kernels[i-1][3];
          W[i-1]  = vector<vector<float>>(size);
          dW[i-1] = vector<vector<float>>(size);
          for (int j=0; j < size; ++j) {
            W[i-1][j] = vector<float>(s.sizes[i]);
            dW[i-1][j] = vector<float>(s.sizes[i], 0.0);
            for (int k=0; k < s.sizes[i]; ++k)
              W[i-1][j][k] = distribution(generator);
          }
        }
      }
    }
    else { // convolution layer
      zk[i] = vector<vector<float>>(s.kernels[i][3]);
      ak[i] = vector<vector<float>>(s.kernels[i][3]);
      dk[i] = vector<vector<float>>(s.kernels[i][3]);
      for (int j=0; j < s.kernels[i][3]; ++j) { // grid unrolled in memory
        zk[i][j] = vector<float>(s.kernels[i][0]*s.kernels[i][1], 0.0);
        ak[i][j] = vector<float>(s.kernels[i][0]*s.kernels[i][1], 0.0);
        dk[i][j] = vector<float>(s.kernels[i][0]*s.kernels[i][1], 0.0);
      }
      if (i > 0) {
        normal_distribution<float> distribution(0.0,1.0/(s.kernels[i][2]*s.kernels[i][2]*s.kernels[i-1][3]));

        bk[i-1] = vector<vector<float>>(s.kernels[i][3]);
        dbk[i-1] = vector<vector<float>>(s.kernels[i][3]);
        for (int j=0; j < s.kernels[i][3]; ++j) {
          bk[i-1][j] = vector<float>(s.kernels[i][0]*s.kernels[i][1]);
          dbk[i-1][j] = vector<float>(s.kernels[i][0]*s.kernels[i][1], 0.0);
          for (int k=0; k < s.kernels[i][0]*s.kernels[i][1]; ++k)
            bk[i-1][j][k] = distribution(generator);
          //bk[i-1].shrink_to_fit();
        }

        if (s.sizes[i-1] > 0) { // from previous non-convolutional layer
          //// As of yet unsupported
          cerr << "Error: Convolutional layer follows non-convolutional layer\n";
          throw;
          //W[i-1]  = vector<vector<float>>(s.sizes[i-1]);
          //dW[i-1] = vector<vector<float>>(s.sizes[i-1]);
          //for (int j=0; j < s.sizes[i-1]; ++j) {
          //  W[i-1][j]  = vector<float>(s.sizes[i]);
          //  dW[i-1][j] = vector<float>(s.sizes[i], 0.0);
          //  for (int k=0; k < s.sizes[i]; ++k)
          //    W[i-1][j][k] = distribution(generator);
          //  W[i-1][j].shrink_to_fit();
          //}
        }
        else { // from previous convolutional layer
          if (!(s.kernels[i][0] == s.kernels[i-1][0] &&
                s.kernels[i][1] == s.kernels[i-1][1]) &&
               (s.kernels[i-1][0] - s.kernels[i][2]/2 != s.kernels[i][0] ||
                s.kernels[i-1][1] - s.kernels[i][2]/2 != s.kernels[i][1])) {
            cerr << "Error: grid/kernel size mismatch in layer " << i << endl;
            cerr << "Error: From size " << s.kernels[i-1][0] << "x" << s.kernels[i-1][1] << " to " << s.kernels[i][0] << "x" << s.kernels[i][1] << " with kernel " << s.kernels[i][2] << endl;
            throw;
          }
          Wk[i-1] = vector<vector<vector<float>>>(s.kernels[i][3]);
          dWk[i-1] = vector<vector<vector<float>>>(s.kernels[i][3]);
          for (int j=0; j < s.kernels[i][3]; ++j) {
            Wk[i-1][j] = vector<vector<float>>(s.kernels[i-1][3]);
            dWk[i-1][j] = vector<vector<float>>(s.kernels[i-1][3]);
            for (int k=0; k < s.kernels[i-1][3]; ++k) {
              Wk[i-1][j][k] = vector<float>(s.kernels[i][2]*s.kernels[i][2]);
              dWk[i-1][j][k] = vector<float>(s.kernels[i][2]*s.kernels[i][2], 0.0);
              for (int v=0; v < s.kernels[i][2]*s.kernels[i][2]; ++v) {
                Wk[i-1][j][k][v] = distribution(generator);
              }
            }
          }
        }
      }
    }
  }
}

Convnet::Convnet(char* filename) : g(), g_prime()
{
  //const ActivFn g(&sig);
  //const ActivFn g_prime(&sig_prime);

  ifstream f;
  f.open(filename, ios::in | ios::binary);

  f.read((char*)&s.sigm, sizeof(bool));

  f.read((char*)&s.lam, sizeof(float));

  f.read((char*)&s.n, sizeof(int));

  //s.sizes = new int[s.n];
  int arr[s.n];
  f.read((char*)arr, s.n*sizeof(int));
  s.sizes.clear();
  for (int i=0; i < s.n; ++i)
    s.sizes.push_back(arr[i]);

  int karr[s.n][4];
  f.read((char*)karr, 4*s.n*sizeof(int));
  s.kernels.clear();
  for (int i=0; i < s.n; ++i) {
    array<int, 4> k = {karr[i][0], karr[i][1], karr[i][2], karr[i][3]};
    s.kernels.push_back(k);
  }

  cout << "Activation:\t" << (s.sigm?"Sigmoid":"ReLu") << endl;
  cout << "Lambda: \t" << s.lam << endl;
  cout << "Layers  \t(" << s.n << "):\n";
  for (int i=0; i < s.n; ++i) {
    if (s.sizes[i] > 0)
      cout << s.sizes[i] << endl;
    else
      cout << s.kernels[i][0] << "x" << s.kernels[i][1] << " " << s.kernels[i][2] << "x" << s.kernels[i][2] << " * " << s.kernels[i][3] << endl;
  }

  z  = vector<vector<float>>(s.n);
  a  = vector<vector<float>>(s.n);
  d  = vector<vector<float>>(s.n);
  b  = vector<vector<float>>(s.n-1);
  db = vector<vector<float>>(s.n-1);
  W  = vector<vector<vector<float>>>(s.n-1);
  dW = vector<vector<vector<float>>>(s.n-1);

  zk  = vector<vector<vector<float>>>(s.n);
  ak  = vector<vector<vector<float>>>(s.n);
  dk  = vector<vector<vector<float>>>(s.n);
  bk  = vector<vector<vector<float>>>(s.n-1);
  dbk = vector<vector<vector<float>>>(s.n-1);
  Wk  = vector<vector<vector<vector<float>>>>(s.n-1);
  dWk = vector<vector<vector<vector<float>>>>(s.n-1);

  for (int i=0; i < s.n; ++i) {
    if (s.sizes[i] > 0) {
      z[i] = vector<float>(s.sizes[i], 0.0);
      a[i] = vector<float>(s.sizes[i], 0.0);
      d[i] = vector<float>(s.sizes[i], 0.0);
      if (i > 0) {
        b[i-1] = vector<float>(s.sizes[i]);
        db[i-1] = vector<float>(s.sizes[i], 0.0);
        f.read((char*)b[i-1].data(), s.sizes[i]*sizeof(float));
        //for (int j=0; j < s.sizes[i]; ++j)
        //  b[i-1][j] = distribution(generator);

        if (s.sizes[i-1] > 0) { // from fully connected layer
          W[i-1]  = vector<vector<float>>(s.sizes[i-1]);
          dW[i-1] = vector<vector<float>>(s.sizes[i-1]);
          for (int j=0; j < s.sizes[i-1]; ++j) {
            W[i-1][j]  = vector<float>(s.sizes[i]);
            dW[i-1][j] = vector<float>(s.sizes[i], 0.0);
            f.read((char*)W[i-1][j].data(), s.sizes[i]*sizeof(float));
            //for (int k=0; k < s.sizes[i]; ++k)
            //  W[i-1][j][k] = distribution(generator);
          }
        }
        else { // from convolutional layer
          int size = s.kernels[i-1][0] * s.kernels[i-1][1];
          W[i-1]  = vector<vector<float>>(size);
          dW[i-1] = vector<vector<float>>(size);
          for (int j=0; j < size; ++j) {
            W[i-1][j] = vector<float>(s.sizes[i]);
            dW[i-1][j] = vector<float>(s.sizes[i], 0.0);
            f.read((char*)W[i-1][j].data(), s.sizes[i]*sizeof(float));
            //for (int k=0; k < s.sizes[i]; ++k)
            //  W[i-1][j][k] = distribution(generator);
          }
        }
      }
    }
    else { // convolution layer
      zk[i] = vector<vector<float>>(s.kernels[i][3]);
      ak[i] = vector<vector<float>>(s.kernels[i][3]);
      dk[i] = vector<vector<float>>(s.kernels[i][3]);
      for (int j=0; j < s.kernels[i][3]; ++j) { // grid unrolled in memory
        zk[i][j] = vector<float>(s.kernels[i][0]*s.kernels[i][1], 0.0);
        ak[i][j] = vector<float>(s.kernels[i][0]*s.kernels[i][1], 0.0);
        dk[i][j] = vector<float>(s.kernels[i][0]*s.kernels[i][1], 0.0);
      }
      if (i > 0) {
        //normal_distribution<float> distribution(0.0,1.0/(s.kernels[i][2]*s.kernels[i][2]*s.kernels[i][3]));

        b[i-1] = vector<float>(s.kernels[i][0]*s.kernels[i][1]);
        db[i-1] = vector<float>(s.kernels[i][0]*s.kernels[i][1], 0.0);
        f.read((char*)b[i-1].size(), s.kernels[i][0]*s.kernels[i][1]*sizeof(float));
        //for (int j=0; j < s.kernels[i][0]*s.kernels[i][1]; ++j)
        //  b[i-1][j] = distribution(generator);

        if (s.sizes[i-1] > 0) { // from previous non-convolutional layer
          //// As of yet unsupported
          cerr << "Error: Convolutional layer " << i << " follows non-convolutional layer\n";
          throw;
          //W[i-1]  = vector<vector<float>>(s.sizes[i-1]);
          //dW[i-1] = vector<vector<float>>(s.sizes[i-1]);
          //for (int j=0; j < s.sizes[i-1]; ++j) {
          //  W[i-1][j]  = vector<float>(s.sizes[i]);
          //  dW[i-1][j] = vector<float>(s.sizes[i], 0.0);
          //  for (int k=0; k < s.sizes[i]; ++k)
          //    W[i-1][j][k] = distribution(generator);
          //  W[i-1][j].shrink_to_fit();
          //}
        }
        else { // from previous convolutional layer
          if (!(s.kernels[i][0] == s.kernels[i-1][0] &&
                s.kernels[i][1] == s.kernels[i-1][1]) ||
                s.kernels[i-1][0] - s.kernels[i][2]/2 != s.kernels[i][0] ||
                s.kernels[i-1][1] - s.kernels[i][2]/2 != s.kernels[i][1]) {
            cerr << "Error: grid/kernel size mismatch in layer " << i << endl;
            throw;
          }
          Wk[i-1] = vector<vector<vector<float>>>(s.kernels[i][3]);
          dWk[i-1] = vector<vector<vector<float>>>(s.kernels[i][3]);
          for (int j=0; j < s.kernels[i][3]; ++j) {
            Wk[i-1][j] = vector<vector<float>>(s.kernels[i-1][3]);
            dWk[i-1][j] = vector<vector<float>>(s.kernels[i-1][3]);
            for (int k=0; k < s.kernels[i-1][3]; ++k) {
              Wk[i-1][j][k] = vector<float>(s.kernels[i][2]*s.kernels[i][2]);
              dWk[i-1][j][k] = vector<float>(s.kernels[i][2]*s.kernels[i][2], 0.0);
              f.read((char*)Wk[i-1][j][k].data(), s.kernels[i][2]*s.kernels[i][2]*sizeof(float));
              //for (int v=0; v < s.kernels[i][2]*s.kernels[i][2]; ++v) {
              //  Wk[i][j][k][v] = distribution(generator);
            }
          }
        }
      }
    }
  }
//  z  = new vector<float>[s.n];
//  a  = new vector<float>[s.n];
//  d  = new vector<float>[s.n];
//  b  = new vector<float>[s.n-1];
//  db = new vector<float>[s.n-1];
//  W  = new vector<vector<float>>[s.n-1];
//  dW = new vector<vector<float>>[s.n-1];
//  for (int i=0; i < s.n; ++i)
//  {
//    z[i] = new float[s.sizes[i]];
//    a[i] = new float[s.sizes[i]];
//    d[i] = new float[s.sizes[i]];
//    if (i > 0)
//    {
//      b[i-1] = new float[s.sizes[i]];
//      db[i-1] = new float[s.sizes[i]];
//      //cout << i << "\tb: " << s.sizes[i] << endl;
//      f.read((char*)b[i-1], s.sizes[i]*sizeof(float));
//      W[i-1] = new vector<float>[s.sizes[i-1]];
//      dW[i-1] = new vector<float>[s.sizes[i-1]];
//
//      for (int j=0; j < s.sizes[i-1]; ++j)
//      {
//        W[i-1][j] = new float[s.sizes[i]];
//        dW[i-1][j] = new float[s.sizes[i]];
//        //cout << i << "\tW[" << j << "] " << s.sizes[i] << endl;
//        f.read((char*)W[i-1][j], s.sizes[i]*sizeof(float));
//      }
//    }
//  }
}

bool Convnet::save(char* filename)
{
  ofstream f;
  f.open(filename, ios::binary);
  if (!f)
    return false;
  f.write((char*)&s.sigm, sizeof(bool));
  f.write((char*)&s.lam, sizeof(float));
  f.write((char*)&s.n, sizeof(int));
  f.write((char*)s.sizes.data(), s.n*sizeof(int));
  f.write((char*)s.kernels.data(), s.n*sizeof(int));
  for (int l=1; l < s.n; ++l)
  {
    if (s.sizes[l] > 0) { // fully connected
      f.write((char*)b[l-1].data(), s.sizes[l]*sizeof(float));
      for (int i=0; i < s.sizes[l-1]; ++i)
      {
        f.write((char*)W[l-1][i].data(), s.sizes[l]*sizeof(float));
      }
    }
    else { // convolutional
      for (int k=0; k < s.kernels[l][3]; ++k)
      {
        f.write((char*)bk[l-1][k].data(), s.kernels[l][0]*s.kernels[l][1]*sizeof(float));
        for (int k0=0; k0 < s.kernels[l-1][3]; ++k0)
        {
          f.write((char*)Wk[l-1][k][k0].data(), s.kernels[l][2]*s.kernels[l][2]*sizeof(float));
        }
      }
    }
  }
  f.close();
  return true;
}

Convnet::~Convnet()
{
//  //cout << "Deleting N\n";
//  delete z[0];
//  delete a[0];
//  delete d[0];
//  for (int i=1; i < n_layers; ++i)
//  {
//    //cout << "deleting z/a/d[" << i << "]\n";
//    delete z[i];
//    delete a[i];
//    delete d[i];
//    //delete connections[i-1];
//    for (int j=0; j < s.sizes[i-1]; ++j)
//    {
//      //cout << "deleting W[" << i-1 << "][" << j << "] of " << s.sizes[i-1] << "\n";
//      delete W[i-1][j];
//      delete dW[i-1][j];
//    }
//    //cout << "deleting W/b[" << i-1 << "]\n";
//    delete b[i-1];
//    delete db[i-1];
//    delete[] W[i-1];
//    delete[] dW[i-1];
//  }
//  //cout << "deleting top level arrays\n";
//  delete[] z;
//  delete[] a;
//  delete[] d;
//  delete[] b;
//  delete[] db;
//  delete[] W;
//  delete[] dW;
}

float Convnet::relu(float z)
{
  // leaky relu
  if (z > 0.0)
    return z;
  else
    //return 0.0;
    return z*0.01;
}
float Convnet::relu_prime(float z)
{
  // leaky relu
  if (z > 0.0)
    return 1.0;
  else
    //return 0.0;
    return 0.01;
}

float Convnet::sig(float z)
{
  return 1.0/(1.0 + exp(-z));
}
float Convnet::sig_prime(float z)
{
  float gz = sig(z);
  return gz * (1.0 - gz);
}

void Convnet::G(vector<float> A, vector<float> Z, int n)
{ // Activation function
  for (int i=0; i < n; ++i)
    A[i] = (*g)(Z[i]);
}
void Convnet::G_prime(vector<float> A, vector<float> Z, int n)
{
  for (int i=0; i < n; ++i)
    A[i] = (*g_prime)(Z[i]);
}

float Convnet::gradC(float a, float y)
{ // Derivative of the loss function
  return a - y;
}
void Convnet::gradC(vector<float> D, vector<float> A, vector<float> Y, int n)
{
  for (int i=0; i < n; ++i)
    D[i] = gradC(A[i], Y[i]) * (*g_prime)(A[i]);
}
void Convnet::gradC(vector<vector<float>> D, vector<vector<float>> A, vector<vector<float>> Y, int n, int k)
{
  for (int i=0; i < k; ++i) {
    for (int j=0; j < n; ++j)
      D[i][j] = gradC(A[i][j], Y[i][j]) * (*g_prime)(A[i][j]);
  }
}

void Convnet::forward_prop(vector<float> X)
{
  // Populate input layer
  for (int i=0; i < s.sizes[0]; ++i)
  {
    a[0][i] = X[i];
  }

  // Propagate values through the network
  for (int l=1; l < s.n; ++l)
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

void Convnet::forward_prop(vector<vector<float>> X)
{
  // Populate input layer
  for (int i=0; i < s.kernels[0][3]; ++i) {
    for (int j=0; j < s.kernels[0][0]*s.kernels[0][1]; ++j) {
      ak[0][i][j] = X[i][j];
    }
  }

  // Propagate values through the network
  for (int l=1; l < s.n; ++l) {
    if (s.sizes[l] > 0) { // feedforward layer
      for (int j=0; j < s.sizes[l]; ++j) {
        z[l][j] = 0.0;
        if (s.sizes[l-1] > 0) { // from previous feedforward layer
          for (int i=0; i < s.sizes[l-1]; ++i) {
            z[l][j] += a[l-1][i] * W[l-1][i][j];
          }
          z[l][j] += b[l-1][j];
          a[l][j] = (*g)(z[l][j]);
        }
        else { // from previous convolutional layer
          int i = 0;
          for (int k=0; k < s.kernels[l-1][3]; ++k) {
            for (int ki=0; ki < s.kernels[l-1][0]*s.kernels[l-1][1]; ++ki) {
              z[l][j] += ak[l-1][k][ki] * W[l-1][i][j];
              ++i;
            }
          }
        }
      }
    }
    else { // convolutional layer
      if (s.kernels[l][0] == s.kernels[l-1][0] &&
          s.kernels[l][1] == s.kernels[l-1][1]) { // with matching grid size
        for (int k=0; k < s.kernels[l][3]; ++k) {
          for (int j=0; j < s.kernels[l][0]*s.kernels[l][1]; ++j) {
            zk[l][k][j] = 0.0;
            for (int k0=0; k0 < s.kernels[l-1][3]; ++k0) {
              for (int x0=0; x0 < s.kernels[l][2]; ++x0) {
                for (int y0=0; y0 < s.kernels[l][2]; ++y0) {
                  int j0 = j - s.kernels[l][1]*(s.kernels[l][2]/2 - x0) - s.kernels[l][2]/2 + y0;
                  if (j0 >= 0 && j0 < s.kernels[l-1][0] * s.kernels[l-1][1]) {
                    zk[l][k][j] += ak[l-1][k0][j0] * Wk[l-1][k][k0][x0*s.kernels[l][2]+y0];
                  }
                }
              }
            }
            ak[l][k][j] = (*g)(zk[l][k][j]);
          }
        }
      }
      else { // grid size shrinks by kernel size
        // unsupported for now
      }
    }
  }
}

//void Convnet::forward_prop(vector<float> X, vector<vector<float>> z, vector<vector<float>> a, vector<vector<float>> b, vector<vector<vector<float>>> W)
//{
//  for (int i=0; i < s.sizes[0]; ++i)
//  {
//    a[0][i] = X[i];
//  }
//
//  for (int l=1; l < n_layers; ++l)
//  {
//    for (int j=0; j < s.sizes[l]; ++j)
//    {
//      z[l][j] = 0.0;
//      for (int i=0; i < s.sizes[l-1]; ++i)
//      {
//        z[l][j] += a[l-1][i] * W[l-1][i][j];
//      }
//      z[l][j] += b[l-1][j];
//      a[l][j] = (*g)(z[l][j]);
//    }
//  }
//}

void Convnet::back_prop(vector<float> X, vector<float> y)
{
  forward_prop(X);
  gradC(d[s.n-1], a[s.n-1], y, s.sizes[s.n-1]);

  for (int l=s.n-2; l >= 0; --l) {
    //dW[l] = -L*W[l] + outer(d[l+1], a[l]);
    for (int i=0; i < s.sizes[l]; ++i) {
      for (int j=0; j < s.sizes[l+1]; ++j) {
        dW[l][i][j] = d[l+1][j] * a[l][i];
      }
    }

    //db[l] = d[l+1];
    for (int i=0; i < s.sizes[l+1]; ++i)
      db[l][i] = d[l+1][i];

    //d[l]  = inner(W[l], d[l+1]) * g_prime(z[l]);
    for (int i=0; i < s.sizes[l]; ++i) {
      d[l][i] = 0.0;
      for (int j=0; j < s.sizes[l+1]; ++j) {
        d[l][i] += d[l+1][j] * W[l][i][j];
      }
      d[l][i] *= (*g_prime)(z[l][i]);
    }
  }
}

void Convnet::back_prop(vector<vector<float>> X, vector<float> y)
{
  forward_prop(X);
  gradC(d[s.n-1], a[s.n-1], y, s.sizes[s.n-1]);

  for (int l=s.n-2; l >= 0; --l) {
    if (s.sizes[l] > 0) { // fully connected layer
      //dW[l] = -L*W[l] + outer(d[l+1], a[l]);
      for (int i=0; i < s.sizes[l]; ++i) {
        for (int j=0; j < s.sizes[l+1]; ++j) {
          dW[l][i][j] = d[l+1][j] * a[l][i];
        }
      }

      //db[l] = d[l+1];
      for (int i=0; i < s.sizes[l+1]; ++i)
        db[l][i] = d[l+1][i];

      //d[l]  = inner(W[l], d[l+1]) * g_prime(z[l]);
      for (int i=0; i < s.sizes[l]; ++i) {
        d[l][i] = 0.0;
        for (int j=0; j < s.sizes[l+1]; ++j) {
          d[l][i] += d[l+1][j] * W[l][i][j];
        }
        d[l][i] *= (*g_prime)(z[l][i]);
      }
    }
    else { // convolution layer
      if (s.sizes[l+1] > 0) { // from fully connected layer
        //dW[l] = -L*W[l] + outer(d[l+1], a[l]);
        int ii = 0;
        for (int k=0; k < s.kernels[l][3]; ++k) {
          for (int i=0; i < s.kernels[l][0]*s.kernels[l][1]; ++i) {
            for (int j=0; j < s.sizes[l+1]; ++j) {
              dW[l][ii][j] = d[l+1][j] * ak[l][k][i];
            }
            ++ii;
          }
        }

        //db[l] = d[l+1];
        for (int i=0; i < s.sizes[l+1]; ++i)
          db[l][i] = d[l+1][i];

        //d[l] = inner(W[l], d[l+1]) * g_prime(z[l]);
        ii = 0;
        for (int k=0; k < s.kernels[l][3]; ++k) {
          for (int i=0; i < s.kernels[l][0]*s.kernels[l][1]; ++i) {
            dk[l][k][i] = 0.0;
            for (int j=0; j < s.sizes[l+1]; ++j) {
              dk[l][k][i] += d[l+1][j] * W[l][ii][j];
            }
            dk[l][k][i] *= (*g_prime)(zk[l][k][i]);
            ++ii;
          }
        }
      }
      else { // from convolution layer
        //dW[l] = -L*W[l] + outer(d[l+1], a[l]);
        for (int k=0; k < s.kernels[l+1][3]; ++k) {
          for (int k0=0; k0 < s.kernels[l][3]; ++k0) {
            for (int x0=0; x0 < s.kernels[l+1][2]; ++x0) {
              for (int y0=0; y0 < s.kernels[l+1][2]; ++y0) {
                dWk[l][k][k0][x0*s.kernels[l+1][2]+y0] = 0.0;
              }
            }
            for (int j=0; j < s.kernels[l+1][0]*s.kernels[l+1][1]; ++j) {
              for (int x0=0; x0 < s.kernels[l+1][2]; ++x0) {
                for (int y0=0; y0 < s.kernels[l+1][2]; ++y0) {
                  int j0 = j - s.kernels[l+1][1]*(s.kernels[l][2]/2 - x0) - s.kernels[l+1][2]/2 + y0;
                  //if (j0 >= 0 && j0 < s.kernels[l][0] * s.kernels[l][1]) {
                  if (j0 >= 0 && j0 < s.kernels[l][0] * s.kernels[l][1] &&
                      j0/s.kernels[l][1] == j/s.kernels[l][1] + (x0 - s.kernels[l][2]/2)) {
                    dWk[l][k][k0][x0*s.kernels[l+1][2]+y0] += dk[l+1][k][j] * ak[l][k0][j0];
                  }
                }
              }
            }
          }
        }

        //db[l] = d[l+1];
        for (int k=0; k < s.kernels[l+1][3]; ++k) {
          for (int i=0; i < s.kernels[l+1][0]*s.kernels[l+1][1]; ++i) {
            dbk[l][k][i] = dk[l+1][k][i];
          }
        }

        //d[l] = inner(W[l], d[l+1]) * g_prime(z[l]);
        for (int k0=0; k0 < s.kernels[l][3]; ++k0) {
          for (int j0=0; j0 < s.kernels[l][0]*s.kernels[l][1]; ++j0) {
            dk[l][k0][j0] = 0.0;
          }
        }
        for (int k0=0; k0 < s.kernels[l][3]; ++k0) {
          for (int k=0; k < s.kernels[l+1][3]; ++k) {
            for (int j=0; j < s.kernels[l+1][0]*s.kernels[l+1][1]; ++j) {
              for (int x0=0; x0 < s.kernels[l+1][2]; ++x0) {
                for (int y0=0; y0 < s.kernels[l+1][2]; ++y0) {
                  int j0 = j - s.kernels[l+1][1]*(s.kernels[l][2]/2 - x0) - s.kernels[l+1][2]/2 + y0;
                  dk[l][k0][j0] += Wk[l][k][k0][x0*s.kernels[l+1][2]+y0] * dk[l+1][k][j];
                }
              }
            }
          }
        }
        for (int k0=0; k0 < s.kernels[l][3]; ++k0) {
          for (int j0=0; j0 < s.kernels[l][0]*s.kernels[l][1]; ++j0) {
            dk[l][k0][j0] *= (*g_prime)(zk[l][k0][j0]);
          }
        }
      }
    }
  }
}

void Convnet::back_prop(vector<vector<float>> X, vector<vector<float>> y)
{
  forward_prop(X);
  gradC(dk[s.n-1], ak[s.n-1], y, s.kernels[s.n-1][0]*s.kernels[s.n-1][1], s.kernels[s.n-1][3]);

  for (int l=s.n-2; l >= 0; --l) {
    if (s.sizes[l] > 0) { // fully connected layer
      //dW[l] = -L*W[l] + outer(d[l+1], a[l]);
      for (int i=0; i < s.sizes[l]; ++i) {
        for (int j=0; j < s.sizes[l+1]; ++j) {
          dW[l][i][j] = d[l+1][j] * a[l][i];
        }
      }

      //db[l] = d[l+1];
      for (int i=0; i < s.sizes[l+1]; ++i)
        db[l][i] = d[l+1][i];

      //d[l]  = inner(W[l], d[l+1]) * g_prime(z[l]);
      for (int i=0; i < s.sizes[l]; ++i) {
        d[l][i] = 0.0;
        for (int j=0; j < s.sizes[l+1]; ++j) {
          d[l][i] += d[l+1][j] * W[l][i][j];
        }
        d[l][i] *= (*g_prime)(z[l][i]);
      }
    }
    else { // convolution layer
      if (s.sizes[l+1] > 0) { // from fully connected layer
        //dW[l] = -L*W[l] + outer(d[l+1], a[l]);
        int ii = 0;
        for (int k=0; k < s.kernels[l][3]; ++k) {
          for (int i=0; i < s.kernels[l][0]*s.kernels[l][1]; ++i) {
            for (int j=0; j < s.sizes[l+1]; ++j) {
              dW[l][ii][j] = d[l+1][j] * ak[l][k][i];
              ++ii;
            }
          }
        }

        //db[l] = d[l+1];
        for (int i=0; i < s.sizes[l+1]; ++i)
          db[l][i] = d[l+1][i];

        //d[l] = inner(W[l], d[l+1]) * g_prime(z[l]);
        ii = 0;
        for (int k=0; k < s.kernels[l][3]; ++k) {
          for (int i=0; i < s.kernels[l][0]*s.kernels[l][1]; ++i) {
            dk[l][k][i] = 0.0;
            for (int j=0; j < s.sizes[l+1]; ++j) {
              dk[l][k][i] += d[l+1][j] * W[l][ii][j];
            }
            dk[l][k][i] *= (*g_prime)(zk[l][k][i]);
            ++ii;
          }
        }
      }
      else { // from convolution layer
        //dW[l] = -L*W[l] + outer(d[l+1], a[l]);
        for (int k=0; k < s.kernels[l+1][3]; ++k) {
          for (int k0=0; k0 < s.kernels[l][3]; ++k0) {
            for (int x0=0; x0 < s.kernels[l+1][2]; ++x0) {
              for (int y0=0; y0 < s.kernels[l+1][2]; ++y0) {
                dWk[l][k][k0][x0*s.kernels[l+1][2]+y0] = 0.0;
              }
            }
            for (int j=0; j < s.kernels[l+1][0]*s.kernels[l+1][1]; ++j) {
              for (int x0=0; x0 < s.kernels[l+1][2]; ++x0) {
                for (int y0=0; y0 < s.kernels[l+1][2]; ++y0) {
                  int j0 = j - s.kernels[l+1][1]*(s.kernels[l][2]/2 - x0) - s.kernels[l+1][2]/2 + y0;
                  //if (j0 >= 0 && j0 < s.kernels[l][0] * s.kernels[l][1]) {
                  if (j0 >= 0 && j0 < s.kernels[l][0] * s.kernels[l][1] &&
                      j0/s.kernels[l][1] == j/s.kernels[l][1] + (x0 - s.kernels[l][2]/2)) {
                    dWk[l][k][k0][x0*s.kernels[l+1][2]+y0] += dk[l+1][k][j] * ak[l][k0][j0];
                  }
                }
              }
            }
          }
        }

        //db[l] = d[l+1];
        for (int k=0; k < s.kernels[l+1][3]; ++k) {
          for (int i=0; i < s.kernels[l+1][0]*s.kernels[l+1][1]; ++i) {
            dbk[l][k][i] = dk[l+1][k][i];
          }
        }

        //d[l] = inner(W[l], d[l+1]) * g_prime(z[l]);
        for (int k0=0; k0 < s.kernels[l][3]; ++k0) {
          for (int j0=0; j0 < s.kernels[l][0]*s.kernels[l][1]; ++j0) {
            dk[l][k0][j0] = 0.0;
          }
        }
        for (int k0=0; k0 < s.kernels[l][3]; ++k0) {
          for (int k=0; k < s.kernels[l+1][3]; ++k) {
            for (int j=0; j < s.kernels[l+1][0]*s.kernels[l+1][1]; ++j) {
              for (int x0=0; x0 < s.kernels[l+1][2]; ++x0) {
                for (int y0=0; y0 < s.kernels[l+1][2]; ++y0) {
                  int j0 = j - s.kernels[l+1][1]*(s.kernels[l][2]/2 - x0) - s.kernels[l+1][2]/2 + y0;
                  dk[l][k0][j0] += Wk[l][k][k0][x0*s.kernels[l+1][2]+y0] * dk[l+1][k][j];
                }
              }
            }
          }
        }
        for (int k0=0; k0 < s.kernels[l][3]; ++k0) {
          for (int j0=0; j0 < s.kernels[l][0]*s.kernels[l][1]; ++j0) {
            dk[l][k0][j0] *= (*g_prime)(zk[l][k0][j0]);
          }
        }
      }
    }
  }
}

//void Convnet::back_prop(vector<float> X, vector<float> y, vector<vector<float>> lz, vector<vector<float>> la, vector<vector<float>> ld, vector<vector<float>> lb, vector<vector<float>> ldb, vector<vector<vector<float>>> lW, vector<vector<vector<float>>> ldW)
//{
//  forward_prop(X, lz, la, lb, lW);
//  gradC(ld[s.n-1], la[s.n-1], y, s.sizes[s.n-1]);
//
////{
//  for (int l=s.n-2; l >= 0; --l)
//  {
//    //dW[l] = outer(ld[l+1], a[l]);
//    for (int i=0; i < s.sizes[l]; ++i)
//    {
//      for (int j=0; j < s.sizes[l+1]; ++j)
//      {
//        ldW[l][i][j] = ld[l+1][j] * la[l][i];
//      }
//    }
//
//    //db[l] = ld[l+1];
////{
//    for (int i=0; i < s.sizes[l+1]; ++i)
//    {
//      ldb[l][i] = ld[l+1][i];
//    }
////}
//
//    //ld[l]  = inner(W[l], ld[l+1]) * g_prime(z[l]);
//    for (int i=0; i < s.sizes[l]; ++i)
//    {
//      ld[l][i] = 0.0;
//      for (int j=0; j < s.sizes[l+1]; ++j)
//      {
//        ld[l][i] += ld[l+1][j] * lW[l][i][j];
//      }
//      ld[l][i] *= (*g_prime)(lz[l][i]);
//    }
//  }
////}
//}

void Convnet::eval(vector<float> X, vector<float> y)
{
  forward_prop(X);
  for (int i=0; i < s.sizes[s.n-1]; ++i)
    y[i] = a[s.n-1][i];
}
void Convnet::eval(vector<vector<float>> X, vector<float> y)
{
  forward_prop(X);
  for (int i=0; i < s.sizes[s.n-1]; ++i)
    y[i] = a[s.n-1][i];
}
void Convnet::eval(vector<vector<float>> X, vector<vector<float>> y)
{
  forward_prop(X);
  for (int i=0; i < s.kernels[s.n-1][3]; ++i) {
    for (int j=0; j < s.kernels[s.n-1][0]*s.kernels[s.n-1][1]; ++j) {
      y[i][j] = ak[s.n-1][i][j];
    }
  }
}

void Convnet::update_weights(float alpha, float lam)
{
  for (int l=0; l < s.n-1; ++l)
  {
    if (s.sizes[l+1] > 0) { //fully connected
      if (s.sizes[l] > 0) { //from fully connected
        for (int i=0; i < s.sizes[l]; ++i) {
          for (int j=0; j < s.sizes[l+1]; ++j) {
            W[l][i][j] = W[l][i][j]*lam - alpha*dW[l][i][j];
          }
        }
      }
      else { //from convolution
        for (int i=0; i < s.kernels[l][0]*s.kernels[l][1]*s.kernels[l][3]; ++i) {
          for (int j=0; j < s.sizes[l+1]; ++j) {
            W[l][i][j] = W[l][i][j]*lam - alpha*dW[l][i][j];
          }
        }
      }
      for (int i=0; i < s.sizes[l+1]; ++i)
      {
        b[l][i] -= alpha*db[l][i];
      }
    }
    else { // convolution
      for (int k=0; k < s.kernels[l+1][3]; ++k) {
        for (int k0=0; k0 < s.kernels[l][3]; ++k0) {
          for (int i=0; i < s.kernels[l+1][2]*s.kernels[l+1][2]; ++i) {
            Wk[l][k][k0][i] = Wk[l][k][k0][i]*lam - alpha * dWk[l][k][k0][i];
          }
        }
        for (int i=0; i < s.kernels[l+1][0]*s.kernels[l+1][1]; ++i) {
          bk[l][k][i] -= alpha*dbk[l][k][i];
        }
      }
    }
  }
}

void Convnet::update_weights(float alpha, vector<vector<float>> lb, vector<vector<float>> ldb, vector<vector<vector<float>>> lW, vector<vector<vector<float>>> ldW)
{
  float lam = 1.0-(alpha*(1.0-Lambda));
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

void Convnet::update_weights(vector<vector<float>> lb, vector<vector<float>> ldb, vector<vector<vector<float>>> lW, vector<vector<vector<float>>> ldW)
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

void Convnet::train(vector<vector<float>> X_train, vector<vector<float>> y_train, int num_epochs, float alpha, float decay)
{
  int interval = 1;
  if (num_epochs >= 20)
    interval = 5;
  if (num_epochs >= 100)
    interval = 10;
  if (num_epochs >= 300)
    interval = 20;
  if (num_epochs >= 1000)
    interval = 50;

  cout << "epoch: " << 0 << "    alpha: " << alpha << "\tTrain loss: " << loss(X_train, y_train) << endl;// "   \t";
  //cout << W[s.n-2][0][0] << "\t" << dW[s.n-2][0][0] << endl;

  float al = alpha;
  for (int e=1; e <= num_epochs; ++e)
  {
    if (decay > 0.0)
      al = alpha * pow(decay/alpha, (float)(e)/num_epochs);
    //cout << "alpha: " << al << endl;
    vector<int> index;
    for (unsigned int i=0; i < X_train.size(); ++i)
      index.push_back(i);
    random_shuffle(index.begin(), index.end());

    float lam = 1.0 - al/alpha*Lambda;
    for (int i : index)
    {
      back_prop(X_train[i], y_train[i]);
      update_weights(al, lam);
    }

    if (e%(interval) == 0 || e == num_epochs)
    {
      cout << "epoch: " << e << "    alpha: " << al << "\tTrain loss: " << loss(X_train, y_train) << endl;// "   \t";
      //cout << W[s.n-2][0][0] << "  \t" << dW[s.n-2][0][0] << endl;
    }
  }
}
void Convnet::train(vector<vector<vector<float>>> X_train, vector<vector<float>> y_train, int num_epochs, float alpha, float decay)
{
  int interval = 1;
  if (num_epochs >= 20)
    interval = 5;
  if (num_epochs >= 100)
    interval = 10;
  if (num_epochs >= 300)
    interval = 20;
  if (num_epochs >= 1000)
    interval = 50;

  cout << "epoch: " << 0 << "    alpha: " << alpha << "\tTrain loss: " << loss(X_train, y_train) << endl;// "   \t";
  //cout << W[s.n-2][0][0] << "\t" << dW[s.n-2][0][0] << endl;

  float al = alpha;
  for (int e=1; e <= num_epochs; ++e)
  {
    if (decay > 0.0)
      al = alpha * pow(decay/alpha, (float)(e)/num_epochs);
    //cout << "alpha: " << al << endl;
    vector<int> index;
    for (unsigned int i=0; i < X_train.size(); ++i)
      index.push_back(i);
    random_shuffle(index.begin(), index.end());

    float lam = 1.0 - al/alpha*Lambda;
    for (int i : index)
    {
      back_prop(X_train[i], y_train[i]);
      update_weights(al, lam);
    }

    if (e%(interval) == 0 || e == num_epochs)
    {
      cout << "epoch: " << e << "    alpha: " << al << "\tTrain loss: " << loss(X_train, y_train) << endl;// "   \t";
      //cout << W[s.n-2][0][0] << "  \t" << dW[s.n-2][0][0] << endl;
    }
  }
}
void Convnet::train(vector<vector<vector<float>>> X_train, vector<vector<vector<float>>> y_train, int num_epochs, float alpha, float decay)
{
  int interval = 1;
  if (num_epochs >= 20)
    interval = 5;
  if (num_epochs >= 100)
    interval = 10;
  if (num_epochs >= 300)
    interval = 20;
  if (num_epochs >= 1000)
    interval = 50;

  cout << "epoch: " << 0 << "    alpha: " << alpha << "\tTrain loss: " << loss(X_train, y_train) << endl;// "   \t";
  //cout << W[s.n-2][0][0] << "\t" << dW[s.n-2][0][0] << endl;

  float al = alpha;
  for (int e=1; e <= num_epochs; ++e)
  {
    if (decay > 0.0)
      al = alpha * pow(decay/alpha, (float)(e)/num_epochs);
    //cout << "alpha: " << al << endl;
    vector<int> index;
    for (unsigned int i=0; i < X_train.size(); ++i)
      index.push_back(i);
    random_shuffle(index.begin(), index.end());

    float lam = 1.0 - al/alpha*Lambda;
    for (int i : index)
    {
      back_prop(X_train[i], y_train[i]);
      update_weights(al, lam);
    }

    if (e%(interval) == 0 || e == num_epochs)
    {
      cout << "epoch: " << e << "    alpha: " << al << "\tTrain loss: " << loss(X_train, y_train) << endl;// "   \t";
      //cout << W[s.n-2][0][0] << "  \t" << dW[s.n-2][0][0] << endl;
    }
  }
}

void Convnet::train(vector<vector<float>> X_train, vector<vector<float>> y_train, vector<vector<float>> X_valid, vector<vector<float>> y_valid, int num_epochs, float alpha, float decay)
{
  int interval = 1;
  if (num_epochs >= 20)
    interval = 5;
  if (num_epochs >= 100)
    interval = 10;
  if (num_epochs >= 300)
    interval = 20;
  if (num_epochs >= 1000)
    interval = 50;

  cout << "epoch: " << 0 << "    alpha: " << alpha << "\tTrain loss: " << loss(X_train, y_train) << "\tValidation Loss: " << loss(X_valid, y_valid) << endl;// "   \t";
  //cout << W[s.n-2][0][0] << "\t" << dW[s.n-2][0][0] << endl;

  float al = alpha;
  for (int e=1; e <= num_epochs; ++e)
  {
    vector<int> index;
    for (unsigned int i=0; i < X_train.size(); ++i)
      index.push_back(i);
    random_shuffle(index.begin(), index.end());

    if (decay > 0.0)
      al = alpha * pow(decay/alpha, (float)(e)/num_epochs);
    //cout << "alpha: " << a << endl;

    //float lam = 1.0 - a*Lambda;
    //float lam_len = pow(1.0-(Lambda*a), 1.0/index.size()); // = nth-root(Lambda)
    float lam = 1.0 - al/alpha*Lambda;
    for (int i : index)
    {
      back_prop(X_train[i], y_train[i]);
      update_weights(al, lam);
    }

    if (e%(interval) == 0 || e == num_epochs)
    {
      cout << "epoch: " << e << "    alpha: " << al << "\tTrain loss: " << loss(X_train, y_train) << "\tValidation Loss: " << loss(X_valid, y_valid) << endl;// "   \t";
      //cout << W[s.n-2][0][0] << "  \t" << dW[s.n-2][0][0] << endl;
    }
  }
}
void Convnet::train(vector<vector<vector<float>>> X_train, vector<vector<float>> y_train, vector<vector<vector<float>>> X_valid, vector<vector<float>> y_valid, int num_epochs, float alpha, float decay) {

  int interval = 1;
  if (num_epochs >= 20)
    interval = 5;
  if (num_epochs >= 100)
    interval = 10;
  if (num_epochs >= 300)
    interval = 20;
  if (num_epochs >= 1000)
    interval = 50;

  cout << "epoch: " << 0 << "    alpha: " << alpha << "\tTrain loss: " << loss(X_train, y_train) << "\tValidation Loss: " << loss(X_valid, y_valid) << endl;// "   \t";
  //cout << W[s.n-2][0][0] << "\t" << dW[s.n-2][0][0] << endl;

  float al = alpha;
  for (int e=1; e <= num_epochs; ++e)
  {
    vector<int> index;
    for (unsigned int i=0; i < X_train.size(); ++i)
      index.push_back(i);
    random_shuffle(index.begin(), index.end());

    if (decay > 0.0)
      al = alpha * pow(decay/alpha, (float)(e)/num_epochs);
    //cout << "alpha: " << a << endl;

    //float lam = 1.0 - a*Lambda;
    //float lam_len = pow(1.0-(Lambda*a), 1.0/index.size()); // = nth-root(Lambda)
    float lam = 1.0 - al/alpha*Lambda;
    for (int i : index)
    {
      back_prop(X_train[i], y_train[i]);
      update_weights(al, lam);
    }

    if (e%(interval) == 0 || e == num_epochs)
    {
      cout << "epoch: " << e << "    alpha: " << al << "\tTrain loss: " << loss(X_train, y_train) << "\tValidation Loss: " << loss(X_valid, y_valid) << endl;// "   \t";
      //cout << W[s.n-2][0][0] << "  \t" << dW[s.n-2][0][0] << endl;
    }
  }
}
void Convnet::train(vector<vector<vector<float>>> X_train, vector<vector<vector<float>>> y_train, vector<vector<vector<float>>> X_valid, vector<vector<vector<float>>> y_valid, int num_epochs, float alpha, float decay)
{

  int interval = 1;
  if (num_epochs >= 20)
    interval = 5;
  if (num_epochs >= 100)
    interval = 10;
  if (num_epochs >= 300)
    interval = 20;
  if (num_epochs >= 1000)
    interval = 50;

  cout << "epoch: " << 0 << "    alpha: " << alpha << "\tTrain loss: " << loss(X_train, y_train) << "\tValidation Loss: " << loss(X_valid, y_valid) << endl;// "   \t";
  //cout << W[s.n-2][0][0] << "\t" << dW[s.n-2][0][0] << endl;

  float al = alpha;
  for (int e=1; e <= num_epochs; ++e)
  {
    vector<int> index;
    for (unsigned int i=0; i < X_train.size(); ++i)
      index.push_back(i);
    random_shuffle(index.begin(), index.end());

    if (decay > 0.0)
      al = alpha * pow(decay/alpha, (float)(e)/num_epochs);
    //cout << "alpha: " << a << endl;

    //float lam = 1.0 - a*Lambda;
    //float lam_len = pow(1.0-(Lambda*a), 1.0/index.size()); // = nth-root(Lambda)
    float lam = 1.0 - al/alpha*Lambda;
    for (int i : index)
    {
      back_prop(X_train[i], y_train[i]);
      update_weights(al, lam);
    }

    if (e%(interval) == 0 || e == num_epochs)
    {
      cout << "epoch: " << e << "    alpha: " << al << "\tTrain loss: " << loss(X_train, y_train) << "\tValidation Loss: " << loss(X_valid, y_valid) << endl;// "   \t";
      //cout << W[s.n-2][0][0] << "  \t" << dW[s.n-2][0][0] << endl;
    }
  }
}

//void Convnet::train_parallel(vector<vector<float>> X_train, vector<vector<float>> y_train, int num_epochs, float alpha)
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
//    vector<vector<float>>  lz  = new vector<float>[s.n];
//    vector<vector<float>>  la  = new vector<float>[s.n];
//    vector<vector<float>>  ld  = new vector<float>[s.n];
//    vector<vector<float>>  lb  = new vector<float>[s.n-1];
//    vector<vector<float>>  ldb = new vector<float>[s.n-1];
//    vector<vector<vector<float>>> lW  = new vector<vector<float>>[s.n-1];
//    vector<vector<vector<float>>> ldW = new vector<vector<float>>[s.n-1];
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
//      lW[i-1] = new vector<float>[s.sizes[i-1]];
//      ldW[i-1] = new vector<float>[s.sizes[i-1]];
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

float Convnet::loss(vector<vector<float>> X_train, vector<vector<float>> y_train)
{
  float L= 0.0;

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

float Convnet::loss(vector<vector<vector<float>>> X_train, vector<vector<float>> y_train)
{
  float L= 0.0;

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

float Convnet::loss(vector<vector<vector<float>>> X_train, vector<vector<vector<float>>> y_train)
{
  float L= 0.0;

  for (unsigned int i=0; i < X_train.size(); ++i)
  {
    forward_prop(X_train[i]);
    for (int j=0; j < s.kernels[s.n-1][3]; ++j)
    {
      for (int k=0; k < s.kernels[s.n-1][0]*s.kernels[s.n-1][1]; ++k)
      {
        float diff = ak[s.n-1][j][k] - y_train[i][j][k];
        if (0.5 * diff * diff < 0)
          cout << "wtf" << endl;
        L += 0.5 * diff * diff;
      }
    }
  }
  return L;
}
