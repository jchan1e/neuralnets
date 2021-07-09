#include <vector>
#include <fstream>

#ifndef NN_NEURALNET
#define NN_NEURALNET

using namespace std;

struct shape
{
  int n;
  int sizes[64] = {0};
  bool sigm;
  float lam;
  //shape(int x)
  //{
  //  n = x;
  //  sizes = new int[n];
  //}
  //~shape()
  //{
  //  delete[] sizes;
  //}
};

class Neuralnet
{
//public:
typedef float (*ActivFn)(float z);

private:
// activation functions and their derivatives
  static float sig(float z);
  static float sig_prime(float z);
  static float relu(float z);
  static float relu_prime(float z);
  ActivFn g;
  ActivFn g_prime;
  void  G(float* A, float* Z, int n);
  void  G_prime(float* A, float* Z, int n);
// gradient of the output loss
  float gradC(float a, float y);
  void  gradC(float* D, float* A, float* Y, int n);
// forward and back propagation
  void forward_prop(float* X);
  void forward_prop(float* X, float** lz, float** la, float** lb, float*** lW);
  void back_prop(float* X, float* y);
  void back_prop(float* X, float* y, float** lz, float** la, float** ld, float** lb, float** ldb, float*** lW, float*** ldw);
  void update_weights(float alpha, float lam_len);
  void update_weights(float alpha, float** lb, float** ldb, float*** lW, float*** ldw);
  void update_weights(float** lb, float** ldb, float*** lW, float*** ldW);
public:
  shape s;
  float Lambda; // ratio between actual lambda and alpha
  int n_layers;
  float** b;
  float** db;
  float*** W;
  float*** dW;
  float** z;
  float** a;
  float** d;

  Neuralnet(struct shape* S);//, bool sigm=false, float lam=0.05);
  ~Neuralnet();
  void eval(float* X, float* y);
  void train(vector<float*> X_train, vector<float*> y_train, int num_epochs=10, float alpha=0.25, float decay=0.0);
  void train(vector<float*> X_train, vector<float*> y_train, vector<float*> X_valid, vector<float*> y_valid, int num_epochs=10, float alpha=0.25, float decay=0.0);
  //void train_parallel(vector<float*> X_train, vector<float*> y_train, int num_epochs=10, float alpha=0.25);
  //void train_parallel(vector<float*> X_train, vector<float*> y_train, vector<float*> X_valid, vector<float*> y_valid, int num_epochs=10, float alpha=0.25);
  float loss(vector<float*> X_train, vector<float*> y_train);
  bool save(char* filename);
  bool load(char* filename);
  Neuralnet(char* filename);
};

#endif
