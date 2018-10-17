#include <vector>
#include <fstream>

#ifndef NN_NEURALNET
#define NN_NEURALNET

using namespace std;

struct shape
{
  int n;
  int* sizes;
  //shape(int x)
  //{
  //  n = x;
  //  sizes = new int[n];
  //}
  ~shape()
  {
    delete[] sizes;
  }
};

class Neuralnet
{
//public:
private:
// activation function and its derivative
  float g(float z);
  void  g(float* A, float* Z, int n);
  float g_prime(float z);
  void  g_prime(float* A, float* Z, int n);
// gradient of the output loss
  float gradC(float a, float y);
  void  gradC(float* D, float* A, float* Y, int n);
// forward and back propagation
  void forward_prop(float* X);
  void forward_prop(float* X, float** lz, float** la, float** lb, float*** lW);
  void back_prop(float* X, float* y);
  void back_prop(float* X, float* y, float** lz, float** la, float** ld, float** lb, float** ldb, float*** lW, float*** ldw);
  void update_weights(float eta);
  void update_weights(float eta, float** lb, float** ldb, float*** lW, float*** ldw);
  void update_weights(float** lb, float** ldb, float*** lW, float*** ldW);
public:
  shape s;
  int n_layers;
  float** b;
  float** db;
  float*** W;
  float*** dW;
  float** z;
  float** a;
  float** d;

  Neuralnet(struct shape* S);
  ~Neuralnet();
  void eval(float* X, float* y);
  void train(vector<float*> X_train, vector<float*> y_train, int num_epochs=10, float eta=0.25);
  void train(vector<float*> X_train, vector<float*> y_train, vector<float*> X_valid, vector<float*> y_valid, int num_epochs=10, float eta=0.25);
  void train_parallel(vector<float*> X_train, vector<float*> y_train, int num_epochs=10, float eta=0.25);
  void train_parallel(vector<float*> X_train, vector<float*> y_train, vector<float*> X_valid, vector<float*> y_valid, int num_epochs=10, float eta=0.25);
  float loss(vector<float*> X_train, vector<float*> y_train);
  bool save(char* filename);
  Neuralnet(char* filename);
};

#endif
