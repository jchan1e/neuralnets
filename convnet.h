#include <vector>
#include <array>
#include <fstream>

#ifndef NN_CONVNET
#define NN_CONVNET

using namespace std;

struct conv_shape
{
  int n;
  vector<int> sizes; // fully connected, else 0
  vector<array<int, 4>> kernels; // {X, Y, kernel size, #kernels} all 0 for std feedforward
  bool sigm;
  float lam;

};

class Convnet
{
//public:
typedef float (*ActivFn)(float z);

private:
// activation functions and their derivatives
  static float sig(float z);
  static float sig_prime(float z);
  static float relu(float z);
  static float relu_prime(float z);
  const ActivFn g;
  const ActivFn g_prime;
  void  G(vector<float> A, vector<float> Z, int n);
  void  G_prime(vector<float> A, vector<float> Z, int n);
// gradient of the output loss
  float gradC(float a, float y);
  void  gradC(vector<float> D, vector<float> A, vector<float> Y, int n);
  void  gradC(vector<vector<float>> D, vector<vector<float>> A, vector<vector<float>> Y, int k, int n);
// forward and back propagation
  void forward_prop(vector<float> X);
  void forward_prop(vector<vector<float>> X);
  //void forward_prop(vector<float> X, vector<vector<float>> lz, vector<vector<float>> la, vector<vector<float>> lb, vector<vector<vector<float>>> lW);
  void back_prop(vector<float> X, vector<float> y);
  void back_prop(vector<vector<float>> X, vector<float> y);
  void back_prop(vector<vector<float>> X, vector<vector<float>> y);
  //void back_prop(vector<float> X, vector<float> y, vector<vector<float>> lz, vector<vector<float>> la, vector<vector<float>> ld, vector<vector<float>> lb, vector<vector<float>> ldb, vector<vector<vector<float>>> lW, vector<vector<vector<float>>> ldw);
  void update_weights(float alpha, float lam_len);
  void update_weights(float alpha, vector<vector<float>> lb, vector<vector<float>> ldb, vector<vector<vector<float>>> lW, vector<vector<vector<float>>> ldw);
  void update_weights(vector<vector<float>> lb, vector<vector<float>> ldb, vector<vector<vector<float>>> lW, vector<vector<vector<float>>> ldW);
public:
  conv_shape s;
  float Lambda; // ratio between actual lambda and alpha

  vector<vector<float>> b; // [L-1][j]
  vector<vector<float>> db;
  vector<vector<vector<float>>> W; // [L-1][i][j]
  vector<vector<vector<float>>> dW;
  vector<vector<float>> z; // [L][i]
  vector<vector<float>> a;
  vector<vector<float>> d;

  vector<vector<vector<float>>> bk; // [L-1][k][h*i+j]
  vector<vector<vector<float>>> dbk;
  vector<vector<vector<vector<float>>>> Wk; // [L-1][k][K][h*i+j]
  vector<vector<vector<vector<float>>>> dWk;
  vector<vector<vector<float>>> zk; // [L][k][h*i+j]
  vector<vector<vector<float>>> ak;
  vector<vector<vector<float>>> dk;

  Convnet(struct conv_shape* S);//, bool sigm=false, float lam=0.05);
  ~Convnet();
  void eval(vector<float> X, vector<float> y);
  void eval(vector<vector<float>> X, vector<float> y);
  void eval(vector<vector<float>> X, vector<vector<float>> y);
  void train(vector<vector<float>> X_train, vector<vector<float>> y_train, int num_epochs=10, float alpha=0.25, float decay=0.0);
  void train(vector<vector<float>> X_train, vector<vector<float>> y_train, vector<vector<float>> X_valid, vector<vector<float>> y_valid, int num_epochs=10, float alpha=0.25, float decay=0.0);
  void train(vector<vector<vector<float>>> X_train, vector<vector<float>> y_train, int num_epochs=10, float alpha=0.25, float decay=0.0);
  void train(vector<vector<vector<float>>> X_train, vector<vector<float>> y_train, vector<vector<vector<float>>> X_valid, vector<vector<float>> y_valid, int num_epochs=10, float alpha=0.25, float decay=0.0);
  void train(vector<vector<vector<float>>> X_train, vector<vector<vector<float>>> y_train, int num_epochs=10, float alpha=0.25, float decay=0.0);
  void train(vector<vector<vector<float>>> X_train, vector<vector<vector<float>>> y_train, vector<vector<vector<float>>> X_valid, vector<vector<vector<float>>> y_valid, int num_epochs=10, float alpha=0.25, float decay=0.0);
  //void train_parallel(vector<vector<float>> X_train, vector<vector<float>> y_train, int num_epochs=10, float alpha=0.25);
  //void train_parallel(vector<vector<float>> X_train, vector<vector<float>> y_train, vector<vector<float>> X_valid, vector<vector<float>> y_valid, int num_epochs=10, float alpha=0.25);
  float loss(vector<vector<float>> X_train, vector<vector<float>> y_train);
  float loss(vector<vector<vector<float>>> X_train, vector<vector<float>> y_train);
  float loss(vector<vector<vector<float>>> X_train, vector<vector<vector<float>>> y_train);
  bool save(char* filename);
  Convnet(char* filename);
};

#endif
