#ifndef LIBVM_MCSVM_H_
#define LIBVM_MCSVM_H_

#include "utilities.h"

enum { EXACT, APPROX, BINARY };  // redopt_type
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED };  // kernel_type

struct MCSVMParameter {
  int redopt_type;  // reduced optimization type
  int kernel_type;
  int save_model;
  int load_model;
  int degree;  // for poly
  double gamma;  // for poly/rbf/sigmoid
  double coef0;  // for poly/sigmoid
  double beta;
  int cache_size; // in Mb
  double epsilon;
  double epsilon0;
  double delta;
  // enum RedOptType redopt_type;
};

struct MCSVMModel {
  struct MCSVMParameter param;
  int num_ex;
  int num_classes;  // number of classes (k)
  int total_sv;
  int *labels;
  int *votes_weight;
  int *num_svs;
  int *sv_indices;
  double **tau;
  struct Node **svs;
};

MCSVMModel *TrainMCSVM(const struct Problem *prob, const struct MCSVMParameter *param);
double PredictMCSVM(const struct MCSVMModel *model, const struct Node *x);

// int SaveMCSVMModel(std::ofstream &model_file, const struct MCSVMModel *model);
// MCSVMModel *LoadMCSVMModel(std::ifstream &model_file);
int SaveMCSVMModel(const char *file_name, const struct MCSVMModel *model);
MCSVMModel *LoadMCSVMModel(const char *file_name);
void FreeMCSVMModel(struct MCSVMModel *model);

void FreeMCSVMParam(struct MCSVMParameter *param);
void InitMCSVMParam(struct MCSVMParameter *param);
const char *CheckMCSVMParameter(const struct MCSVMParameter *param);

void SetPrintNull();
void SetPrintCout();

#endif  // LIBVM_MCSVM_H_