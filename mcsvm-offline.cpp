#include "mcsvm.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

void ExitWithHelp();
void ParseCommandLine(int argc, char *argv[], char *train_file_name, char *test_file_name, char *output_file_name, char *model_file_name);

struct MCSVMParameter param;

int main(int argc, char *argv[]) {
  char train_file_name[256];
  char test_file_name[256];
  char output_file_name[256];
  char model_file_name[256];
  struct Problem *train, *test;
  struct MCSVMModel *model;
  struct ErrStatistics *errors;
  int num_correct;
  const char *error_message;

  ParseCommandLine(argc, argv, train_file_name, test_file_name, output_file_name, model_file_name);
  error_message = CheckMCSVMParameter(&param);

  if (error_message != NULL) {
    std::cerr << error_message << std::endl;
    exit(EXIT_FAILURE);
  }

  train = ReadProblem(train_file_name);
  test = ReadProblem(test_file_name);

  std::ofstream output_file(output_file_name);
  if (!output_file.is_open()) {
    std::cerr << "Unable to open output file: " << output_file_name << std::endl;
    exit(EXIT_FAILURE);
  }

  std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::high_resolution_clock::now();

  if (param.load_model == 1) {
    model = LoadMCSVMModel(model_file_name);
    if (model == NULL) {
      exit(EXIT_FAILURE);
    }
  } else {
    model = TrainMCSVM(train, &param);
  }

  if (param.save_model == 1) {
    if (SaveMCSVMModel(model_file_name, model) != 0) {
      std::cerr << "Unable to save model file" << std::endl;
    }
  }

  // if (param.probability == 1) {
  //   output_file << "                      ";
  //   for (int i = 0; i < model->num_classes; ++i) {
  //     output_file << model->labels[i] << "        ";
  //   }
  //   output_file << '\n';
  // }

  errors = new ErrStatistics;
  errors->num_errors = 0;
  errors->error_statistics = new int*[model->num_classes];
  for (int i = 0; i < model->num_classes; ++i) {
    errors->error_statistics[i] = new int[model->num_classes];
    for (int j = 0; j < model->num_classes; ++j) {
      errors->error_statistics[i][j] = 0;
    }
  }

  for (int i = 0; i < test->num_ex; ++i) {
    int predicted_label = PredictMCSVM(model, test->x[i]);

    output_file << test->y[i] << ' ' << predicted_label << '\n';
    if (predicted_label != test->y[i]) {
      ++errors->num_errors;
    }
    int j;
    for (j = 0; j < model->num_classes; ++j) {
      if (model->labels[j] == predicted_label) {
        break;
      }
    }
    int y;
    for (y = 0; y < model->num_classes; ++y) {
      if (model->labels[y] == test->y[i]) {
        break;
      }
    }
    ++errors->error_statistics[y][j];
  }

  std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::high_resolution_clock::now();

  num_correct = test->num_ex - errors->num_errors;
  std::cout << "Accuracy: " << 100.0*num_correct/test->num_ex << '%'
            << " (" << num_correct << '/' << test->num_ex << ") " << '\n';
  output_file.close();

  std::cout << "Time cost: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()/1000.0 << " s\n";

  int *labels;
  clone(labels, model->labels, model->num_classes);
  size_t *index = new size_t[model->num_classes];
  for (size_t i = 0; i < model->num_classes; ++i) {
    index[i] = i;
  }
  QuickSortIndex(labels, index, 0, static_cast<size_t>(model->num_classes-1));
  delete[] labels;

  std::cout << "\nError Statsitics\n";
  std::cout << " test error : " << 100.0*(errors->num_errors)/test->num_ex
            << "% (" << errors->num_errors
            << " / " << test->num_ex
            << ")\n";

  std::cout << " error statistics (correct/predicted)\n" << "     ";

  for (int i = 0; i < model->num_classes; ++i) {
    std::cout << std::setw(4) << model->labels[index[i]] << ' ';
  }
  std::cout << '\n';
  for (int i = 0; i < model->num_classes; ++i) {
    std::cout << std::setw(4) << model->labels[index[i]] << ' ';
    for (int j = 0; j < model->num_classes; ++j) {
      std::cout << std::setw(4) << errors->error_statistics[index[i]][index[j]] << ' ';
    }
    std::cout << '\n';
  }
  std::cout << std::endl;

  FreeProblem(train);
  FreeProblem(test);
  FreeMCSVMModel(model);
  FreeMCSVMParam(&param);

  for (int i = 0; i < model->num_classes; ++i) {
    delete[] errors->error_statistics[i];
  }
  delete[] errors->error_statistics;
  delete errors;
  delete[] index;

  return 0;
}

void ExitWithHelp() {
  std::cout << "Usage: mcsvm-offline [options] train_file test_file [output_file]\n"
            << "options:\n"
            << "  -t redopt_type : set type of reduced optimization (default 0)\n"
            << "    0 -- exact (EXACT)\n"
            << "    1 -- approximate (APPROX)\n"
            << "    2 -- binary (BINARY)\n"
            << "  -k kernel_type : set type of kernel function (default 2)\n"
            << "    0 -- linear: u'*v\n"
            << "    1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
            << "    2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
            << "    3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
            << "    4 -- precomputed kernel (kernel values in training_set_file)\n"
            << "  -d degree : set degree in kernel function (default 1)\n"
            << "  -g gamma : set gamma in kernel function (default 1)\n"
            << "  -r coef0 : set coef0 in kernel function (default 0)\n"
            << "  -s model_file_name : save model\n"
            << "  -l model_file_name : load model\n"
            << "  -b beta : set beta (default 1e-4)\n"
            << "  -c delta : set delta (default 1e-4)\n"
            << "  -m cachesize : set cache memory size in MB (default 100)\n"
            << "  -e epsilon : set tolerance of termination criterion (default 1e-3)\n"
            << "  -p epsilon0 : set tolerance of termination criterion (default 1-1e-6)\n"
            << "  -q : quiet mode (no outputs)\n";
  exit(EXIT_FAILURE);
}

void ParseCommandLine(int argc, char **argv, char *train_file_name, char *test_file_name, char *output_file_name, char *model_file_name) {
  int i;
  param.save_model = 0;
  param.load_model = 0;
  InitMCSVMParam(&param);
  SetPrintCout();

  for (i = 1; i < argc; ++i) {
    if (argv[i][0] != '-') break;
    if ((i+2) >= argc)
      ExitWithHelp();
    switch (argv[i][1]) {
      case 't': {
        ++i;
        param.redopt_type = std::atoi(argv[i]);
        break;
      }
      case 'k': {
        ++i;
        param.kernel_type = std::atoi(argv[i]);
        break;
      }
      case 's': {
        ++i;
        param.save_model = 1;
        std::strcpy(model_file_name, argv[i]);
        break;
      }
      case 'l': {
        ++i;
        param.load_model = 1;
        std::strcpy(model_file_name, argv[i]);
        break;
      }
      case 'b': {
        ++i;
        param.beta = std::atof(argv[i]);
        break;
      }
      case 'd': {
        ++i;
        param.degree = std::atoi(argv[i]);
        break;
      }
      case 'g': {
        ++i;
        param.gamma = std::atof(argv[i]);
        break;
      }
      case 'r': {
        ++i;
        param.coef0 = std::atof(argv[i]);
        break;
      }
      case 'c': {
        ++i;
        param.delta = std::atof(argv[i]);
        break;
      }
      case 'm': {
        ++i;
        param.cache_size = std::atoi(argv[i]);
        break;
      }
      case 'e': {
        ++i;
        param.epsilon = std::atof(argv[i]);
        break;
      }
      case 'p': {
        ++i;
        param.epsilon0 = std::atof(argv[i]);
        break;
      }
      case 'q': {
        SetPrintNull();
        break;
      }
      default: {
        std::cerr << "Unknown option: -" << argv[i][1] << std::endl;
        ExitWithHelp();
      }
    }
  }

  if ((i+1) >= argc)
    ExitWithHelp();
  std::strcpy(train_file_name, argv[i]);
  std::strcpy(test_file_name, argv[i+1]);
  if ((i+2) < argc) {
    std::strcpy(output_file_name, argv[i+2]);
  } else {
    char *p = std::strrchr(argv[i+1],'/');
    if (p == NULL) {
      p = argv[i+1];
    } else {
      ++p;
    }
    std::sprintf(output_file_name, "%s_output", p);
  }

  return;
}