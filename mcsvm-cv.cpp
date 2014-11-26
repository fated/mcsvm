#include "mcsvm.h"
#include <iostream>
#include <fstream>
#include <iomanip>

void ExitWithHelp();
void ParseCommandLine(int argc, char *argv[], char *data_file_name, char *output_file_name);

struct MCSVMParameter param;

int main(int argc, char *argv[]) {
  char data_file_name[256];
  char output_file_name[256];
  struct Problem *prob;
  struct ErrStatistics *errors = NULL;
  int num_correct;
  double avg_brier = 0, avg_logloss = 0, avg_prob = 0;
  int *predict_labels = NULL;
  double *probs = NULL, *brier = NULL, *logloss = NULL;
  const char *error_message;

  ParseCommandLine(argc, argv, data_file_name, output_file_name);
  error_message = CheckMCSVMParameter(&param);

  if (error_message != NULL) {
    std::cerr << error_message << std::endl;
    exit(EXIT_FAILURE);
  }

  prob = ReadProblem(data_file_name);

  if (param.gamma == 0) {
    param.gamma = 1.0 / prob->max_index;
  }

  std::ofstream output_file(output_file_name);
  if (!output_file.is_open()) {
    std::cerr << "Unable to open output file: " << output_file_name << std::endl;
    exit(EXIT_FAILURE);
  }

  predict_labels = new int[prob->num_ex];
  errors = new ErrStatistics;
  errors->num_errors = 0;
  if (param.probability == 1) {
    probs = new double[prob->num_ex];
    brier = new double[prob->num_ex];
    logloss = new double[prob->num_ex];
  }

  std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::high_resolution_clock::now();

  CrossValidation(prob, &param, errors, predict_labels, probs, brier, logloss);

  std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < prob->num_ex; ++i) {
    output_file << prob->y[i] << ' ' << predict_labels[i];

    if (param.probability == 1) {
      avg_prob += probs[i];
      avg_brier += brier[i];
      avg_logloss += logloss[i];
      output_file << ' ' << probs[i];
    }
    output_file << '\n';
  }

  num_correct = prob->num_ex - errors->num_errors;
  std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(4)
            << "CV Accuracy: " << 100.0*num_correct/prob->num_ex << '%'
            << " (" << num_correct << '/' << prob->num_ex << ") " << '\n';
  output_file.close();

  if (param.probability == 1) {
    avg_prob /= prob->num_ex;
    avg_brier /= prob->num_ex;
    avg_logloss /= prob->num_ex;
    std::cout << "Probabilities: " << 100*avg_prob << "%\n"
              << "Brier Score: " << avg_brier << ' ' << "Logarithmic Loss: " << avg_logloss << '\n';
  }

  std::cout << "Time cost: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()/1000.0 << " s\n";

  int num_classes;
  int *labels = GetLabels(prob, &num_classes);
  int *old_labels;
  clone(old_labels, labels, num_classes);
  size_t *index = new size_t[num_classes];
  for (size_t i = 0; i < num_classes; ++i) {
    index[i] = i;
  }
  QuickSortIndex(old_labels, index, 0, static_cast<size_t>(num_classes-1));
  delete[] old_labels;

  std::cout << "\nError Statsitics\n";
  std::cout << " test error : " << 100.0*(errors->num_errors)/prob->num_ex
            << "% (" << errors->num_errors
            << " / " << prob->num_ex
            << ")\n";

  std::cout << " error statistics (correct/predicted)\n" << "     ";

  for (int i = 0; i < num_classes; ++i) {
    std::cout << std::setw(4) << labels[index[i]] << ' ';
  }
  std::cout << '\n';
  for (int i = 0; i < num_classes; ++i) {
    std::cout << std::setw(4) << labels[index[i]] << ' ';
    for (int j = 0; j < num_classes; ++j) {
      std::cout << std::setw(4) << errors->error_statistics[index[i]][index[j]] << ' ';
    }
    std::cout << '\n';
  }
  std::cout << std::endl;

  FreeProblem(prob);
  FreeMCSVMParam(&param);

  for (int i = 0; i < num_classes; ++i) {
    delete[] errors->error_statistics[i];
  }
  delete[] errors->error_statistics;
  delete errors;
  delete[] index;
  delete[] labels;
  delete[] predict_labels;
  if (param.probability == 1) {
    delete[] probs;
    delete[] brier;
    delete[] logloss;
  }
  return 0;
}

void ExitWithHelp() {
  std::cout << "Usage: mcsvm-cv [options] data_file [output_file]\n"
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
            << "  -g gamma : set gamma in kernel function (default 1.0/num_features)\n"
            << "  -r coef0 : set coef0 in kernel function (default 0)\n"
            << "  -v num_folds : set number of folders in cross validation (default 5)\n"
            << "  -b beta : set margin (default 1e-4)\n"
            << "  -w delta : set approximation tolerance for approximate method (default 1e-4)\n"
            << "  -m cachesize : set cache memory size in MB (default 100)\n"
            << "  -e epsilon : set tolerance of termination criterion (default 1e-3)\n"
            << "  -z epsilon0 : set initialize margin (default 1-1e-6)\n"
            << "  -q : turn off quiet mode (no outputs)\n";
  exit(EXIT_FAILURE);
}

void ParseCommandLine(int argc, char **argv, char *data_file_name, char *output_file_name) {
  int i;
  param.save_model = 0;
  param.load_model = 0;
  param.num_folds = 5;
  param.probability = 0;
  InitMCSVMParam(&param);
  SetPrintNull();

  for (i = 1; i < argc; ++i) {
    if (argv[i][0] != '-') break;
    if ((i+1) >= argc)
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
      case 'v': {
        ++i;
        param.num_folds = std::atoi(argv[i]);
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
      case 'w': {
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
      case 'z': {
        ++i;
        param.epsilon0 = std::atof(argv[i]);
        break;
      }
      case 'p': {
        ++i;
        param.probability = std::atoi(argv[i]);
        break;
      }
      case 'q': {
        SetPrintCout();
        break;
      }
      default: {
        std::cerr << "Unknown option: -" << argv[i][1] << std::endl;
        ExitWithHelp();
      }
    }
  }

  if ((i) >= argc)
    ExitWithHelp();
  std::strcpy(data_file_name, argv[i]);
  if ((i+1) < argc) {
    std::strcpy(output_file_name, argv[i+1]);
  } else {
    char *p = std::strrchr(argv[i],'/');
    if (p == NULL) {
      p = argv[i];
    } else {
      ++p;
    }
    std::sprintf(output_file_name, "%s_output", p);
  }

  return;
}