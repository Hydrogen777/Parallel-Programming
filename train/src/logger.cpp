#include "../../include/logger.h"
#include <fstream>
#include <iostream>
#include <ctime>
#include <iomanip>
#include <sstream>

Logger::Logger(const std::string& log_file) : log_file_(log_file) {
    std::ofstream fout(log_file_, std::ios::app);
    if (fout.is_open()) {
        fout.close();
    }
}

Logger::~Logger() {}

void Logger::write_log(const std::string& message) {
    std::ofstream fout(log_file_, std::ios::app);
    if (fout.is_open()) {
        auto now = std::time(nullptr);
        auto tm = *std::localtime(&now);
        fout << "[" << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "] " 
             << message << std::endl;
        fout.close();
    }
}

void Logger::log_training_start(int num_epochs, int batch_size, float learning_rate) {
    std::ostringstream oss;
    oss << "Training started - Epochs: " << num_epochs 
        << ", Batch Size: " << batch_size 
        << ", Learning Rate: " << learning_rate;
    write_log(oss.str());
}

void Logger::log_epoch(int epoch, float avg_loss) {
    std::ostringstream oss;
    oss << "Epoch " << epoch << " - Avg Loss: " << avg_loss;
    write_log(oss.str());
}

void Logger::log_training_end() {
    write_log("Training completed");
}

