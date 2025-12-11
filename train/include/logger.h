#ifndef LOGGER_H
#define LOGGER_H

#include <string>

class Logger {
public:
    Logger(const std::string& log_file);
    ~Logger();
    
    void log_training_start(int num_epochs, int batch_size, float learning_rate);
    void log_epoch(int epoch, float avg_loss);
    void log_training_end();
    
private:
    std::string log_file_;
    void write_log(const std::string& message);
};

#endif

