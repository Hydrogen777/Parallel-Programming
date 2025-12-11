#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>

class DataLoader {
public:
    DataLoader(const std::string& data_dir, int batch_size);
    ~DataLoader();
    
    bool has_next() const;
    float* next_batch();
    void reset();
    int get_num_batches() const;
    
private:
    std::string data_dir_;
    int batch_size_;
    std::vector<float> images_;
    std::vector<unsigned char> labels_;
    int current_idx_;
    int total_images_;
    int img_size_;
    float* current_batch_;
    
    void load_cifar_file(const std::string& filename);
    void load_dataset();
};

#endif

