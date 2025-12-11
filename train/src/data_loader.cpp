#include "../include/data_loader.h"
#include <fstream>
#include <iostream>
#include <cstring>

DataLoader::DataLoader(const std::string& data_dir, int batch_size)
    : data_dir_(data_dir), batch_size_(batch_size), current_idx_(0), 
      total_images_(0), img_size_(3 * 32 * 32), current_batch_(nullptr)
{
    load_dataset();
    current_batch_ = new float[batch_size_ * img_size_];
}

DataLoader::~DataLoader() {
    delete[] current_batch_;
}

void DataLoader::load_cifar_file(const std::string& filename) {
    std::ifstream fin(filename, std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "[ERROR] Cannot open " << filename << std::endl;
        return;
    }
    
    const int record_bytes = 1 + img_size_;
    std::vector<unsigned char> buf(record_bytes);
    
    while (fin.read(reinterpret_cast<char*>(buf.data()), record_bytes)) {
        labels_.push_back(buf[0]);
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < 32; ++h) {
                for (int w = 0; w < 32; ++w) {
                    int src_idx = 1 + c * 32 * 32 + h * 32 + w;
                    images_.push_back(static_cast<float>(buf[src_idx]) / 255.0f);
                }
            }
        }
        total_images_++;
    }
}

void DataLoader::load_dataset() {
    for (int i = 1; i <= 5; ++i) {
        std::string filename = data_dir_ + "/data_batch_" + std::to_string(i) + ".bin";
        load_cifar_file(filename);
    }
    std::cout << "[INFO] Loaded " << total_images_ << " images from " << data_dir_ << std::endl;
}

bool DataLoader::has_next() const {
    return current_idx_ < total_images_;
}

float* DataLoader::next_batch() {
    int actual_batch = (current_idx_ + batch_size_ <= total_images_) ? 
                       batch_size_ : (total_images_ - current_idx_);
    
    for (int i = 0; i < actual_batch; ++i) {
        if (current_idx_ + i < total_images_) {
            std::memcpy(current_batch_ + i * img_size_,
                       images_.data() + (current_idx_ + i) * img_size_,
                       img_size_ * sizeof(float));
        }
    }
    
    current_idx_ += actual_batch;
    return current_batch_;
}

void DataLoader::reset() {
    current_idx_ = 0;
}

int DataLoader::get_num_batches() const {
    return (total_images_ + batch_size_ - 1) / batch_size_;
}

