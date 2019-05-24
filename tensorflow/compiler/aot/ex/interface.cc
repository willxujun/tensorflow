/* reads .npy files and convert them to tensorflow::Tensor */
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <google/protobuf/repeated_field.h>
#include "common_tensor.pb.h"
using namespace std;

void list_common_tensors(const CommonTensors & common_tensors) {
    for(int i=0; i<common_tensors.data_size(); i++) {
        const CommonTensor& common_tensor = common_tensors.data(i);
        for(int j=0; j<common_tensor.data_size(); j++)
            cout << common_tensor.data(j) << endl;
    }
}

std::vector<float> to_vector(const CommonTensor& tensor) {
    const RepeatedField<float>& dat = tensor.data();
    std::vector<float> ret;
    std::copy(dat.begin(), dat.end(), ret.begin());
    return ret;
}

int main() {
    CommonTensors tensors;
    fstream in_file;
    in_file.open("./save/common_tensors.pb", ios::in | ios::binary);
    tensors.ParseFromIstream(&in_file);
    list_common_tensors(tensors);
    return 0;
}