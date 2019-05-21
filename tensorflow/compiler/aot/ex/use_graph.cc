#include <iostream>
#include "tensorflow/compiler/aot/ex/test_graph.h"

using namespace std;

int main() {
    MatMulAndAddComp compute;
    
    const float args[8] = {1, 2, 3, 4, 5, 6, 7, 8};    
    std::copy(args + 0, args + 4, compute.arg0_data());
    std::copy(args + 4, args + 8, compute.arg1_data());
    
    compute.Run();

    const MatMulAndAddComp& compute_const = compute;
    
    for(int i=0; i<4;i++) {
        cout << compute_const.arg0(i/2, i%2) << " " << args[i] << endl;
    }
    for(int i=0; i<4;i++) {
        cout << compute_const.result0(i/2, i%2) << " ";
        cout << compute_const.result1(i/2, i%2) << endl;
    }

    return 0;
}
