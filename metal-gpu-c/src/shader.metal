#include <metal_stdlib>

using namespace metal;

kernel void mult(device int* out [[buffer(0)]], uint id  [[thread_position_in_grid]]) {
    out[id] = out[id] * 2;
};

kernel void adder(device int* arr1 [[buffer(0)]], device int* arr2 [[buffer(1)]], device int* arr3 [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    arr3[id] = arr2[id] + arr1[id];
};

kernel void sqrt_func(device float *arr [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    arr[id] = sqrt(arr[id]);
};

