/*
 * @Author: zerollzeng
 * @Date: 2019-10-10 18:07:54
 * @LastEditors: zerollzeng
 * @LastEditTime: 2019-10-16 09:07:41
 */
#ifndef OPENPOSE_GPU_CUDA_HPP
#define OPENPOSE_GPU_CUDA_HPP

#include <utility> // std::pair
#include <vector>
#include <array>
#include <iostream>

namespace op
{
    
    const auto CUDA_NUM_THREADS = 512u;

    void cudaCheck(const int line = -1, const std::string& function = "", const std::string& file = "");

    int getCudaGpuNumber();

    inline unsigned int getNumberCudaBlocks(
        const unsigned int totalRequired, const unsigned int numberCudaThreads = CUDA_NUM_THREADS)
    {
        return (totalRequired + numberCudaThreads - 1) / numberCudaThreads;
    }
    template <typename T>
    void reorderAndNormalize(
        T* targetPtr, const unsigned char* const srcPtr, const int width, const int height, const int channels);

    template <typename T>
    void uCharImageCast(unsigned char* targetPtr, const T* const srcPtr, const int volume);

}

#endif // OPENPOSE_GPU_CUDA_HPP
