/*
 * @Author: zerollzeng
 * @Date: 2019-10-10 18:07:54
 * @LastEditors: zerollzeng
 * @LastEditTime: 2019-10-10 18:07:54
 */
#ifndef RESIZE_AND_MERGE_HPP
#define RESIZE_AND_MERGE_HPP

#include <vector>
#include <array>

#include "cuda_runtime.h"

namespace op
{

    template <typename T>
    __global__ void resizeKernel(
        T* targetPtr, const T* const sourcePtr, const int widthSource, const int heightSource, const int widthTarget,
        const int heightTarget);
    
    template <typename T>
    void resizeAndMergeCpu(
        T* targetPtr, const std::vector<const T*>& sourcePtrs, const std::array<int, 4>& targetSize,
        const std::vector<std::array<int, 4>>& sourceSizes, const std::vector<T>& scaleInputToNetInputs = {1.f});

    // Windows: Cuda functions do not include OP_API
    template <typename T>
    void resizeAndMergeGpu(
        T* 
        
        targetPtr, const std::vector<const T*>& sourcePtrs, const std::array<int, 4>& targetSize,
        const std::vector<std::array<int, 4>>& sourceSizes, const std::vector<T>& scaleInputToNetInputs = {1.f});

    // Windows: OpenCL functions do not include OP_API
    template <typename T>
    void resizeAndMergeOcl(
        T* targetPtr, const std::vector<const T*>& sourcePtrs, std::vector<T*>& sourceTempPtrs,
        const std::array<int, 4>& targetSize, const std::vector<std::array<int, 4>>& sourceSizes,
        const std::vector<T>& scaleInputToNetInputs = {1.f}, const int gpuID = 0);

    // Functions for cvMatToOpInput/cvMatToOpOutput
    template <typename T>
    void resizeAndPadRbgGpu(
        T* targetPtr, const T* const srcPtr, const int sourceWidth, const int sourceHeight,
        const int targetWidth, const int targetHeight, const T scaleFactor);

    template <typename T>
    void resizeAndPadRbgGpu(
        T* targetPtr, const unsigned char* const srcPtr, const int sourceWidth, const int sourceHeight,
        const int targetWidth, const int targetHeight, const T scaleFactor);
}
#endif