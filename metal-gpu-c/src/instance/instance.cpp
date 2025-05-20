#include "instance.h"

#include <cstring>
#include <iostream>
#include <cmath> // Added for std::sqrt and std::floor

#include "Foundation/NSString.hpp"

void Instance::init() {
    device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        throw std::runtime_error("No Metal device found");
    }

    commandQueue = device->newCommandQueue();
    totbuf = -1;
    buffers = nullptr;
    functionPSO = nullptr;
    function = nullptr;
    library = nullptr;
    errPtr = nullptr;
}

Instance::~Instance() {
    for (int i = 0; i <= totbuf; i++) {
        if (buffers[i].buffer != nullptr) {
            buffers[i].buffer->release();
        }
    }
    free(buffers);
 
    if (functionPSO != nullptr) {
        functionPSO->release();
    }
    if (function != nullptr) {
        function->release();
    }
    if (library != nullptr) {
        library->release();
    }
}

void Instance::createLibrary(const char *filename) {
    std::ifstream file;
    file.open(filename);
    std::stringstream reader;
    reader << file.rdbuf();
    std::string raw_string = reader.str();
    NS::String *source_code = NS::String::string(raw_string.c_str(), NS::StringEncoding::UTF8StringEncoding);

    errPtr = nullptr;
    MTL::CompileOptions *options = nullptr;

    if (library != nullptr) {
        library->release();
    }
    library = device->newLibrary(source_code, options, &errPtr);

    if (library == nullptr) { 
        std::cout << errPtr->localizedDescription()->utf8String() << std::endl;
    }
}

void Instance::createLibraryFromString(const char *fileString) {
    NS::String *source_code = NS::String::string(fileString, NS::StringEncoding::UTF8StringEncoding);
    errPtr = nullptr;
    MTL::CompileOptions *options = nullptr;
    if (library != nullptr) {
        library->release();
    }
    library = device->newLibrary(source_code, options, &errPtr);
    if (library == nullptr) { 
        std::cout << errPtr->localizedDescription()->utf8String() << std::endl;
        return;
    }
}   

void Instance::setFunction(const char *funcname) {
    auto funcstring = NS::String::string(funcname, NS::ASCIIStringEncoding);

    if (function != nullptr) {
        function->release();
        functionPSO->release();
    }

    function = library->newFunction(funcstring);
    if (function == nullptr) { 
        printf("[MetalGPU] Couldn't create function");
        return;
    }

    functionPSO = device->newComputePipelineState(function, &errPtr);
    if (function == nullptr) { 
        std::cout << errPtr->localizedDescription()->utf8String() << std::endl;
        return;
    }
}

int Instance::maxThreadsPerGroup() {
    if (function == nullptr) {
        return -1;
    }
    return functionPSO->maxTotalThreadsPerThreadgroup();
}

int Instance::threadExecutionWidth() {
    if (function == nullptr) {
        return -1;
    }
    return functionPSO->threadExecutionWidth();
}

int Instance::createBuffer(int bufsize) {
    MTL::Buffer *buffer = device->newBuffer(bufsize, MTL::ResourceStorageModeShared);

    totbuf += 1;

    BufferStorer newBufStore;

    newBufStore.buffer = buffer;
    newBufStore.bufferNum = totbuf;

    buffers = (BufferStorer*)realloc(buffers, sizeof(BufferStorer) * totbuf + 1);
    buffers[totbuf] = newBufStore;

    return totbuf;
}
 
void Instance::runFunction(int *MetalSize, int *requestedBuffers, int numRequestedBuffers, bool waitForCompletion) {
    MTL::CommandBuffer *commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(functionPSO);
    for(int i = 0; i < numRequestedBuffers; i++) {
        if(requestedBuffers[i] == -1 ) {
            continue; 
        }
        encoder->setBuffer(buffers[requestedBuffers[i]].buffer, 0, i);
    }

    MTL::Size gridSize = MTL::Size(MetalSize[0], MetalSize[1], MetalSize[2]);

    // Corrected calculation for threadsPerGroup
    NS::UInteger psoTEW = functionPSO->threadExecutionWidth();
    NS::UInteger psoMTTPTG = functionPSO->maxTotalThreadsPerThreadgroup();

    // Ensure PSO values are somewhat sane (should be >0 from a valid PSO)
    if (psoTEW == 0) psoTEW = 1; 
    if (psoMTTPTG == 0) psoMTTPTG = 1;

    NS::UInteger tgpWidth, tgpHeight, tgpDepth;

    if (gridSize.height == 1 && gridSize.depth == 1) { // 1D dispatch
        tgpWidth = psoMTTPTG;
        tgpHeight = 1;
        tgpDepth = 1;
    } else if (gridSize.depth == 1) { // 2D dispatch
        tgpWidth = psoTEW;
        if (tgpWidth == 0) tgpWidth = 1; // Safety check
        
        tgpHeight = psoMTTPTG / tgpWidth;
        if (tgpHeight == 0) tgpHeight = 1; // Ensure height is at least 1
        tgpDepth = 1;
    } else { // 3D dispatch
        tgpWidth = psoTEW;
        if (tgpWidth == 0) tgpWidth = 1; // Safety check

        NS::UInteger threadsForHxD = psoMTTPTG / tgpWidth;
        if (threadsForHxD == 0) threadsForHxD = 1;
        
        tgpHeight = static_cast<NS::UInteger>(std::floor(std::sqrt(static_cast<double>(threadsForHxD))));
        if (tgpHeight == 0) tgpHeight = 1;
        
        tgpDepth = threadsForHxD / tgpHeight;
        if (tgpDepth == 0) tgpDepth = 1;
    }

    // Ensure all dimensions are at least 1, as MTL::Size requires.
    if (tgpWidth == 0) tgpWidth = 1;
    if (tgpHeight == 0) tgpHeight = 1;
    if (tgpDepth == 0) tgpDepth = 1;
    
    MTL::Size threadsPerGroup = MTL::Size(tgpWidth, tgpHeight, tgpDepth);

    encoder->dispatchThreads(gridSize, threadsPerGroup);

    encoder->endEncoding();
    commandBuffer->commit();

    if(waitForCompletion) {
        commandBuffer->waitUntilCompleted();
    }

    return;

}

void Instance::releaseBuffer(int bufnum) {
    buffers[bufnum].buffer->release();
    buffers[bufnum].buffer = nullptr;
}

void *Instance::getBufferPointer(int bufnum) {
    return buffers[bufnum].buffer->contents();
}
