#include "instance.h"

#include "Foundation/NSString.hpp"
#include "MTLDevice.hpp"
#include "MTLResource.hpp"

void Instance::init() {
    device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        error(std::string("Failed to create MTL Default device"));
    }

    commandQueue = device->newCommandQueue();
    totbuf = -1;
    buffers = NULL;
    functionPSO = NULL;
    function = NULL;
    library = NULL;
}

Instance::~Instance() {
    for (int i = 0; i <= totbuf; i++) {
        if (buffers[i].buffer != NULL) {
            buffers[i].buffer->release();
        }
    }
    free(buffers);
 
    if (functionPSO != NULL) {
        functionPSO->release();
    }
    if (function != NULL) {
        function->release();
    }
    if (library != NULL) {
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
    if (library != NULL) {
        library->release();
    }
    library = device->newLibrary(source_code, options, &errPtr);

    if (library == NULL) { 
        std::cout << errPtr->localizedDescription()->utf8String() << std::endl;
    }
}

void Instance::setFunction(const char *funcname) {
    auto funcstring = NS::String::string(funcname, NS::ASCIIStringEncoding);

    if (function != NULL) {
        function->release();
        functionPSO->release();
    }

    function = library->newFunction(funcstring);
    if (function == NULL) { error(std::string("[MetalGPU] Failed to create function")); }

    functionPSO = device->newComputePipelineState(function, &errPtr);
    if(functionPSO == NULL) { error(std::string("[MetalGPU] Failed to create pipeline state")); }
}

void *Instance::createBuffer(int bufsize, int userBufNum) {
    MTL::Buffer *buffer = device->newBuffer(bufsize, MTL::ResourceStorageModeShared);

    totbuf += 1;
    buffers = (BufferStorer*)realloc(buffers, sizeof(BufferStorer) * totbuf + 1);
    
    BufferStorer newBufStore;

    newBufStore.buffer = buffer;
    newBufStore.userBufNum = userBufNum;

    buffers[totbuf] = newBufStore;

    return buffer->contents();
}

void Instance::runFunction(int numThreads) {
    MTL::CommandBuffer *commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();

    encoder->setComputePipelineState(functionPSO);
    for (int i = 0; i <= totbuf; i++) {
        if(buffers[i].buffer != NULL) {
            encoder->setBuffer(buffers[i].buffer, 0, buffers[i].userBufNum);
        }
    }
    MTL::Size gridSize = MTL::Size(numThreads, 1, 1);

    NS::UInteger threadsGroupSize = NS::UInteger(numThreads);
    if(threadsGroupSize > functionPSO->maxTotalThreadsPerThreadgroup()) {
        threadsGroupSize = functionPSO->maxTotalThreadsPerThreadgroup();
    }

    MTL::Size threadsPerGroup = MTL::Size(threadsGroupSize, 1, 1);

    encoder->dispatchThreads(gridSize, threadsPerGroup);

    encoder->endEncoding();
    commandBuffer->commit();

    commandBuffer->waitUntilCompleted();
}

void Instance::releaseBuffer(int bufnum) {
    buffers[bufnum].buffer->release();
    buffers[bufnum].buffer = NULL;
}

