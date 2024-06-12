#include "instance.h"

#include "Foundation/NSString.hpp"

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
    errPtr = nullptr;
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

void Instance::createLibraryFromString(const char *fileString) {
    NS::String *source_code = NS::String::string(fileString, NS::StringEncoding::UTF8StringEncoding);
    errPtr = nullptr;
    MTL::CompileOptions *options = nullptr;
    if (library != NULL) {
        library->release();
    }
    library = device->newLibrary(source_code, options, &errPtr);
    if (library == NULL) { 
        std::cout << errPtr->localizedDescription()->utf8String() << std::endl;
        return;
    }
}   

void Instance::setFunction(const char *funcname) {
    auto funcstring = NS::String::string(funcname, NS::ASCIIStringEncoding);

    if (function != NULL) {
        function->release();
        functionPSO->release();
    }

    function = library->newFunction(funcstring);
    if (function == NULL) { 
        printf("[MetalGPU] Couldn't create function");
        return;
    }

    functionPSO = device->newComputePipelineState(function, &errPtr);
    if (function == NULL) { 
        std::cout << errPtr->localizedDescription()->utf8String() << std::endl;
        return;
    }
}

int Instance::maxThreadsPerGroup() {
    if (function == NULL) {
        return -1;
    }
    return functionPSO->maxTotalThreadsPerThreadgroup();
}

int Instance::threadExecutionWidth() {
    if (function == NULL) {
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
 
void Instance::runFunction(int *MetalSize, int *requestedBuffers, int numRequestedBuffers) {
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

    NS::UInteger threadsGroupSize = NS::UInteger(MetalSize[0]);
    NS::UInteger threadExecutionWidth = NS::UInteger(MetalSize[1]);

    if(threadsGroupSize > functionPSO->maxTotalThreadsPerThreadgroup()) {
        threadsGroupSize = functionPSO->maxTotalThreadsPerThreadgroup();
    }

    if(threadExecutionWidth > functionPSO->threadExecutionWidth()) {
        threadExecutionWidth = functionPSO->threadExecutionWidth();
    }

    if(threadsGroupSize % threadExecutionWidth != 0) {
        threadsGroupSize = threadsGroupSize - (threadsGroupSize % threadExecutionWidth);
    }

    MTL::Size threadsPerGroup = MTL::Size(threadsGroupSize, threadExecutionWidth, MetalSize[2]);

    encoder->dispatchThreads(gridSize, threadsPerGroup);

    encoder->endEncoding();
    commandBuffer->commit();

    commandBuffer->waitUntilCompleted();
}

void Instance::releaseBuffer(int bufnum) {
    buffers[bufnum].buffer->release();
    buffers[bufnum].buffer = NULL;
}

void *Instance::getBufferPointer(int bufnum) {
    return buffers[bufnum].buffer->contents();
}
