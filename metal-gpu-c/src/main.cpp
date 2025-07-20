#define NS_PRIVATE_IMPLEMENTATION                                                                                                                  •
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include "config.h"
#include "instance/instance.h"

int main() {}  

extern "C" {
    // Returns pointer to new Instance object
    Instance* init() {
        Instance* instance = new Instance();
        instance->init();
        return instance;
    }

    void createLibrary(Instance* instance, const char *filename) {
        if (instance == nullptr) return;
        instance->createLibrary(filename);
    }

    int createBuffer(Instance* instance, int bufsize) {
        if (instance == nullptr) return -1;
        return instance->createBuffer(bufsize);
    }

    void setFunction(Instance* instance, const char *funcname) {
        if (instance == nullptr) return;
        instance->setFunction(funcname);
    }

    void runFunction(Instance* instance, int *MetalSize, int *requestedBuffers, int numRequestedBuffers, bool waitForCompletion) {
        if (instance == nullptr) return;
        instance->runFunction(MetalSize, requestedBuffers, numRequestedBuffers, waitForCompletion);
    }

    void releaseBuffer(Instance* instance, int bufnum) {
        if (instance == nullptr) return;
        instance->releaseBuffer(bufnum);
    }

    void deleteInstance(Instance* instance) {
        if (instance != nullptr) {
            delete instance;
        }
    }

    void *getBufferPointer(Instance* instance, int bufnum) {
        if (instance == nullptr) return nullptr;
        return instance->getBufferPointer(bufnum);
    }

    void createLibraryFromString(Instance* instance, const char* string) {
        if (instance == nullptr) return;
        instance->createLibraryFromString(string);
    }

    int maxThreadsPerGroup(Instance* instance) {
        if (instance == nullptr) return 0;
        return instance->maxThreadsPerGroup();
    }

    int threadExecutionWidth(Instance* instance) {
        if (instance == nullptr) return 0;
        return instance->threadExecutionWidth();
    }
}


