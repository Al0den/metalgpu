#define NS_PRIVATE_IMPLEMENTATION                                                                                                                  •
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include "config.h"
#include "instance/instance.h"
#include <mutex>

Instance *mainInstance = nullptr;
int instanceRefCount = 0;
std::mutex instanceMutex;  // Thread safety for singleton management

// Helper function to ensure instance exists
static bool ensureInstance() {
    return mainInstance != nullptr;
}

int main() {}  

extern "C" {
    void init() {
        std::lock_guard<std::mutex> lock(instanceMutex);
        if (mainInstance == nullptr) {
            mainInstance = new Instance();
            mainInstance->init();
        }
        instanceRefCount++;
    }

    void createLibrary(const char *filename) {
        if (!ensureInstance()) return;
        mainInstance->createLibrary(filename);
    }

    int createBuffer(int bufsize) {
        if (!ensureInstance()) return -1;
        int bufferNumber = mainInstance->createBuffer(bufsize);
        return bufferNumber;
    }

    void setFunction(const char *funcname) {
        if (!ensureInstance()) return;
        mainInstance->setFunction(funcname);
    }

    void runFunction(int *MetalSize, int *requestedBuffers, int numRequestedBuffers, bool waitForCompletion) {
        if (!ensureInstance()) return;
        mainInstance->runFunction(MetalSize, requestedBuffers, numRequestedBuffers, waitForCompletion);
    }

    void releaseBuffer(int bufnum) {
        if (!ensureInstance()) return;
        mainInstance->releaseBuffer(bufnum);
    }

    void deleteInstance() {
        std::lock_guard<std::mutex> lock(instanceMutex);
        instanceRefCount--;
        if (instanceRefCount <= 0 && mainInstance != nullptr) {
            delete mainInstance;
            mainInstance = nullptr;
            instanceRefCount = 0;  // Safety reset
        }
    }

    void *getBufferPointer(int bufnum) {
        if (!ensureInstance()) return nullptr;
        return mainInstance->getBufferPointer(bufnum);
    }

    void createLibraryFromString(const char* string) {
        if (!ensureInstance()) return;
        mainInstance->createLibraryFromString(string);
    }

    int maxThreadsPerGroup() {
        if (!ensureInstance()) return 0;
        return mainInstance->maxThreadsPerGroup();
    }

    int threadExecutionWidth() {
        if (!ensureInstance()) return 0;
        return mainInstance->threadExecutionWidth();
    }
}


