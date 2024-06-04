#define NS_PRIVATE_IMPLEMENTATION                                                                                                                  •
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include "config.h"
#include "instance/instance.h"

Instance *mainInstance;

int main() {}  

extern "C" {
    void init() {
        mainInstance = new Instance();
        mainInstance->init();
    }

    void createLibrary(const char *filename) {
        mainInstance->createLibrary(filename);
    }

    int createBuffer(int bufsize) {
        int bufferNumber = mainInstance->createBuffer(bufsize);
        return bufferNumber;
    }

    void setFunction(const char *funcname) {
        mainInstance->setFunction(funcname);
    }

    void runFunction(int numThreads, int *requestedBuffers, int numRequestedBuffers) {
        mainInstance->runFunction(numThreads, requestedBuffers, numRequestedBuffers);
    }

    void releaseBuffer(int bufnum) {
        mainInstance->releaseBuffer(bufnum);
    }

    void deleteInstance() {
        delete mainInstance;
        mainInstance = NULL;
    }

    void *getBufferPointer(int bufnum) {
        return mainInstance->getBufferPointer(bufnum);
    }
}


