#define NS_PRIVATE_IMPLEMENTATION                                                                                                                  •
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include "config.h"
#include "instance/instance.h"

Instance *mainInstance;

int main() {
    mainInstance = new Instance();
    mainInstance->init();

    const char *fileName = "./src/test.metal";
    mainInstance->createLibrary(fileName);

    const char* funcName = "adder";
    mainInstance->setFunction(funcName);
    int n = 100;

    int bufsize = sizeof(int) * n;

    mainInstance->createBuffer(bufsize, 0);
    mainInstance->createBuffer(bufsize, 1);
    mainInstance ->createBuffer(bufsize, 2);
    
    int *buf0 = (int*)mainInstance->buffers[0].buffer->contents();
    int *buf1 = (int*)mainInstance->buffers[1].buffer->contents();
    int *buf2 = (int*)mainInstance->buffers[2].buffer->contents();

    for(int i=0; i<n; i++) {
        buf0[i] = i;
        buf1[i] = 2;
    }
  
    mainInstance->runFunction(n);

    for(int i=0; i<n; i++) {
        std::cout << buf2[i] << std::endl;
    } 
}  

extern "C" {
    void init() {
        mainInstance = new Instance();
        mainInstance->init();
    }

    void createLibrary(const char *filename) {
        mainInstance->createLibrary(filename);
    }

    void *createBuffer(int bufsize, int userBufNum) {
        return mainInstance->createBuffer(bufsize, userBufNum);
    }

    void setFunction(const char *funcname) {
        mainInstance->setFunction(funcname);
    }

    void runFunction(int numThreads) {
        mainInstance->runFunction(numThreads);
    }

    void releaseBuffer(int bufnum) {
        mainInstance->releaseBuffer(bufnum);
    }

    void deleteInstance() {
        delete mainInstance;
        mainInstance = NULL;
    }

    void setBufferValue(int bufnum, int value) {
        printf("%d-%d\n", bufnum, value);
        int *arr = (int*)mainInstance->testBuffer->contents();
        arr[0] = value;
    }
}


