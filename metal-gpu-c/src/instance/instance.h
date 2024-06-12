#include "../config.h"
#include "../api/utils.h"

struct BufferStorer {
    MTL::Buffer *buffer;
    int bufferNum;
};

class Instance {
    public:
        void init();
        void createLibrary(const char* filename);
        void createLibraryFromString(const char *fileString);
        void setFunction(const char *funcname);
        void releaseBuffer(int bufnum);
        void runFunction(int *MetalSize, int *requestedBuffers, int numRequestedBuffers);

        int maxThreadsPerGroup();
        int threadExecutionWidth();

        ~Instance();

        int createBuffer(int bufsize);
        void *getBufferPointer(int bufnum);

        BufferStorer *buffers;
        int totbuf;

    private:
        MTL::Device *device;
        MTL::CommandQueue *commandQueue;
        MTL::Library *library;

        MTL::Function *function;
        MTL::ComputePipelineState *functionPSO;

        NS::Error *errPtr;
};
