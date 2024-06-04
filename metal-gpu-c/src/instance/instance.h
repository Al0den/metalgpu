#include "../config.h"
#include "../api/utils.h"

struct BufferStorer {
    MTL::Buffer *buffer;
    int userBufNum;
};

class Instance {
    public:
        void init();
        void createLibrary(const char* filename);
        void setFunction(const char *funcname);
        void releaseBuffer(int bufnum);
        void runFunction(int numThreads);

        ~Instance();

        void *createBuffer(int bufsize, int userBufNum);

        MTL::Buffer *testBuffer;

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
