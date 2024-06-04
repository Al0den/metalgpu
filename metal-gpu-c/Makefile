compile:
	clang++ src/*.cpp src/instance/*.cpp src/api/*.cpp -I/Users/alois/Desktop/projects/metal-cpp -I/Users/alois/Desktop/projects/metal-cpp/Metal -framework Metal -framework Foundation -framework Cocoa -o ./build/main.out
run:
	./build/main.out
object:
	clang++ src/*.cpp src/instance/*.cpp src/api/*.cpp -c -fPIC -I/Users/alois/Desktop/projects/metal-cpp -I/Users/alois/Desktop/projects/metal-cpp/Metal
library:
	clang++ *.o -shared -o ./build/lib.so -framework Metal -framework Foundation -framework Cocoa
	cp ./build/lib.so ../metalgpu/src/metalgpu/binaries/lib.so
