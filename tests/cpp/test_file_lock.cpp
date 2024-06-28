#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>

#include "merlin/filelock.hpp"
#include "merlin/logger.hpp"

using namespace merlin;

int main(int argc, char * argv[]) {
    if (argc != 2) {
        Fatal<std::invalid_argument>("Expected exactly 1 argument too run. Enter \"1\" for creating the file, \"2\" "
                                     "for reading the file, and anything for append content to the file.");
    }
    std::string arg(argv[1]);
    if (arg.compare("1") == 0) {
        FILE * fp_init = fopen("monkey.txt", "wb");
        std::fwrite("mamemimo\n", sizeof(char), 9, fp_init);
        std::fclose(fp_init);
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
    FILE * fp;
    if (arg.compare("2") == 0) {
        fp = fopen("monkey.txt", "rb");
    } else {
        fp = fopen("monkey.txt", "ab");
    }
    merlin::FileLock flock(fp);
    if (arg.compare("2") == 0) {
        flock.lock_shared();
    } else {
        flock.lock();
    }
    if (fp == NULL) {
        Fatal<std::runtime_error>("Cannot open file.\n");
    }
    std::this_thread::sleep_for(std::chrono::seconds(10));
    if (arg.compare("2") == 0) {
        int written_char = std::fwrite("abcdefg\n", sizeof(char), 8, fp);
        std::fflush(fp);
        Message("Written character: %d.\n", written_char);
    }
    flock.unlock();
    std::fclose(fp);
    std::chrono::time_point<std::chrono::high_resolution_clock> stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = stop - start;
    Message("Elapsed time: %.4f.\n", duration.count());
}
