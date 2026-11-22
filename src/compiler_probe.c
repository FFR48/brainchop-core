/* Auto-selection must test the runtime ABI, not just whether main() links.
 * MSVC-targeting clang cannot supply the POSIX interfaces used by this build;
 * MinGW-targeting clang remains eligible, including its LTO/lld fast path. */
#if defined(_WIN32) && !defined(__MINGW32__)
#error "brainchop requires a MinGW-targeting compiler on Windows"
#endif
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#include <time.h>
#include <pthread.h>
int main(void) {
    struct timespec now;
    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    if (pthread_mutex_lock(&lock)) return 1;
    if (pthread_mutex_unlock(&lock)) return 1;
    return clock_gettime(CLOCK_MONOTONIC, &now);
}
