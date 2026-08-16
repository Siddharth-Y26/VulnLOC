#include <stddef.h>
#define MAX(a, b) \
    ((a) > (b) ? (a) : (b))

/* multi-line
   comment */
static size_t clamp_len(size_t want, size_t cap)
{
    size_t n = MAX(want, 1);
#ifdef STRICT
    if (n > cap) {
        n = cap;
    }
#endif
    return n;
}
