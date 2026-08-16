void alias_and_copy(char *dst, const char *src, unsigned long n)
{
    char **handle = &dst;
    const char *cursor = src;

    if (dst == src) {
        return;
    }
    while (n-- > 0) {
        *(*handle)++ = *cursor++;
    }
    (*handle)[0] = '\0';
}
