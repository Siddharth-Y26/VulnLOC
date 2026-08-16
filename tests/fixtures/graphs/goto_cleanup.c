int open_and_read(const char *path, char *buf, unsigned long len)
{
    int fd = -1;
    int rc = 0;

    fd = do_open(path);
    if (fd < 0) {
        rc = -1;
        goto out;
    }
    if (do_read(fd, buf, len) < 0) {
        rc = -2;
        goto out_close;
    }
    goto out_close;

out_close:
    do_close(fd);
out:
    return rc;
}
