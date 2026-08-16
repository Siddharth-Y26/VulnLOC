int classify(int value, int flags)
{
    if (value < 0) {
        return -1;
    } else if (value == 0) {
        return 0;
    }

    switch (flags & 0x3) {
    case 0:
        return 10;
    case 1:
        return 20;
    default:
        break;
    }
    return value > 100 ? 100 : value;
}
