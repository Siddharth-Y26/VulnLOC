r"""Functions to try the Findings tab against — written by hand, none from the corpus.

**Asked for by the project owner on 2026-10-12** ("examples of codes to test it against").
Ten have a weakness written into a named line; seven have none intended, five of them the
fixed twin of a vulnerable one. The second kind matters as much as the first: the model
ranks lines in every function it is given and a pattern is a text match, so a harmless
function is where a reader sees what neither can say.

Three things these are **not**:

- **Not corpus text.** A real Big-Vul function on screen beside a ranker invites the
  misreading ADR-006 came from (see :func:`frontend.pipeline.sample_functions`).
- **Not labels.** ``lines`` is where the author wrote the weakness in. The model is sent
  only the code; the page compares its ranking with ``lines`` after the fact.
- **Not a measurement.** Seventeen functions written by one author are an illustration.
  "No weakness intended" is the claim — nobody audited the seven.

They carry no comments: the released model strips comments before it sees a line, so a
comment would be a line the reader sees and the model does not.
"""

from __future__ import annotations

from typing import Any

__all__ = ["EXAMPLES", "GROUPS", "examples"]

GROUPS = {
    "weakness": "A weakness was written in",
    "none": "No weakness intended",
    "pipeline": "Pipeline samples",
}
"""Sample groups, in the order the page lists them."""


def _example(name: str, note: str, code: str, lines: tuple[int, ...] = ()) -> dict[str, Any]:
    return {
        "name": name,
        "note": note,
        "code": code.strip("\n") + "\n",
        "group": "weakness" if lines else "none",
        "lines": list(lines),
    }


EXAMPLES: tuple[dict[str, Any], ...] = (
    _example(
        "Buffer overflow",
        "strcpy copies name into a 64-byte buffer and nothing bounds its length.",
        r"""
void set_hostname(struct config *cfg, const char *name)
{
    char buf[64];
    strcpy(buf, name);
    normalize(buf);
    cfg->hostname = strdup(buf);
}
""",
        lines=(4,),
    ),
    _example(
        "Format string",
        "line holds text taken from user and path, and it is passed to fprintf as the format.",
        r"""
void log_request(FILE *log, const char *user, const char *path)
{
    char line[256];
    snprintf(line, sizeof(line), "%s requested %s\n", user, path);
    fprintf(log, line);
    fflush(log);
}
""",
        lines=(5,),
    ),
    _example(
        "Use after free",
        "s is freed on line 5 and its flags are read on the next line.",
        r"""
int session_close(struct session *s)
{
    int fd = s->fd;
    free(s->buffer);
    free(s);
    if (s->flags & SESSION_LOGGED)
        log_close(fd);
    return close(fd);
}
""",
        lines=(6,),
    ),
    _example(
        "Integer overflow",
        "count * sizeof(struct record) can wrap for a large count, and the loop below still "
        "writes count records.",
        r"""
struct record *records_read(FILE *in, size_t count)
{
    struct record *items;
    size_t i;
    items = malloc(count * sizeof(struct record));
    if (items == NULL)
        return NULL;
    for (i = 0; i < count; i++) {
        if (fread(&items[i], sizeof(struct record), 1, in) != 1) {
            free(items);
            return NULL;
        }
    }
    return items;
}
""",
        lines=(5,),
    ),
    _example(
        "Off-by-one",
        "The loop writes len + 1 bytes into the len bytes allocated on line 3.",
        r"""
char *copy_token(const char *src, size_t len)
{
    char *dst = malloc(len);
    size_t i;
    if (dst == NULL)
        return NULL;
    for (i = 0; i <= len; i++)
        dst[i] = src[i];
    return dst;
}
""",
        lines=(7, 8),
    ),
    _example(
        "NULL dereference",
        "malloc can return NULL on line 3, and n is written through here without a check.",
        r"""
struct node *list_push(struct node *head, int value)
{
    struct node *n = malloc(sizeof(struct node));
    n->value = value;
    n->next = head;
    return n;
}
""",
        lines=(4,),
    ),
    _example(
        "Double free",
        "When parse_config fails, text is freed on line 7 and again on line 13.",
        r"""
int load_config(const char *path, struct config *out)
{
    char *text = read_file(path);
    if (text == NULL)
        return -1;
    if (parse_config(text, out) != 0) {
        free(text);
        goto fail;
    }
    free(text);
    return 0;
fail:
    free(text);
    return -1;
}
""",
        lines=(7, 13),
    ),
    _example(
        "Command injection",
        "dir is pasted into the command on line 4, and the string is handed to a shell.",
        r"""
int archive_directory(const char *dir)
{
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "tar -czf backup.tgz %s", dir);
    return system(cmd);
}
""",
        lines=(5,),
    ),
    _example(
        "Trusted length field",
        "claimed comes from the message itself and is never compared with msg_len, so memcpy "
        "can read past the end of msg.",
        r"""
int echo_reply(struct conn *c, const unsigned char *msg, size_t msg_len)
{
    unsigned short claimed;
    unsigned char *reply;
    if (msg_len < 3)
        return -1;
    claimed = (msg[1] << 8) | msg[2];
    reply = malloc(3 + claimed);
    if (reply == NULL)
        return -1;
    reply[0] = MSG_ECHO_REPLY;
    reply[1] = msg[1];
    reply[2] = msg[2];
    memcpy(reply + 3, msg + 3, claimed);
    conn_write(c, reply, 3 + claimed);
    free(reply);
    return 0;
}
""",
        lines=(14,),
    ),
    _example(
        "Unchecked index (C++)",
        "index comes from the caller and is never compared with fields_.size().",
        r"""
std::string Packet::fieldAt(size_t index) const
{
    const Field &field = fields_[index];
    return std::string(field.data, field.length);
}
""",
        lines=(3,),
    ),
    _example(
        "Bounded copy",
        "The buffer-overflow sample with a length check before anything is copied.",
        r"""
int set_hostname(struct config *cfg, const char *name)
{
    char buf[64];
    size_t len = strlen(name);
    if (len >= sizeof(buf))
        return -1;
    memcpy(buf, name, len + 1);
    normalize(buf);
    cfg->hostname = strdup(buf);
    return cfg->hostname != NULL ? 0 : -1;
}
""",
    ),
    _example(
        "Literal format",
        "The format-string sample with fputs: the message is written as data.",
        r"""
void log_request(FILE *log, const char *user, const char *path)
{
    char line[256];
    snprintf(line, sizeof(line), "%s requested %s\n", user, path);
    fputs(line, log);
    fflush(log);
}
""",
    ),
    _example(
        "Read before free",
        "The use-after-free sample reordered: the flag is copied out of s before s is freed.",
        r"""
int session_close(struct session *s)
{
    int fd = s->fd;
    int logged = s->flags & SESSION_LOGGED;
    free(s->buffer);
    free(s);
    if (logged)
        log_close(fd);
    return close(fd);
}
""",
    ),
    _example(
        "Checked size",
        "The integer-overflow sample with count refused before the multiplication.",
        r"""
struct record *records_read(FILE *in, size_t count)
{
    struct record *items;
    size_t i;
    if (count > SIZE_MAX / sizeof(struct record))
        return NULL;
    items = malloc(count * sizeof(struct record));
    if (items == NULL)
        return NULL;
    for (i = 0; i < count; i++) {
        if (fread(&items[i], sizeof(struct record), 1, in) != 1) {
            free(items);
            return NULL;
        }
    }
    return items;
}
""",
    ),
    _example(
        "Checked length field",
        "The trusted-length sample with claimed compared against msg_len first.",
        r"""
int echo_reply(struct conn *c, const unsigned char *msg, size_t msg_len)
{
    unsigned short claimed;
    unsigned char *reply;
    if (msg_len < 3)
        return -1;
    claimed = (msg[1] << 8) | msg[2];
    if ((size_t)claimed + 3 > msg_len)
        return -1;
    reply = malloc(3 + (size_t)claimed);
    if (reply == NULL)
        return -1;
    reply[0] = MSG_ECHO_REPLY;
    reply[1] = msg[1];
    reply[2] = msg[2];
    memcpy(reply + 3, msg + 3, claimed);
    conn_write(c, reply, 3 + (size_t)claimed);
    free(reply);
    return 0;
}
""",
    ),
    _example(
        "One statement",
        "Adds two integers. It touches no memory at all.",
        r"""
int add(int a, int b)
{
    return a + b;
}
""",
    ),
    _example(
        "Checksum loop",
        "Adler-32 over a buffer: every read of data is inside the len bytes passed.",
        r"""
unsigned int checksum(const unsigned char *data, size_t len)
{
    unsigned int a = 1;
    unsigned int b = 0;
    size_t i;
    for (i = 0; i < len; i++) {
        a = (a + data[i]) % 65521;
        b = (b + a) % 65521;
    }
    return (b << 16) | a;
}
""",
    ),
)
"""The examples, weakness-bearing ones first."""


def examples() -> list[dict[str, Any]]:
    """The hand-written examples.

    Returns:
        One ``{name, note, code, group, lines}`` per example; ``lines`` is 1-indexed and
        empty for an example written with no weakness.
    """
    return [dict(example) for example in EXAMPLES]
