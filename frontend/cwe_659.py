"""MITRE's CWE-659 view, kept as reference data for the Findings response.

**Asked for on 2026-10-22**: the view in the backend, returned with ``/api/findings`` and
drawn nowhere on the page. **It is a reference list, not detection** — the site gains no
pattern from it. Each entry says whether a weakness pattern the service tested is filed
under that exact identifier (11 of the 97 when this was written, read from the request's
own ``patterns.checked``, never typed in). Every other entry is ``checked: false``: the site
never names that weakness, which says nothing about whether the code has it.

Identifiers are matched exactly, as ``docs/Memory.md`` §1.27 matches them: the pattern
filed under CWE-190 does not count as CWE-680 (Integer Overflow to Buffer Overflow), and
one filed under CWE-787 does not count as its parent CWE-119.

The identifiers and names are MITRE's, copied unedited from the view's CSV
(:data:`SOURCE`), retrieved 2026-10-22: 97 weaknesses, no categories.
"""

from __future__ import annotations

from typing import Any

__all__ = ["MEMBERS", "NOTE", "RETRIEVED", "SOURCE", "VIEW", "VIEW_NAME", "reference"]

VIEW = "CWE-659"
VIEW_NAME = "Weaknesses in Software Written in C++"
SOURCE = "https://cwe.mitre.org/data/csv/659.csv.zip"
RETRIEVED = "2026-10-22"

NOTE = (
    "Reference list, not detection. checked: true means a weakness pattern the site tested "
    "is filed under this exact identifier; checked: false means the site never names this "
    "weakness, not that the code is free of it. A pattern is a text match and matches "
    "harmless code too."
)
"""Travels with the list, so the JSON cannot be read as a claim of coverage on its own."""

MEMBERS: tuple[tuple[str, str], ...] = (
    ("CWE-14", "Compiler Removal of Code to Clear Buffers"),
    ("CWE-119", "Improper Restriction of Operations within the Bounds of a Memory Buffer"),
    ("CWE-120", "Buffer Copy without Checking Size of Input ('Classic Buffer Overflow')"),
    ("CWE-121", "Stack-based Buffer Overflow"),
    ("CWE-122", "Heap-based Buffer Overflow"),
    ("CWE-123", "Write-what-where Condition"),
    ("CWE-124", "Buffer Underwrite ('Buffer Underflow')"),
    ("CWE-125", "Out-of-bounds Read"),
    ("CWE-126", "Buffer Over-read"),
    ("CWE-127", "Buffer Under-read"),
    ("CWE-128", "Wrap-around Error"),
    ("CWE-129", "Improper Validation of Array Index"),
    ("CWE-130", "Improper Handling of Length Parameter Inconsistency"),
    ("CWE-131", "Incorrect Calculation of Buffer Size"),
    ("CWE-134", "Use of Externally-Controlled Format String"),
    ("CWE-135", "Incorrect Calculation of Multi-Byte String Length"),
    ("CWE-158", "Improper Neutralization of Null Byte or NUL Character"),
    ("CWE-170", "Improper Null Termination"),
    ("CWE-188", "Reliance on Data/Memory Layout"),
    ("CWE-191", "Integer Underflow (Wrap or Wraparound)"),
    ("CWE-192", "Integer Coercion Error"),
    ("CWE-194", "Unexpected Sign Extension"),
    ("CWE-195", "Signed to Unsigned Conversion Error"),
    ("CWE-196", "Unsigned to Signed Conversion Error"),
    ("CWE-197", "Numeric Truncation Error"),
    ("CWE-242", "Use of Inherently Dangerous Function"),
    ("CWE-243", "Creation of chroot Jail Without Changing Working Directory"),
    ("CWE-244", "Improper Clearing of Heap Memory Before Release ('Heap Inspection')"),
    ("CWE-248", "Uncaught Exception"),
    (
        "CWE-362",
        "Concurrent Execution using Shared Resource with Improper Synchronization "
        "('Race Condition')",
    ),
    ("CWE-364", "Signal Handler Race Condition"),
    ("CWE-366", "Race Condition within a Thread"),
    ("CWE-374", "Passing Mutable Objects to an Untrusted Method"),
    ("CWE-375", "Returning a Mutable Object to an Untrusted Caller"),
    ("CWE-396", "Declaration of Catch for Generic Exception"),
    ("CWE-397", "Declaration of Throws for Generic Exception"),
    ("CWE-401", "Missing Release of Memory after Effective Lifetime"),
    ("CWE-415", "Double Free"),
    ("CWE-416", "Use After Free"),
    ("CWE-457", "Use of Uninitialized Variable"),
    ("CWE-460", "Improper Cleanup on Thrown Exception"),
    ("CWE-462", "Duplicate Key in Associative List (Alist)"),
    ("CWE-463", "Deletion of Data Structure Sentinel"),
    ("CWE-464", "Addition of Data Structure Sentinel"),
    ("CWE-466", "Return of Pointer Value Outside of Expected Range"),
    ("CWE-467", "Use of sizeof() on a Pointer Type"),
    ("CWE-468", "Incorrect Pointer Scaling"),
    ("CWE-469", "Use of Pointer Subtraction to Determine Size"),
    ("CWE-476", "NULL Pointer Dereference"),
    ("CWE-478", "Missing Default Case in Multiple Condition Expression"),
    ("CWE-479", "Signal Handler Use of a Non-reentrant Function"),
    ("CWE-480", "Use of Incorrect Operator"),
    ("CWE-481", "Assigning instead of Comparing"),
    ("CWE-482", "Comparing instead of Assigning"),
    ("CWE-483", "Incorrect Block Delimitation"),
    ("CWE-484", "Omitted Break Statement in Switch"),
    ("CWE-493", "Critical Public Variable Without Final Modifier"),
    ("CWE-495", "Private Data Structure Returned From A Public Method"),
    ("CWE-496", "Public Data Assigned to Private Array-Typed Field"),
    ("CWE-498", "Cloneable Class Containing Sensitive Information"),
    ("CWE-500", "Public Static Field Not Marked Final"),
    ("CWE-543", "Use of Singleton Pattern Without Synchronization in a Multithreaded Context"),
    ("CWE-558", "Use of getlogin() in Multithreaded Application"),
    ("CWE-562", "Return of Stack Variable Address"),
    ("CWE-587", "Assignment of a Fixed Address to a Pointer"),
    ("CWE-588", "Attempt to Access Child of a Non-structure Pointer"),
    ("CWE-676", "Use of Potentially Dangerous Function"),
    ("CWE-680", "Integer Overflow to Buffer Overflow"),
    ("CWE-690", "Unchecked Return Value to NULL Pointer Dereference"),
    ("CWE-704", "Incorrect Type Conversion or Cast"),
    ("CWE-733", "Compiler Optimization Removal or Modification of Security-critical Code"),
    ("CWE-762", "Mismatched Memory Management Routines"),
    ("CWE-763", "Release of Invalid Pointer or Reference"),
    ("CWE-766", "Critical Data Element Declared Public"),
    ("CWE-767", "Access to Critical Private Variable via Public Method"),
    ("CWE-781", "Improper Address Validation in IOCTL with METHOD_NEITHER I/O Control Code"),
    ("CWE-782", "Exposed IOCTL with Insufficient Access Control"),
    ("CWE-783", "Operator Precedence Logic Error"),
    ("CWE-785", "Use of Path Manipulation Function without Maximum-sized Buffer"),
    ("CWE-786", "Access of Memory Location Before Start of Buffer"),
    ("CWE-787", "Out-of-bounds Write"),
    ("CWE-788", "Access of Memory Location After End of Buffer"),
    ("CWE-789", "Memory Allocation with Excessive Size Value"),
    ("CWE-805", "Buffer Access with Incorrect Length Value"),
    ("CWE-806", "Buffer Access Using Size of Source Buffer"),
    ("CWE-822", "Untrusted Pointer Dereference"),
    ("CWE-823", "Use of Out-of-range Pointer Offset"),
    ("CWE-824", "Access of Uninitialized Pointer"),
    ("CWE-825", "Expired Pointer Dereference"),
    ("CWE-839", "Numeric Range Comparison Without Minimum Check"),
    ("CWE-843", "Access of Resource Using Incompatible Type ('Type Confusion')"),
    ("CWE-910", "Use of Expired File Descriptor"),
    ("CWE-911", "Improper Update of Reference Count"),
    ("CWE-1325", "Improperly Controlled Sequential Memory Allocation"),
    ("CWE-1335", "Incorrect Bitwise Shift of Integer"),
    ("CWE-1341", "Multiple Releases of Same Resource or Handle"),
    (
        "CWE-1429",
        "Missing Security-Relevant Feedback for Unexecuted Operations in Hardware Interface",
    ),
)
"""``(identifier, MITRE's name)`` for every member of the view, in identifier order."""


def reference(checked: list[dict[str, str]], cards: list[dict[str, Any]]) -> dict[str, Any]:
    """The view, each entry marked with whether a pattern tested on this request names it.

    Args:
        checked: The patterns the service tested, ``{id, label, cwe}`` each — the report's
            ``patterns.checked``.
        cards: This request's Findings cards, most suspicious first.

    Returns:
        ``{view, name, source, retrieved, note, members, named_here}``. ``members`` holds one
        ``{cwe, name, checked, patterns}`` per entry; ``named_here`` the view's identifiers
        this request's cards name, once each, in card order.
    """
    patterns: dict[str, list[str]] = {}
    for pattern in checked:
        patterns.setdefault(pattern["cwe"], []).append(pattern["id"])
    in_view = {cwe for cwe, _ in MEMBERS}
    return {
        "view": VIEW,
        "name": VIEW_NAME,
        "source": SOURCE,
        "retrieved": RETRIEVED,
        "note": NOTE,
        "members": [
            {
                "cwe": cwe,
                "name": name,
                "checked": cwe in patterns,
                "patterns": patterns.get(cwe, []),
            }
            for cwe, name in MEMBERS
        ],
        "named_here": list(dict.fromkeys(card["cwe"] for card in cards if card["cwe"] in in_view)),
    }
