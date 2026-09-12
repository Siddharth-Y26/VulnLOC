"""Pipeline inspector — a development instrument, outside the `vulnloc` layer graph.

A package only so `python frontend/server.py` can import `frontend.pipeline`. Nothing in
`src/vulnloc` imports anything from here, and nothing here is part of Phase 9's serving
deliverable. See `README.md`.
"""
