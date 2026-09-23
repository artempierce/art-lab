"""
tools/ — the actions agents can take, and the rules around taking them.

    registry.py   checks every tool call: does the tool exist, may this agent use it, is it safe to run
    untrusted.py  wraps outside text (documents, web pages) so the model treats it as data, not orders
    catalog.py    the list of every tool in the app and which agents may use it
"""
