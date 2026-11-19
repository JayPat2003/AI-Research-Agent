"""Conversation memory: short-term Redis turns + long-term Postgres summaries.

Follow-up questions (e.g. \"what about kidney outcomes?\") are resolved against
a compressed conversation summary and a bounded window of recent turns, not a
raw transcript dump.
"""
