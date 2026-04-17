"""
Static messages and constants used across the agent.
"""

# Fallback messages — single source of truth for all user-facing rejection text.
MSG_OFF_TOPIC = (
    "I can only help with questions about our B2B retail platform "
    "(orders, returns, payments, account management, delivery, etc.)."
)
MSG_NO_CONTENT = (
    "No relevant content found for your question in the available documents."
)
MSG_NO_SCOPE = (
    "No content found in the database for country='{country}', language='{language}'."
)
MSG_FALLBACK_DEFAULT = "No content available for your request."
