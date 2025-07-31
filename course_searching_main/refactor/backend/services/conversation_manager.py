from datetime import datetime

class ConversationManager:
    def __init__(self):
        self.store = {}

    def get(self, conv_id: str):
        return self.store.setdefault(conv_id, {
            "messages": [], "previous_queries": [], "preferences": {}
        })

    def update(self, conv_id: str, user_query: str, bot_response: str, courses):
        conv = self.get(conv_id)
        conv["messages"].append({
            "user": user_query,
            "bot": bot_response,
            "timestamp": datetime.utcnow().isoformat()
        })
        conv["previous_queries"].append(user_query)
        q = user_query.lower()
        if "beginner" in q:
            conv["preferences"]["level"] = "beginner"
        elif "advanced" in q:
            conv["preferences"]["level"] = "advanced"